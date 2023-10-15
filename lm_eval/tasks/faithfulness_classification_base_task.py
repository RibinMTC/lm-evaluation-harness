import math
from functools import partial
from typing import List

import evaluate
import numpy as np
import wandb
from datasets import Dataset

from lm_eval.base import Task, rf
from lm_eval.metrics import complex_metric_agg, complex_metric_agg_with_class_labels
from lm_eval.tasks.base_plotter import Plotter
import pandas as pd

roc_auc_metric = evaluate.load("roc_auc")


def _roc_auc_agg(items):
    predictions, references = zip(*items)
    try:
        result = roc_auc_metric.compute(
            prediction_scores=predictions, references=references
        )['roc_auc']
    except ValueError:
        print("Only one class. Cannot compute roc score")
        return 0
    return result


class FaithfulnessClassificationBaseTask(Task, Plotter):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None

    article_key_name = None
    summary_key_name = None
    language = ""
    label_key_name = "is_faithful"
    true_label_name = None

    positive_label = "faithful"
    negative_label = "unfaithful"

    plot_class_names = [negative_label, positive_label]

    negative_samples_df = None
    positive_samples_df = None

    default_prompt_template = (
        "You'll be given a {language} sentence and a {language} article. Your task is to determine if "
        "the sentence accurately reflects the content of the article without adding any "
        "unmentioned details or contradicting any information from the article. It's not "
        "necessary for the sentence to cover all the main ideas of the article, just that the "
        "details it does mention are correctly derived from the text. Provide your judgment by "
        "outputting only 'faithful' if the sentence is faithful, and 'unfaithful' if it's "
        "not.\nArticle: {article}\nSentence: {sentence}\nLabel:\n"
    )

    task_results_info_list = []

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                if not self.true_label_name:
                    self._training_docs = self.dataset["train"]
                else:
                    train_df = self.dataset["train"].to_pandas()
                    train_df["num_words_article"] = train_df["lead_with_article"].str.len()
                    sorted_train_df = train_df.sort_values(by="num_words_article", ascending=True)
                    negative_samples_df = sorted_train_df.loc[
                        lambda example: example["label"] != self.true_label_name].head(100)
                    positive_samples_df = sorted_train_df.loc[
                        lambda example: example["label"] == self.true_label_name].head(100)
                    self._training_docs = {
                        "positive": Dataset.from_pandas(positive_samples_df),
                        "negative": Dataset.from_pandas(negative_samples_df),
                        "full": sorted_train_df
                    }
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        if not self.prompt_template:
            self.prompt_template = self.default_prompt_template
        prompt = self.prompt_template.format(article=doc[self.article_key_name],
                                             sentence=doc[self.summary_key_name], label="", language=self.language)
        return prompt

    def doc_to_target(self, doc):
        if doc[self.label_key_name] == self.true_label_name:
            return self.positive_label
        else:
            return self.negative_label

    def get_unique_elements_with_preserved_order(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    def get_num_samples_per_article_from_num_articles_and_num_fewshot(self, num_fewshot: int, num_articles: int) -> \
    List[int]:
        # Step 1: Divide the total by the number of slots.
        basic_share = num_fewshot / num_articles

        # Step 2: Initialize a list to hold the distribution and calculate the basic share for each slot (rounding down).
        num_samples_per_article = [math.floor(basic_share) for _ in range(num_articles)]

        # Step 3: Calculate the remainder that needs to be distributed.
        remainder = num_fewshot - sum(num_samples_per_article)

        # Step 4: Calculate the remainders for each slot.
        remainders = [(index, basic_share - math.floor(basic_share)) for index in range(num_articles)]

        # Step 5: Sort the slots by the largest remainders.
        remainders.sort(key=lambda x: -x[1])

        # Step 6: Distribute the remainder to the slots with the largest remainders.
        for i in range(int(remainder)):
            num_samples_per_article[remainders[i][0]] += 1

        return num_samples_per_article

    def _format_packed_examples(self, doc, num_samples_per_articles: List[int], rnd=None):
        """
        Format the examples using the 'packed' strategy.

        :param examples: List of document examples.
        :return: A string representing the formatted examples.
        """
        assert doc is not None and num_samples_per_articles is not None, "doc and num_samples_per_article cannot be None at the same time"

        num_articles = len(num_samples_per_articles)
        sorted_full_dataset = self._training_docs["full"]
        sorted_unique_article_ids = self.get_unique_elements_with_preserved_order(sorted_full_dataset["article_id"])[
                                    :20]
        selected_article_ids = rnd.sample(sorted_unique_article_ids, num_articles)
        examples_per_articles = []
        for article_id, num_samples_per_article in zip(selected_article_ids, num_samples_per_articles):
            num_pos_samples = math.ceil(num_samples_per_article / 2)
            num_neg_samples = num_samples_per_article - num_pos_samples
            article_id_samples = sorted_full_dataset[sorted_full_dataset["article_id"] == article_id]
            negative_samples_df = article_id_samples.loc[
                lambda example: example["label"] != self.true_label_name]
            positive_samples_df = article_id_samples.loc[
                lambda example: example["label"] == self.true_label_name]
            seed = 42
            positive_examples = positive_samples_df.sample(n=num_pos_samples, random_state=seed)
            negative_examples = negative_samples_df.sample(n=num_neg_samples, random_state=seed)
            examples_per_articles.append(
                pd.concat([positive_examples, negative_examples]).sort_index().reset_index(drop=True))
        doc = {key: [value] for key, value in doc.items()}
        examples_per_articles.append(pd.DataFrame(doc))
        formatted_examples = []
        for idx, example_per_article in enumerate(examples_per_articles, start=1):
            example_text = f"### Example {idx}\n"
            example_text += f"Article: {example_per_article.iloc[0][self.article_key_name]}\n"
            last_sample = example_per_article.shape[0] == 1
            # Iterate through sentences and labels in the example
            for index, example in example_per_article.iterrows():
                example_text += f"Sentence: {example[self.summary_key_name]}\n"
                if last_sample:
                    label_text = ""
                else:
                    label_text = 'true' if example[self.label_key_name] == self.true_label_name else 'false'
                example_text += f"Label: {label_text}\n"
            formatted_examples.append(example_text)

        return "\n".join(formatted_examples) + "\n\n"

    def _format_default_examples(self, rnd, num_fewshot, num_pos_samples, num_neg_samples, doc):
        """
        Format the examples using the default strategy.

        :param examples: List of document examples.
        :return: A string representing the formatted examples.
        """
        # Randomly sample positive and negative examples
        positive_examples = rnd.sample(self._training_docs["positive"].to_list(), num_pos_samples)
        negative_examples = rnd.sample(self._training_docs["negative"].to_list(), num_neg_samples)

        # Merge and filter out the current document
        combined_examples = positive_examples + negative_examples
        unique_examples = [example for example in combined_examples if example != doc][:num_fewshot]
        rnd.shuffle(unique_examples)
        formatted_examples = [
            self.doc_to_text(example) + self.doc_to_target(example) for example in unique_examples
        ]
        return "\n\n".join(formatted_examples) + "\n\n"

    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None,
                        description=None, fewshot_sampling: str = None):
        # Ensure a random generator is provided
        if rnd is None:
            raise ValueError("A `random.Random` generator argument must be provided to `rnd`")

        # Warn about the deprecation of 'provide_description'
        if provide_description is not None:
            print("WARNING: 'provide_description' is deprecated. Use 'description' instead.")

        # Add a newline after the description if it's provided
        description = f"{description}\n\n" if description else ""

        # Handle case with no few-shot examples
        if num_fewshot == 0:
            labeled_examples = ""
            prompt_for_current_doc = self.doc_to_text(doc)
        else:
            # Raise an error if no training documents are available
            if not self.has_training_docs():
                raise ValueError("Training documents are required for few-shot prompting!")

            self._training_docs = self.training_docs()  # Load training documents

            # Define the number of positive and negative samples based on the fewshot_sampling strategy
            if fewshot_sampling == "stratified":
                num_pos_samples = num_fewshot // 2
                num_neg_samples = num_fewshot - num_pos_samples
            elif fewshot_sampling in ["positive_only", "negative_only"]:
                num_pos_samples = num_fewshot if fewshot_sampling == "positive_only" else 0
                num_neg_samples = num_fewshot - num_pos_samples
            elif fewshot_sampling == "packed":
                assert num_fewshot > 7, f"{num_fewshot} has to be larger than 8 for fewshot_sampling strategy packed"
                # we want to at maximum 8 samples per article
                num_articles = math.ceil(num_fewshot / 8)
                num_samples_per_articles = self.get_num_samples_per_article_from_num_articles_and_num_fewshot(
                    num_fewshot=num_fewshot, num_articles=num_articles)
            else:  # Default or "packed" strategy
                num_pos_samples = rnd.randint(0, num_fewshot)
                num_neg_samples = num_fewshot - num_pos_samples

            # Format the examples based on the 'packed' strategy or default
            if fewshot_sampling == "packed":
                labeled_examples_with_doc = self._format_packed_examples(doc=doc,
                                                                         num_samples_per_articles=num_samples_per_articles,
                                                                         rnd=rnd)
                prompt_for_current_doc = ""
                task_only_text = self.prompt_template.split("\n")[0]
                labeled_examples = task_only_text + "\n" + labeled_examples_with_doc
            else:
                labeled_examples = self._format_default_examples(rnd=rnd, num_fewshot=num_fewshot,
                                                                 num_pos_samples=num_pos_samples,
                                                                 num_neg_samples=num_neg_samples, doc=doc)
                prompt_for_current_doc = self.doc_to_text(doc)

        full_prompt = f"{description}{labeled_examples}{prompt_for_current_doc}"

        return full_prompt

    def construct_requests(self, doc, ctx):
        ll_false, _ = rf.loglikelihood(ctx, f" {self.negative_label}")
        ll_true, _ = rf.loglikelihood(ctx, f" {self.positive_label}")
        return ll_false, ll_true

    def convert_label(self, label) -> int:
        if self.true_label_name:
            return int(label == self.true_label_name)
        else:
            return int(label)

    def process_results(self, doc, results):
        prediction = np.argmax(results)
        # sklearn documentation: roc prediction probability corresponds to the probability of the class with the
        # greater label(=1)
        results_probabilities = np.exp(results)
        true_prediction_probability = results_probabilities[1] / np.sum(results_probabilities)
        truth = self.convert_label(doc[self.label_key_name])
        self.task_results_info_list.append({"prediction": prediction, "reference": truth})
        print(f"Results: {results}, Prediction {prediction}, Truth: {truth}")
        return {"bacc": (prediction, truth),
                "f1_macro": (prediction, truth),
                "f1_all": (prediction, truth),
                "f1_micro": (prediction, truth),
                "precision_macro": (prediction, truth),
                "recall_macro": (prediction, truth),
                "roc_auc": (true_prediction_probability, truth)
                }

    def aggregation(self):
        return {
            "bacc": partial(
                complex_metric_agg, "bacc"
            ),
            "f1_macro": partial(
                complex_metric_agg, "f1_macro"
            ),
            "f1_all": partial(
                complex_metric_agg_with_class_labels, "f1_all", self.plot_class_names
            ),
            "f1_micro": partial(
                complex_metric_agg, "f1_micro"
            ),
            "precision_macro": partial(
                complex_metric_agg, "precision_macro"
            ),
            "recall_macro": partial(
                complex_metric_agg, "recall_macro"
            ),
            "roc_auc": partial(
                _roc_auc_agg
            )
        }

    def higher_is_better(self):
        return {"bacc": True,
                "f1_macro": True,
                "f1_all": True,
                "f1_micro": True,
                "precision_macro": True,
                "recall_macro": True,
                "roc_auc": True
                }

    def get_plots(self):
        task_results_info_df = pd.DataFrame(self.task_results_info_list)
        predictions = task_results_info_df["prediction"].tolist()
        references = task_results_info_df["reference"].tolist()
        all_plots = {"confusion_matrix": wandb.plot.confusion_matrix(y_true=references, preds=predictions,
                                                                     class_names=self.plot_class_names)}
        return all_plots


class FaithfulnessClassificationTaskFactCC(FaithfulnessClassificationBaseTask):
    DATASET_PATH = "mtc/faithfulness_benchmark_sanity_check_factcc"
    language = "English"
    article_key_name = "text"
    summary_key_name = "claim"


class FaithfulnessClassificationTaskFrank(FaithfulnessClassificationBaseTask):
    DATASET_PATH = "mtc/faithfulness_benchmark_sanity_check_frank"
    language = "English"
    article_key_name = "article"
    summary_key_name = "summary"


class FaithfulnessClassificationTaskSwissText23GoldAnnotation(FaithfulnessClassificationBaseTask):
    DATASET_PATH = "mtc/faithfulness_benchmark_sanity_check_gold_annotation"
    language = "German"
    article_key_name = "article_with_lead"
    summary_key_name = "text"


class FaithfulnessClassificationTaskFinalSwissText23Benchmark(FaithfulnessClassificationBaseTask):
    DATASET_PATH = "mtc/final_german_faithfulness_benchmark"
    label_key_name = "label"
    positive_label = "true"
    negative_label = "false"
    language = "German"
    article_key_name = "lead_with_article"
    summary_key_name = "text"


class FaithfulnessClassificationTaskFinalSwissText23BenchmarkFaithful(
    FaithfulnessClassificationTaskFinalSwissText23Benchmark):
    positive_label = "Faithful"
    negative_label = "Hallucination"
    true_label_name = positive_label
    plot_class_names = [negative_label, positive_label]


class FaithfulnessClassificationTaskFinalSwissText23BenchmarkIntrinsic(
    FaithfulnessClassificationTaskFinalSwissText23Benchmark):
    true_label_name = "Intrinsic Hallucination"
    plot_class_names = ["Not Intrinsic", "Intrinsic"]


class FaithfulnessClassificationTaskFinalSwissText23BenchmarkExtrinsic(
    FaithfulnessClassificationTaskFinalSwissText23Benchmark):
    true_label_name = "Extrinsic Hallucination"
    plot_class_names = ["Not Extrinsic", "Extrinsic"]


class FaithfulnessClassificationTaskExtrinsicOnlySwissText23GoldAnnotation(
    FaithfulnessClassificationTaskSwissText23GoldAnnotation):
    DATASET_PATH = "mtc/faithfulness_benchmark_sanity_check_extrinsic_only_gold_annotation"
    positive_label = "true"
    negative_label = "false"
    default_prompt_template = (
        "Your task is to carefully read an article and a given sentence. You should classify whether the sentence "
        "contains any information, fact, or detail that is not mentioned in the article. If the sentence adds no new "
        "information and only summarizes or rephrases the contents of the article, you should return 'true'. "
        "However, if the sentence introduces any new information or details, even if they are factually correct but "
        "are not found in the article, you should return 'false'.\nArticle: {article}\nSentence: {"
        "sentence}\nLabel:\n"
    )


class FaithfulnessClassificationTaskIntrinsicOnlySwissText23GoldAnnotation(
    FaithfulnessClassificationTaskSwissText23GoldAnnotation):
    DATASET_PATH = "mtc/faithfulness_benchmark_sanity_check_intrinsic_only_gold_annotation"


class FaithfulnessClassificationTaskXsumFaith(FaithfulnessClassificationBaseTask):
    DATASET_PATH = "mtc/faithfulness_benchmark_sanity_check_xsum_faith"
    language = "English"
    article_key_name = "text"
    summary_key_name = "summary"
