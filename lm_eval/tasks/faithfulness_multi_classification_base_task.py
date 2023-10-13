from functools import partial
from typing import List

import numpy as np
import wandb
from datasets import Dataset

from lm_eval.metrics import complex_metric_agg
from lm_eval.tasks.base_plotter import Plotter
from lm_eval.base import MultipleChoiceTask
import pandas as pd


class FaithfulnessMultiClassificationBaseTask(MultipleChoiceTask, Plotter):
    VERSION = 0
    DATASET_PATH = "mtc/final_german_faithfulness_benchmark"
    label_key_name = "label"
    positive_label = "true"
    negative_label = "false"
    language = "German"
    article_key_name = "lead_with_article"
    sentence_key_name = "text"

    choices = ["Faithful", "Intrinsic Hallucination", "Extrinsic Hallucination"]

    negative_samples_df = None
    positive_samples_df = None

    default_prompt_template = (
        "Analyze whether the given sentence is faithful to the article. If the sentence solely conveys information "
        "that comes directly from the article, without any additions or omissions, respond with 'Faithful'. If the "
        "sentence contains information that is in direct contradiction to the article, respond with 'Intrinsic "
        "Hallucination'. If the sentence introduces information or details that are not explicitly mentioned in the "
        "article itself, respond with 'Extrinsic Hallucination'.\nArticle: {article}\nSentence: {sentence}\nLabel:\n"

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
                train_df = self.dataset["train"].to_pandas()
                train_df["num_words_article"] = train_df["lead_with_article"].str.len()
                sorted_train_df = train_df.sort_values(by="num_words_article", ascending=True)
                faithful_samples_df = sorted_train_df.loc[
                    lambda example: example["label"] == self.choices[0]].head(100)
                intrinsic_samples_df = sorted_train_df.loc[
                    lambda example: example["label"] == self.choices[1]].head(100)
                extrinsic_samples_df = sorted_train_df.loc[
                    lambda example: example["label"] == self.choices[2]].head(100)
                self._training_docs = {
                    "faithful": Dataset.from_pandas(faithful_samples_df),
                    "intrinsic": Dataset.from_pandas(intrinsic_samples_df),
                    "extrinsic": Dataset.from_pandas(extrinsic_samples_df),
                    "full": sorted_train_df
                }

            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc["query"]

    def format_prompt(self, doc):
        if not self.prompt_template:
            self.prompt_template = self.default_prompt_template
        prompt = self.prompt_template.format(article=doc[self.article_key_name],
                                             sentence=doc[self.sentence_key_name])
        return prompt

    def format_prompt_target(self, doc):
        return " " + doc["label"]

    def get_article_ids_with_at_least_all_labels(self) -> List[int]:
        grouped_df = self._training_docs["full"].groupby("article_id")["label"].nunique()
        valid_article_ids = grouped_df[grouped_df >= 3].index
        return list(valid_article_ids)

    def _format_packed_examples(self, doc, rnd=None, num_articles: int = 0, num_samples_per_article_per_label=None):
        """
        Format the examples using the 'packed' strategy.

        :param examples: List of document examples.
        :return: A string representing the formatted examples.
        """

        sorted_full_dataset = self._training_docs["full"]
        valid_article_ids = self.get_article_ids_with_at_least_all_labels()[:3]
        valid_dataset = sorted_full_dataset[sorted_full_dataset["article_id"].isin(valid_article_ids)].sort_values(
            by='num_words_article', ascending=True)
        selected_article_ids = rnd.sample(valid_article_ids, num_articles)
        examples_per_articles = []
        seed = 42
        for article_id in selected_article_ids:
            article_id_samples = sorted_full_dataset[sorted_full_dataset["article_id"] == article_id]
            samples = []
            for choice in self.choices:
                selected_choice_samples = article_id_samples.loc[
                    lambda example: example["label"] == choice]
                selected_samples = selected_choice_samples.sample(n=num_samples_per_article_per_label,
                                                                  random_state=seed)
                samples.append(selected_samples)

            examples_per_articles.append(pd.concat(samples, ignore_index=True))

        doc = {key: [value] for key, value in doc["original_doc"].items()}
        examples_per_articles.append(pd.DataFrame(doc))
        formatted_examples = []
        for idx, example_per_article in enumerate(examples_per_articles, start=1):
            example_text = f"### Example {idx}\n"
            example_text += f"Article: {example_per_article.iloc[0][self.article_key_name]}\n"
            last_sample = example_per_article.shape[0] == 1
            # Iterate through sentences and labels in the example
            for index, example in example_per_article.iterrows():
                example_text += f"Sentence: {example[self.sentence_key_name]}\n"
                if last_sample:
                    label_text = ""
                else:
                    label_text = example[self.label_key_name]
                example_text += f"Label: {label_text}\n"
            formatted_examples.append(example_text)

        return "\n".join(formatted_examples) + "\n\n"

    def _format_default_examples(self, rnd, num_fewshot, num_samples_per_class, doc):
        """
        Format the examples using the default strategy.

        :param examples: List of document examples.
        :return: A string representing the formatted examples.
        """
        # Randomly sample positive and negative examples
        faithful_examples = rnd.sample(self._training_docs["faithful"].to_list(), num_samples_per_class)
        intrinsic_examples = rnd.sample(self._training_docs["intrinsic"].to_list(), num_samples_per_class)
        extrinsic_examples = rnd.sample(self._training_docs["extrinsic"].to_list(), num_samples_per_class)

        # Merge and filter out the current document
        combined_examples = faithful_examples + intrinsic_examples + extrinsic_examples
        unique_examples = [example for example in combined_examples if example != doc][:num_fewshot]
        formatted_examples = [
            self.format_prompt(example) + self.format_prompt_target(example) for example in unique_examples
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
            full_prompt = f"{doc['query']}"
        else:
            # Raise an error if no training documents are available
            if not self.has_training_docs():
                raise ValueError("Training documents are required for few-shot prompting!")

            self._training_docs = self.training_docs()  # Load training documents

            # Define the number of positive and negative samples based on the fewshot_sampling strategy
            if fewshot_sampling == "stratified":
                assert num_fewshot % 3 == 0, ("When selecting stratified strategy, num_fewshot has to be multiple of 3 "
                                              "for the multi-labeling task")
                num_samples_per_class = num_fewshot // 3
                labeled_examples = self._format_default_examples(rnd=rnd, num_fewshot=num_fewshot,
                                                                 num_samples_per_class=num_samples_per_class, doc=doc)
                full_prompt = f"{description}{labeled_examples}\n{doc['query']}"
            elif fewshot_sampling == "packed":
                # raise NotImplementedError
                assert num_fewshot > 5 and num_fewshot % 3 == 0, (f"{num_fewshot} has to be larger than 6 and a "
                                                                  f"multiple of 3 for fewshot_sampling strategy packed")
                num_articles = num_fewshot // 3
                labeled_examples_with_doc = self._format_packed_examples(num_samples_per_article_per_label=1,
                                                                         num_articles=num_articles, doc=doc, rnd=rnd)
                task_only_text = self.prompt_template.split("\n")[0]
                labeled_examples = task_only_text + "\n" + labeled_examples_with_doc
                full_prompt = f"{description}{labeled_examples}"
            else:  # Default or "packed" strategy
                raise ValueError(f"Unsupported fewshot sampling strategy: {fewshot_sampling}")

        return full_prompt

    def _process_doc(self, doc):
        out_doc = {
            "query": self.format_prompt(doc),
            "choices": self.choices,
            "gold": self.choices.index(doc["label"]),
            "original_doc": doc
        }
        return out_doc

    def process_results(self, doc, results):
        prediction = np.argmax(results)
        # sklearn documentation: roc prediction probability corresponds to the probability of the class with the
        # greater label(=1)
        results_probabilities = np.exp(results)
        truth = doc["gold"]
        # true_prediction_probability = results_probabilities[truth] / np.sum(results_probabilities)
        self.task_results_info_list.append({"prediction": prediction, "reference": truth})
        print(f"Results: {results}, Prediction {prediction}, Truth: {truth}")
        return {"bacc": (prediction, truth),
                "f1_macro": (prediction, truth),
                "f1_all": (prediction, truth),
                "f1_micro": (prediction, truth),
                "precision_macro": (prediction, truth),
                "recall_macro": (prediction, truth),
                # "roc_auc": (true_prediction_probability, truth)
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
                complex_metric_agg, "f1_all"
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
            # "roc_auc": partial(
            #     _roc_auc_agg
            # )
        }

    def higher_is_better(self):
        return {"bacc": True,
                "f1_macro": True,
                "f1_all": True,
                "f1_micro": True,
                "precision_macro": True,
                "recall_macro": True,
                # "roc_auc": True
                }

    def get_plots(self):
        task_results_info_df = pd.DataFrame(self.task_results_info_list)
        predictions = task_results_info_df["prediction"].tolist()
        references = task_results_info_df["reference"].tolist()
        all_plots = {"confusion_matrix": wandb.plot.confusion_matrix(y_true=references, preds=predictions,
                                                                     class_names=self.choices)}
        return all_plots
