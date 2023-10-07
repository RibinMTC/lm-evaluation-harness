from functools import partial
import evaluate
import numpy as np
import wandb

from lm_eval.base import Task, rf
from lm_eval.metrics import complex_metric_agg
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
                self._training_docs = self.dataset["train"]
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
        label = str(doc[self.label_key_name])
        return label

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
                "f1": (prediction, truth),
                "precision": (prediction, truth),
                "recall": (prediction, truth),
                "roc_auc": (true_prediction_probability, truth)
                }

    def aggregation(self):
        return {
            "bacc": partial(
                complex_metric_agg, "bacc"
            ),
            "f1": partial(
                complex_metric_agg, "f1"
            ),
            "precision": partial(
                complex_metric_agg, "precision"
            ),
            "recall": partial(
                complex_metric_agg, "recall"
            ),
            "roc_auc": partial(
                _roc_auc_agg
            )
        }

    def higher_is_better(self):
        return {"bacc": True,
                "f1": True,
                "precision": True,
                "recall": True,
                "roc_auc": True}

    def get_plots(self):
        task_results_info_df = pd.DataFrame(self.task_results_info_list)
        predictions = task_results_info_df["prediction"].tolist()
        references = task_results_info_df["reference"].tolist()
        class_names = ["Unfaithful", "Faithful"]
        all_plots = {"confusion_matrix": wandb.plot.confusion_matrix(y_true=references, preds=predictions,
                                                                     class_names=class_names)}
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
    true_label_name = "Faithful"


class FaithfulnessClassificationTaskFinalSwissText23BenchmarkIntrinsic(
    FaithfulnessClassificationTaskFinalSwissText23Benchmark):
    true_label_name = "Intrinsic Hallucination"


class FaithfulnessClassificationTaskFinalSwissText23BenchmarkExtrinsic(
    FaithfulnessClassificationTaskFinalSwissText23Benchmark):
    true_label_name = "Extrinsic Hallucination"


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
