from functools import partial

import evaluate
import numpy as np

from lm_eval.base import Task, rf
from lm_eval.metrics import complex_metric_agg

roc_auc_metric = evaluate.load("roc_auc")


def _roc_auc_agg(items):
    predictions, references = zip(*items)
    result = roc_auc_metric.compute(
        prediction_scores=predictions, references=references
    )['roc_auc']
    return result


class SwissText23FaithfulnessClassificationTask(Task):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "mtc/swisstext23-20min-gold_annotation_train_test_data"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    positive_label = "faithful"  # True Yes
    negative_label = "unfaithful"  # False No

    default_prompt_template = ("### System:\nYou are StableBeluga, an AI that follows instructions extremely well. "
                               "Help as much as you can.\n\n### User: You'll be given a German sentence and a German "
                               "article. Your task is to determine if the sentence accurately reflects the content of "
                               "the article without adding any unmentioned details or contradicting any information "
                               "from the article. It's not necessary for the sentence to cover all the main ideas of "
                               "the article, just that the details it does mention are correctly derived from the "
                               "text. Provide your judgment by outputting only 'faithful' if the sentence is "
                               "faithful, and 'unfaithful' if it's not.\nArticle: {article}\nSentence: {"
                               "sentence}\n\n### Assistant:\n")


    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():

            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
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
        prompt = self.prompt_template.format(article=doc['article_with_lead'],
                                             sentence=doc['text'], label="")
        return prompt

    def doc_to_target(self, doc):
        label = str(doc["label"] == 1)
        return " " + label

    def construct_requests(self, doc, ctx):
        ll_false, _ = rf.loglikelihood(ctx, f" {self.negative_label}")
        ll_true, _ = rf.loglikelihood(ctx, f" {self.positive_label}")
        return ll_false, ll_true


    def process_results(self, doc, results):
        prediction = np.argmax(results)
        # sklearn documentation: roc prediction probability corresponds to the probability of the class with the
        # greater label(=1)
        results_probabilities = np.exp(results)
        true_prediction_probability = results_probabilities[1] / np.sum(results_probabilities)
        truth = doc["label"]
        print(f"Results: {results}, Prediction {prediction}, Truth: {truth}")
        return {"bacc": (prediction, truth),
                "f1": (prediction, truth),
                "precision": (prediction, truth),
                "recall": (prediction, truth),
                "roc_auc": (true_prediction_probability, truth)}

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
