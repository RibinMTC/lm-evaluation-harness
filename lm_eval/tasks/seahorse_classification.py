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


class SeahorseClassificationTask(Task):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "mtc/german_seahorse_dataset_with_articles"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    tgt_question = 'question4'

    worker_lang_to_language_dict = {
        "en-US": "English",
        "es-ES": "Spanish",
        "ru": "Russian",
        "vi": "Vietnamese",
        "tr": "Turkish",
        "de": "German"
    }

    default_prompt_template = """### System:
You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can.

### User: Given an article and its corresponding summary in {lang}, reply with True if all information in the summary originates from the article and reply False otherwise. 
Article: {article}
Summary: {summary}

### Assistant:"
"""

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
            test_set = self.dataset["test"]
            hq_test_set_german = test_set.filter(lambda example: example['question1'] == 'Yes')
            print(f"Num of samples: {hq_test_set_german.num_rows}")
            test = hq_test_set_german.filter(lambda example: example[self.tgt_question] == "Yes")
            print(
                f"Num of positive samples: {test.num_rows}. Num of negative samples: {hq_test_set_german.num_rows - test.num_rows}")
            return hq_test_set_german

    def doc_to_text(self, doc):
        if not self.prompt_template:
            self.prompt_template = self.default_prompt_template
        prompt = self.prompt_template.format(article=doc['article'],
                                             summary=doc['summary'], label="",
                                             lang=self.worker_lang_to_language_dict[doc['worker_lang']])
        return prompt

    def doc_to_target(self, doc):
        label = str(doc[self.tgt_question] == "Yes")
        return " " + label

    def construct_requests(self, doc, ctx):
        ll_false, _ = rf.loglikelihood(ctx, " False")
        ll_true, _ = rf.loglikelihood(ctx, " True")
        return ll_false, ll_true

    @staticmethod
    def convert_label(label):
        label = label.strip()
        if label.lower() == 'false':
            return 0
        elif label.lower() == 'true':
            return 1
        else:
            raise ValueError("Invalid label!")

    def process_results(self, doc, results):
        prediction = np.argmax(results)
        # sklearn documentation: roc prediction probability corresponds to the probability of the class with the
        # greater label(=1)
        results_probabilities = np.exp(results)
        true_prediction_probability = results_probabilities[1]/np.sum(results_probabilities)
        truth = self.convert_label(self.doc_to_target(doc))
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

class SeahorseClassificationTask_comprehensible(SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_TEST"
    tgt_question = 'question1'

class SeahorseClassificationTask_repetition(SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_TEST"
    tgt_question = 'question2'

class SeahorseClassificationTask_grammar(SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_TEST"
    tgt_question = 'question3'
class SeahorseClassificationTask_attribution(SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_TEST"
    tgt_question = 'question4'

class SeahorseClassificationTask_main_ideas(SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_TEST"
    tgt_question = 'question5'

class SeahorseClassificationTask_conciseness(SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_TEST"
    tgt_question = 'question6'