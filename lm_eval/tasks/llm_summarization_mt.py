"""
The Task is based on the newsum2021 Dataset for summarization
"""

import evaluate
import nltk
import numpy as np

from lm_eval.base import Task, rf
from lm_eval.fragments import Fragments
from lm_eval.metrics import mean

summarization_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


def _rouge_metric(predictions, references, rouge_type=None):
    result = summarization_metric.compute(
        predictions=[predictions],
        references=[references],
        use_stemmer=True,
    )[rouge_type]
    return result


def _rouge_agg(key, items):
    predictions, references = zip(*items)
    result = _rouge_metric(
        predictions=predictions, references=references, rouge_type=key
    )
    return result


def _bertscore_metric(predictions, references, key=None):
    assert key in ["precision", "recall", "f1", None]
    result = bertscore_metric.compute(
        predictions=[predictions],
        references=[references],
        model_type='microsoft/mdeberta-v3-base',
    )
    if key is None:
        return result
    return result[key]


class SummarizationTaskBase(Task):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "roysc/20minuten_sample_250"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None
    LANGUAGE = "german"
    TRAINING_SPLIT = "train"
    VALIDATION_SPLIT = "validation"
    TEST_SPLIT = "test"

    default_prompt_template = (
        f"Generate a summary in German for the following article. The summary should be around 3 to 5 sentences.\n"
        f"Article: {{article}}\n\nSummary:\n")

    def __init__(self, *args, **kwargs):
        super(SummarizationTaskBase, self).__init__(*args, **kwargs)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
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
        if self.prompt_template:
            prompt = self.prompt_template.format(article=doc['article'])
        else:
            prompt = self.default_prompt_template.format(article=doc['article'])
        return prompt

    def doc_to_target(self, doc):
        summary = doc["summary"]
        return " " + summary

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        return continuation

    def postprocess_text(self, prediction, reference):
        prediction = prediction.strip()
        reference = reference.strip()

        # rougeLSum expects newline after each sentence
        prediction = "\n".join(nltk.sent_tokenize(prediction, language=self.LANGUAGE))
        reference = "\n".join(nltk.sent_tokenize(reference, language=self.LANGUAGE))
        return prediction, reference

    def round_to_3_decimals(self, value: float) -> float:
        return round(value, 3)

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: Make this work for multiple results
        assert len(results) == 1

        prediction, reference = self.postprocess_text(results[0], doc["summary"])
        article = doc["article"]
        fragment = Fragments(article, prediction, language=self.LANGUAGE)

        bertscore_result = _bertscore_metric(prediction, reference)

        return {
            "rouge1": self.round_to_3_decimals(_rouge_metric(prediction, reference, "rouge1")),
            "rouge2": self.round_to_3_decimals(_rouge_metric(prediction, reference, "rouge2")),
            "rougeL": self.round_to_3_decimals(_rouge_metric(prediction, reference, "rougeL")),
            "bertscore_precision": self.round_to_3_decimals(np.mean(bertscore_result["precision"])),
            "bertscore_recall": self.round_to_3_decimals(np.mean(bertscore_result["recall"])),
            "bertscore_f1": self.round_to_3_decimals(np.mean(bertscore_result["f1"])),
            'coverage': self.round_to_3_decimals(fragment.coverage()),
            'density': self.round_to_3_decimals(fragment.density()),
            'compression': self.round_to_3_decimals(fragment.compression())
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            "rouge1": mean,
            "rouge2": mean,
            "rougeL": mean,
            "bertscore_precision": mean,
            "bertscore_recall": mean,
            "bertscore_f1": mean,
            'coverage': mean,
            'density': mean,
            'compression': mean
        }

    def higher_is_better(self):
        """
                :returns: {str: bool}
                    A dictionary where keys are the names of submetrics and values are
                    whether a higher value of the submetric is better
                """
        return {
            "rouge1": True,
            "rouge2": True,
            "rougeL": True,
            "bertscore_precision": True,
            "bertscore_recall": True,
            "bertscore_f1": True,
            "coverage": False,
            "density": False,
            "compression": True,
        }


class SummarizationTask_20Minuten(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten"
    # DATASET_PATH = "roysc/20minuten_sample_250"


class SummLtM_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20minuten_sample_250"

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": ["\n[TASK]", "\n[END]"]})
        return cont_request

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: Make this work for multiple results
        # take the last result for score processing if there are multiple ones
        if len(results) > 1:
            results = [results[-1]]
        assert len(results) == 1

        prediction, reference = self.postprocess_text(results[0], doc["summary"])
        article = doc["article"]
        fragment = Fragments(article, prediction, language=self.LANGUAGE)

        bertscore_result = _bertscore_metric(prediction, reference)

        return {
            "rouge1": self.round_to_3_decimals(_rouge_metric(prediction, reference, "rouge1")),
            "rouge2": self.round_to_3_decimals(_rouge_metric(prediction, reference, "rouge2")),
            "rougeL": self.round_to_3_decimals(_rouge_metric(prediction, reference, "rougeL")),
            "bertscore_precision": self.round_to_3_decimals(np.mean(bertscore_result["precision"])),
            "bertscore_recall": self.round_to_3_decimals(np.mean(bertscore_result["recall"])),
            "bertscore_f1": self.round_to_3_decimals(np.mean(bertscore_result["f1"])),
            'coverage': self.round_to_3_decimals(fragment.coverage()),
            'density': self.round_to_3_decimals(fragment.density()),
            'compression': self.round_to_3_decimals(fragment.compression())
        }


class SummLtMDe_20Minuten(SummLtM_20Minuten):
    VERSION = 0

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": ["\n"]})
        return cont_request


class SummarizationTask_Klexikon(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/Klexikon_sample_250"
