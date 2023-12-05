"""
The Task is based on the newsum2021 Dataset for summarization
"""

import evaluate
import nltk
import numpy as np
import json

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


# TODO: Include BartScore???
# from transformers import MBartForConditionalGeneration, MBart50Tokenizer
# self.tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang=YOUR_SRC_LANG, tgt_lang=YOUR_TGT_LANG)
# self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")


class SummarizationTaskBase(Task):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "roysc/20minuten_sample_250"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None
    LANGUAGE = "german"
    TRAINING_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
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
        continuation = rf.greedy_until(ctx, {"until": []})  # ["\n"]
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

    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
                rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                    "\n\n".join(
                        [
                            self.doc_to_text(doc) + self.doc_to_target(doc)
                            # self.doc_to_fewshot_prompt(doc)
                            for doc in fewshotex
                        ]
                    )
                    + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example


class SummarizationTaskWithIdentifier(SummarizationTaskBase):
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
            "identifier": doc["identifier"],
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
            "identifier": lambda x: x,
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
            "identifier": True,
            "bertscore_precision": True,
            "bertscore_recall": True,
            "bertscore_f1": True,
            "coverage": False,
            "density": False,
            "compression": True,
        }

class SummarizationTaskLocal(SummarizationTaskBase):
    VERSION = 0
    # LCL_DATASET_PATH = "./results_extended_input/df_large_prompts.json"
    LCL_DATASET_PATH = "./results_extended_input/df_large_prompts_sample.json"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False
    def has_test_docs(self):
        return True

    def test_docs(self):
        # read the json file and return as generator
        data = []
        with open(self.LCL_DATASET_PATH, encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                data.append(json_obj)

        # if list of lists, flatten
        if isinstance(data[0], list):
            data = [item for sublist in data for item in sublist]

        return data

    def doc_to_text(self, doc):
        # prompts already applied
        return doc["article"]



class RepeatExperiments_after_bugfix_0_Llama7b(SummarizationTaskWithIdentifier):
    VERSION = 0
    DATASET_PATH = "roysc/repeated_results_Llama_2_7b_0"

class RepeatExperiments_after_bugfix_0_Llama70b(SummarizationTaskWithIdentifier):
    VERSION = 0
    DATASET_PATH = "roysc/repeated_results_Llama_2_70b_0"


class RepeatExperiments_after_bugfix_1_Llama70b(SummarizationTaskWithIdentifier):
    VERSION = 0
    DATASET_PATH = "roysc/repeated_results_Llama_2_70b_1"


class RepeatExperiments_after_bugfix_2_Llama70b(SummarizationTaskWithIdentifier):
    VERSION = 0
    DATASET_PATH = "roysc/repeated_results_Llama_2_70b_2"


class RepeatExperiments_after_bugfix_3_Llama70b(SummarizationTaskWithIdentifier):
    VERSION = 0
    DATASET_PATH = "roysc/repeated_results_Llama_2_70b_3"


class RepeatExperiments_after_bugfix_4_Llama70b(SummarizationTaskWithIdentifier):
    VERSION = 0
    DATASET_PATH = "roysc/repeated_results_Llama_2_70b_4"




class SummShard0_20Minuten_NonEmpty(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    # DATASET_PATH = "roysc/20min0"
    DATASET_PATH = "roysc/20minuten_sample_250"

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": []})  # ["\n"]
        return continuation


class SummarizationFCO_Fewshot_Base(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_sample_250"

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["---"]})  # ["\n"]
        return continuation

class SummSampleSmolSmol_20Minuten(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_sample_10"

class SummSampleSmolSmol_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    # DATASET_PATH = "roysc/20minuten_sample_250"
    DATASET_PATH = "roysc/20minuten_sample_10"


class SummFewshot_250TestSample_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20minuten_test_sample_250"


class SummShard0_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20min0"


class SummShard1_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20min1"


class SummShard2_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20min2"


class SummShard3_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20min3"


class SummSampleSmol_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    DATASET_PATH = "roysc/20minuten_sample_250"
    # DATASET_PATH = "roysc/20minuten_sample_10"


class SummSample_BadPalm2Examples_20Minuten(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20min_palm2_bad_examples"


class SummSampleSmolSmol_20Minuten(SummarizationTaskBase):
    VERSION = 0
    # DATASET_PATH = "roysc/20minuten"
    # DATASET_PATH = "roysc/20minuten_sample_250"
    DATASET_PATH = "roysc/20minuten_sample_10"


class SummarizationTask_20Minuten(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten"
    # DATASET_PATH = "roysc/20minuten_sample_250"


class SummLtM2_20min_prompt22_start(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_LtM1_22_processed_header_start_de"


class SummLtM2_20min_prompt22_end(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_LtM1_22_processed_header_end_de"


class SummLtM2_20min_prompt31_start(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_LtM1_31_processed_header_start_de"


class SummLtM2_20min_prompt31_end(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_LtM1_31_processed_header_end_de"


class SummLtM2_20min_prompt33_start(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_LtM1_33_processed_header_start_de"


class SummLtM2_20min_prompt33_end(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/20minuten_LtM1_33_processed_header_end_de"


class SummarizationTask_Wikinewssum(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum"


class SummarizationTask_Wikinewssum_Cleaned(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_cleaned"


class SummarizationTask_Wikinewssum_Simple(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_simple"


class SummarizationTask_Wikinewssum_SingleLine(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_simple_single_line"


class SummarizationTask_Wikinewssum_SingleLine_Shuffled(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_simple_single_line_shuffled"


class SummarizationTask_Wikinewssum_Simple_Shuffled(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_simple_shuffled"


class SummarizationTask_Wikinewssum_Simple_ArticleIdxAnn(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_simple_article_idx_ann"


class SummarizationTask_Wikinewssum_Simple_ArticleIdxAnn_Shuffled(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_simple_article_idx_ann_shuffled"


class SummarizationTask_Wikinewssum_Simple_Chunked_32(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_32"


class SummarizationTask_Wikinewssum_Simple_Chunked_64(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_64"


class SummarizationTask_Wikinewssum_Simple_Chunked_128(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_128"


class SummarizationTask_Wikinewssum_Simple_Chunked_256(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_256"


class SummarizationTask_Wikinewssum_Simple_Chunked_512(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_512"


class SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_2(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_by_sentences_maxSize_2"


class SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_4(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_by_sentences_maxSize_4"


class SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_8(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_by_sentences_maxSize_8"


class SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_16(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_by_sentences_maxSize_16"


class SummarizationTask_Wikinewssum_Simple_Chunked_Sentences_32(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_prefix_chunked_by_sentences_maxSize_32"


class SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_input_doc_split"


class SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_OriginalOrder(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_2stageSummary_stage2_originalOrder"


class SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_Shuffled(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_2stageSummary_stage2_shuffled"


class SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_BasePrompt41_OriginalOrder(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_2stageSummary_41_stage2_originalOrder"


class SummarizationTask_Wikinewssum_2Stage_Split_Input_Docs_Stage2_BasePrompt41_Shuffled(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/wikinewssum_2stageSummary_41_stage2_shuffled"


class MDS_WikinewsSum_SentenceChunk_1_00_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_1_MMR_0_0_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_1_05_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_1_MMR_0_5_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_3_00_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_3_MMR_0_0_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_3_05_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_3_MMR_0_5_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_5_00_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_5_MMR_0_0_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_5_05_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_5_MMR_0_5_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_10_00_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_10_MMR_0_0_512_sbert_comparison"


class MDS_WikinewsSum_SentenceChunk_10_05_512_sbert_comparison(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/ordered_chunks_sentences_10_MMR_0_5_512_sbert_comparison"


class MDS_WikinewsSum_ClusterChunk_1_Random(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_1_max_512_order_random"


class MDS_WikinewsSum_ClusterChunk_1_Original_Order(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_1_max_512_order_original-order"


class MDS_WikinewsSum_ClusterChunk_1_Cluster_Size(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_1_max_512_order_cluster-size"


class MDS_WikinewsSum_ClusterChunk_3_Random(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_3_max_512_order_random"


class MDS_WikinewsSum_ClusterChunk_3_Original_Order(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_3_max_512_order_original-order"


class MDS_WikinewsSum_ClusterChunk_3_Cluster_Size(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_3_max_512_order_cluster-size"


class MDS_WikinewsSum_ClusterChunk_5_Random(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_5_max_512_order_random"


class MDS_WikinewsSum_ClusterChunk_5_Original_Order(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_5_max_512_order_original-order"


class MDS_WikinewsSum_ClusterChunk_5_Cluster_Size(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_5_max_512_order_cluster-size"


class MDS_WikinewsSum_ClusterChunk_10_Random(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_10_max_512_order_random"


class MDS_WikinewsSum_ClusterChunk_10_Original_Order(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_10_max_512_order_original-order"


class MDS_WikinewsSum_ClusterChunk_10_Cluster_Size(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/chunk_clustering_sentences_10_max_512_order_cluster-size"


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
        cont_request = rf.greedy_until(ctx, {"until": []})
        return cont_request


class SummarizationTask_Klexikon(SummarizationTaskBase):
    VERSION = 0
    DATASET_PATH = "roysc/Klexikon_sample_250"




class MDS_CHAIN_Wikinews_Stage2_SDS_Prep(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_chain_S2_clust_total_2048_leave_512"


class MDS_FCO_Wikinews_Cheat_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_cheat_orig_0shot_chunk_1024"

class MDS_FCO_Wikinews_Cheat_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_cheat_orig_0shot_chunk_1536"

class MDS_FCO_Wikinews_Lead_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_lead_orig_0shot_chunk_1024"

class MDS_FCO_Wikinews_Lead_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_lead_orig_0shot_chunk_1536"

class MDS_FCO_Wikinews_Lead_1shot_20min_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_lead_orig_1shot_ex_20Min_seed_42_chunk_1024"

class MDS_FCO_Wikinews_Lead_1shot_20min_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_lead_orig_1shot_ex_20Min_seed_42_chunk_1536"

class MDS_FCO_Wikinews_Rand_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_rand_orig_0shot_chunk_1024"

class MDS_FCO_Wikinews_Rand_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_rand_orig_0shot_chunk_1536"


class MDS_FCO_Wikinews_Rand_1shot_20min_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_rand_orig_1shot_ex_20Min_seed_42_chunk_1024"

class MDS_FCO_Wikinews_Rand_1shot_20min_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_rand_orig_1shot_ex_20Min_seed_42_chunk_1536"


class MDS_FCO_Wikinews_Rand_1shot_Wikinews_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_rand_orig_1shot_ex_Wikinews_seed_42_chunk_1024"

class MDS_FCO_Wikinews_Rand_1shot_Wikinews_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_rand_orig_1shot_ex_Wikinews_seed_42_chunk_1536"



class MDS_FCO_Wikinews_Clust_0shot_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_0shot_chunk_1024"

class MDS_FCO_Wikinews_Clust_0shot_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_0shot_chunk_1536"

class MDS_FCO_Wikinews_Clust_0shot_2048(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_0shot_chunk_2048"



class MDS_FCO_Wikinews_Clust_1shot_20min_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_1shot_ex_20Min_chunk_1024"

class MDS_FCO_Wikinews_Clust_1shot_20min_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_1shot_ex_20Min_seed_42_chunk_1536"

class MDS_FCO_Wikinews_Clust_1shot_20min_42_2048(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_1shot_ex_20Min_seed_42_chunk_2048"


class MDS_FCO_Wikinews_Clust_2shot_20min_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_2shot_ex_20Min_seed_42_chunk_1024"

class MDS_FCO_Wikinews_Clust_2shot_20min_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_2shot_ex_20Min_seed_42_chunk_1536"

class MDS_FCO_Wikinews_Clust_2shot_20min_42_2048(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_2shot_ex_20Min_seed_42_chunk_2048"



class MDS_FCO_Wikinews_Clust_1shot_Wikinews_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_1shot_ex_Wikinews_seed_42_chunk_1024"

class MDS_FCO_Wikinews_Clust_1shot_Wikinews_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_1shot_ex_Wikinews_seed_42_chunk_1536"



class MDS_FCO_Wikinews_Clust_2shot_Wikinews_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_2shot_ex_Wikinews_seed_42_chunk_1024"


class MDS_FCO_Wikinews_Clust_2shot_Wikinews_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_clust_orig_2shot_ex_Wikinews_seed_42_chunk_1536"



class MDS_FCO_Wikinews_DistMMR_0shot_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_0shot_chunk_1024"

class MDS_FCO_Wikinews_DistMMR_0shot_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_0shot_chunk_1536"



class MDS_FCO_Wikinews_DistMMR_1shot_20min_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_1shot_ex_20Min_seed_42_chunk_1024"

class MDS_FCO_Wikinews_DistMMR_1shot_20min_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_1shot_ex_20Min_seed_42_chunk_1536"




class MDS_FCO_Wikinews_DistMMR_2shot_20min_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_2shot_ex_20Min_seed_42_chunk_1024"

class MDS_FCO_Wikinews_DistMMR_2shot_20min_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_2shot_ex_20Min_seed_42_chunk_1536"



class MDS_FCO_Wikinews_DistMMR_1shot_Wikinews_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_1shot_ex_Wikinews_seed_42_chunk_1024"

class MDS_FCO_Wikinews_DistMMR_1shot_Wikinews_42_1536(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_1shot_ex_Wikinews_seed_42_chunk_1536"

class MDS_FCO_Wikinews_DistMMR_2shot_Wikinews_42_1024(SummarizationFCO_Fewshot_Base):
    VERSION = 0
    DATASET_PATH = "roysc/mds_FCO_distMMR_orig_2shot_ex_Wikinews_seed_42_chunk_1024"

