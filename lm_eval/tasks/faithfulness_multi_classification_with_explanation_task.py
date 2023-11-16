import re
from collections import Counter

from lm_eval.base import rf

from lm_eval.tasks.faithfulness_multi_classification_base_task import FaithfulnessMultiClassificationBaseTask


class FaithfulnessMultiClassificationWithExplanationTask(FaithfulnessMultiClassificationBaseTask):
    ANS_RE = re.compile(r"(?i)label:\s*(.*?)(?=\n|$)")
    INVALID_ANS = "Invalid"

    choices = ["Faithful", "Intrinsic Hallucination", "Extrinsic Hallucination", INVALID_ANS]

    # def test_docs(self):
    #     if self.has_test_docs():
    #         return map(self._process_doc, self.dataset["validation"])

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, {"until": []})
        return completion

    def convert_label(self, label) -> int:
        label_index = self.choices.index(self.INVALID_ANS)
        try:
            label_index = self.choices.index(label)
        except ValueError:
            print(f"Label: {label} is not a valid label choice")
        return label_index

    def extract_label(self, completion) -> int:
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return self.convert_label(label=match_str)
        else:
            return self.convert_label(label=self.INVALID_ANS)

    def process_results(self, doc, results):
        completion = results[0]
        prediction = self.extract_label(completion=completion)
        truth = doc["gold"]
        self.task_results_info_list.append({"prediction": prediction, "reference": truth})
        print(f"Results: {results}, Prediction {prediction}, Truth: {truth}")
        return {"bacc": (prediction, truth),
                "f1_macro": (prediction, truth),
                "f1_all": (prediction, truth),
                "f1_micro": (prediction, truth),
                "precision_macro": (prediction, truth),
                "recall_macro": (prediction, truth)
                }


class FullDisagreementsFaithfulnessMultiClassificationWithExplanationTask(
    FaithfulnessMultiClassificationWithExplanationTask):
    DATASET_PATH = "mtc/final_german_faithfulness_benchmark_with_full_disagreements"


class SeahorseFaithfulnessMultiClassificationWithExplanationTask(
    FaithfulnessMultiClassificationWithExplanationTask):
    DATASET_PATH = "mtc/german_seahorse_dataset_with_articles"
    article_key_name = "article"
    sentence_key_name = "summary"
    label_key_name = "question4"

    def test_docs(self):
        if self.has_test_docs():
            def is_high_quality_sample(row):
                return row['question1'].lower() == 'yes' and row['question2'].lower() == 'yes' and row[
                    'question3'].lower() == 'yes'

            filtered_dataset = self.dataset["test"].filter(is_high_quality_sample)
            # attribution_values = filtered_dataset['question4']
            # value_counts = Counter(attribution_values)
            #
            # # Display the counts
            # for value, count in value_counts.items():
            #     print(f"'{value}': {count}")

            return map(self._process_doc, filtered_dataset)

    @staticmethod
    def seahorse_to_absinth_label(label: str):
        if label.lower() == "yes":
            return "Faithful"
        elif label.lower() == "no":
            return "Extrinsic Hallucination"
        else:
            raise ValueError(f"Unsupported value for label {label}")

    def _process_doc(self, doc):
        out_doc = {
            "query": self.format_prompt(doc),
            "choices": self.choices,
            "gold": self.choices.index(self.seahorse_to_absinth_label(doc[self.label_key_name])),
            "original_doc": doc
        }
        return out_doc


class XnliFaithfulnessMultiClassificationWithExplanationTask(
    FaithfulnessMultiClassificationWithExplanationTask):
    DATASET_PATH = "xnli"
    DATASET_NAME = "de"
    article_key_name = "premise"
    sentence_key_name = "hypothesis"
    label_key_name = "label"

    @staticmethod
    def xnli_to_absinth_label(label: int):
        if label == 0:
            return "Faithful"
        elif label == 1:
            return "Extrinsic Hallucination"
        elif label == 2:
            return "Intrinsic Hallucination"
        else:
            raise ValueError(f"Unsupported value for label {label}")

    def _process_doc(self, doc):
        out_doc = {
            "query": self.format_prompt(doc),
            "choices": self.choices,
            "gold": self.choices.index(self.xnli_to_absinth_label(doc[self.label_key_name])),
            "original_doc": doc
        }
        return out_doc
