import re
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
