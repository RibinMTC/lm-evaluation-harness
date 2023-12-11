import re
from collections import Counter

from lm_eval.base import rf

from lm_eval.tasks.faithfulness_multi_classification_base_task import FaithfulnessMultiClassificationBaseTask


class FaithfulnessMultiClassificationWithExplanationTask(FaithfulnessMultiClassificationBaseTask):
    DATASET_PATH = "mtc/final_german_faithfulness_benchmark_with_explanations"
    explanation_key_name = "explanation"

    ANS_RE = re.compile(r"(?i)label:\s*(.*?)(?=\n|$)")
    INVALID_ANS = "Invalid"

    choices = ["Faithful", "Intrinsic Hallucination", "Extrinsic Hallucination", INVALID_ANS]

    def format_prompt_target(self, doc):
        explanation = doc[self.explanation_key_name]
        label = doc[self.label_key_name]
        explanation_with_label = f"Erklärung:{explanation}\nLabel:{label}"
        return explanation_with_label

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, {"until": []})
        return completion

    def get_label_index(self, label) -> int:
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
            return self.get_label_index(label=match_str)
        else:
            return self.get_label_index(label=self.INVALID_ANS)

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


class AbsinthWithExplanationEmptyArticleTask(
    FaithfulnessMultiClassificationWithExplanationTask):
    DATASET_PATH = "mtc/absinth_correctly_predicted_samples_by_leo_mistral_cot_3epochs"

    def format_prompt(self, doc):
        if not self.prompt_template:
            self.prompt_template = self.default_prompt_template
        prompt = self.prompt_template.format(article="",
                                             sentence=doc[self.sentence_key_name])
        return prompt

    def _process_doc(self, doc):
        out_doc = {
            "query": self.format_prompt(doc),
            "choices": self.choices,
            "gold": self.choices.index(self.convert_label("Extrinsic Hallucination")),
            "original_doc": doc
        }
        return out_doc


class AbsinthWithExplanationUnrelatedArticleTask(
    FaithfulnessMultiClassificationWithExplanationTask):
    DATASET_PATH = "mtc/absinth_correctly_predicted_samples_by_leo_mistral_cot_3epochs"
    UNRELATED_ARTICLE = """Das Hämoglobin im Blut ist nicht nur für die typisch rote Farbe verantwortlich. Es ist der wichtigste Bestandteil der roten Blutkörperchen (Erythrozyten). Diese haben die Aufgabe, alle Körperzellen mit lebenswichtigem Sauerstoff zu versorgen und auf dem Rückweg zur Lunge Kohlendioxid als Stoffwechselendprodukt zu entfernen. Sauerstoff und Kohlendioxid werden mit Hilfe des Hämoglobins transportiert. Im Hämoglobin ist Eisen enthalten, das den Sauerstoff bindet.
Rote Blutkörperchen haben eine Lebensdauer 100 bis 140 Tagen, d. h. es werden ständig ältere Erythrozyten abgebaut und neue kommen hinzu. Im Monat werden insgesamt rund 1,2 Liter Blut neu gebildet. Das im roten Blutfarbstoff (Hämoglobin) enthaltene Eisen wird hierbei fast vollständig wieder zum Neubau von Hämoglobin verwendet. Jeder gesunde Mensch verfügt über eine natürliche Eisenreserve, mit der Verluste normalerweise rasch ausgeglichen werden können. Im Bedarfsfall steigt die Neubildung von roten Blutkörperchen bis auf das 15-Fache des Normalwertes an.
Bei einer Blutspende oder bei größerem Blutverlust geht für den Neuaufbau von Hämoglobin wichtiges Eisen verloren. Besitzt ein Spender zu wenig roten Blutfarbstoff, d. h. ist sein Hämoglobinwert zu niedrig oder an der unteren Grenze, so hat er keine ausreichenden Eisenreserven für eine gesteigerte Neubildung von voll funktionsfähigen Erythrozyten. Eine Blutspende ist zu diesem Zeitpunkt dann nicht möglich."""

    def format_prompt(self, doc):
        if not self.prompt_template:
            self.prompt_template = self.default_prompt_template
        prompt = self.prompt_template.format(article=self.UNRELATED_ARTICLE,
                                             sentence=doc[self.sentence_key_name])
        return prompt

    def _process_doc(self, doc):
        out_doc = {
            "query": self.format_prompt(doc),
            "choices": self.choices,
            "gold": self.choices.index(self.convert_label("Extrinsic Hallucination")),
            "original_doc": doc
        }
        return out_doc

class SeahorseFaithfulnessMultiClassificationWithExplanationTask(FaithfulnessMultiClassificationWithExplanationTask):
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

            return map(self._process_doc, filtered_dataset)

    def convert_label(self, label: str) -> str:
        if label.lower() == "yes":
            return "Faithful"
        elif label.lower() == "no":
            return "Extrinsic Hallucination"
        else:
            raise ValueError(f"Unsupported value for label {label}")


class XnliFaithfulnessMultiClassificationWithExplanationTask(FaithfulnessMultiClassificationWithExplanationTask):
    DATASET_PATH = "xnli"
    DATASET_NAME = "de"
    article_key_name = "premise"
    sentence_key_name = "hypothesis"
    label_key_name = "label"

    def convert_label(self, label: int) -> str:
        if label == 0:
            return "Faithful"
        elif label == 1:
            return "Extrinsic Hallucination"
        elif label == 2:
            return "Intrinsic Hallucination"
        else:
            raise ValueError(f"Unsupported value for label {label}")
