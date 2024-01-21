from . import seahorse_classification
import json
import numpy as np
from lm_eval.base import rf
from lm_eval.metrics import mean


class AutomaticSeahorseClassificationTask(seahorse_classification.SeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/base_datasets"

    load_lcl = False
    # LCL_DATASET_PATH = "./results_extended_input/example_input.json"
    LCL_DATASET_PATH = "./results_extended_input/base_datasets.json"

    # LCL_DATASET_PATH = "./results_extended_input/mds_fco_experiments.json"

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

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.load_lcl:
            # read the json file and return as generator
            data = []
            with open(self.LCL_DATASET_PATH, encoding='utf-8') as f:
                for line in f:
                    json_obj = json.loads(line)
                    json_obj["worker_lang"] = 'de'
                    data.append(json_obj)

                # data = json.load(f)
                # # add worker_lang to each example
                # for example in data:
                #     example["worker_lang"] = 'de'

            return data
        elif self.has_test_docs():
            test_set = self.dataset["test"]
            for idx, doc in enumerate(test_set):
                test_set[idx]["worker_lang"] = 'de'
            return test_set

    def doc_to_text(self, doc):
        self.prompt_template = self.default_prompt_template # override the prompt template in any case
        prompt = self.prompt_template.format(article=doc['article'],
                                             summary=doc['summary'], label="",
                                             lang=self.worker_lang_to_language_dict['de']) # and use german as language
        return prompt

    def doc_to_target(self, doc):
        # label = str(doc["question4"] == "Yes")
        return " " + str(True)

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
        true_prediction_probability = results_probabilities[1] / np.sum(results_probabilities)

        return {
            "prediction": int(prediction),
            "true_prediction_probability": float(true_prediction_probability),
            "id": doc["id"],
            "article": doc["article"],
            "summary": doc["summary"],
            "gt_summary": doc["gt_summary"],
            "experiment_id": doc["experiment_id"],
            "sub_id": doc["sub_id"],
        }

    def aggregation(self):
        # don't aggregate at all
        return {
            "prediction": lambda x: 0,
            "true_prediction_probability": mean,
            "id": lambda x: 0,
            "article": lambda x: 0,
            "summary": lambda x: 0,
            "gt_summary": lambda x: 0,
            "experiment_id": lambda x: 0,
            "sub_id": lambda x: 0,
        }


class AutomaticSeahorseClassificationTask_Local(AutomaticSeahorseClassificationTask):
    VERSION = 0
    # DATASET_PATH = "roysc/base_datasets"
    load_lcl = True
    LCL_DATASET_PATH = "./results_extended_input/base_datasets.json"


class AutomaticSeahorseClassificationTask_BaseDatasets(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_base_datasets"
    load_lcl = False

class AutomaticSeahorseClassificationTask_FCOExperiments(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_mds_fco_experiments"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_falcon7b_1(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_falcon7b_1"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_falcon7b_2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_falcon7b_2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_bloomz7b_1(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_bloomz7b_1"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_bloomz7b_2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_bloomz7b_2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_palm2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_palm2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_gpt4(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_gpt4"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_platypus2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_platypus2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_orcallama2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_orcallama2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_leolm7b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_leolm7b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_leolm13b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_leolm13b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_7b_1(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_7b_1"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_7b_2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_7b_2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_7b_3(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_7b_3"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_13b_1(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_13b_1"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_13b_2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_13b_2"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_1(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_1"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_2_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_2_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_2_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_2_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_3(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_3"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_4(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_4"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_5_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_5_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_5_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_5_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_6_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_6_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_6_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_6_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_7_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_7_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_7_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_7_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_8_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_8_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_8_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_8_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_9_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_9_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_9_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_9_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_10_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_10_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_10_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_10_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_11_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_11_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_11_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_11_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_12_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_12_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_12_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_12_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_13_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_13_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_13_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_13_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_14_a(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_14_a"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_14_b(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_14_b"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_15(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_15"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_16(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_16"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_17(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_17"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_llama2_70b_18(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_llama2_70b_18"
    load_lcl = False

class AutomaticSeahorseClassificationTask_SEAHORSE_missing(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing"
    load_lcl = False

class AutomaticSeahorseClassificationTask_SEAHORSE_missing2(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing2"
    load_lcl = False

class AutomaticSeahorseClassificationTask_SEAHORSE_missing3(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing3"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_missing4(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing4"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_missing5(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing5"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_missing6(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing6"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_missing7(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing7"
    load_lcl = False


class AutomaticSeahorseClassificationTask_SEAHORSE_missing8(AutomaticSeahorseClassificationTask):
    VERSION = 0
    DATASET_PATH = "roysc/SEAHORSE_missing8"
    load_lcl = False
