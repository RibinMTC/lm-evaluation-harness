from . import seahorse_classification
import json
import numpy as np


class AutomaticSeahorseClassificationTask(seahorse_classification.SeahorseClassificationTask):
    VERSION = 0
    # LCL_DATASET_PATH = "./results_extended_input/example_input.json"
    LCL_DATASET_PATH = "./results_extended_input/base_datasets.json"
    # LCL_DATASET_PATH = "./results_extended_input/mds_fco_experiments.json"

    def has_test_docs(self):
        return True

    def test_docs(self):
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

    def doc_to_target(self, doc):
        # label = str(doc["question4"] == "Yes")
        return " " + str(True)

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
            "prediction": lambda x: x,
            "true_prediction_probability": lambda x: x,
            "id": lambda x: x,
            "article": lambda x: x,
            "summary": lambda x: x,
            "gt_summary": lambda x: x,
            "experiment_id": lambda x: x,
            "sub_id": lambda x: x,
        }

