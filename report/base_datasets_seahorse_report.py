import json
import os
import pandas as pd
import numpy as np


def load_seahorse_results_by_experiment_id(experiment_id):
    base_folders = [
        ("../results/mtc-NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged", "attribution"),
        ("../results/mtc-NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged", "conciseness"),
        ("../results/mtc-NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged", "main-ideas"),
    ]

    results = []

    attr_val_map = {
        "1": True,
        "0": False,
    }

    # load all files from all base folders, filter by experiment_id, and merge entries for different fields with the same article
    for base_folder, attribute in base_folders:
        for filename in os.listdir(base_folder):
            if filename.endswith(".json"):
                with open(os.path.join(base_folder, filename), "r") as f:
                    file_results = json.load(f)
                    file_results = [entry for entry in file_results if entry["experiment_id"] == experiment_id]
                    for entry_idx, entry in enumerate(file_results):
                        file_results[entry_idx]["attribute"] = attribute
                    results.extend(file_results)
    # prepare the merge
    results_dict = {}
    for entry in results:
        if entry["id"] not in results_dict:
            results_dict[entry["id"]] = []
        results_dict[entry["id"]].append(entry)
    # merge entries with the same id
    merged_results = []
    for id in results_dict:
        # make sure we have all attributes
        if len(results_dict[id]) != 3:
            raise ValueError(f"Expected 3 entries for id {id}, but got {len(results_dict[id])}")

        # make sure all entries have the same ground-truth summary
        article = results_dict[id][0]["article"]
        gt_summary = results_dict[id][0]["gt_summary"]
        for entry in results_dict[id]:
            assert entry["gt_summary"] == gt_summary

        # merge the actual entries
        merged_entry = {
            "id": id,
            "article": article,
            "gt_summary": gt_summary,
        }
        for entry in results_dict[id]:
            merged_entry[f"{entry['attribute']}"] = attr_val_map[entry["prediction"]]
            merged_entry[f"{entry['attribute']}-prob"] = entry["true_prediction_probability"]
        merged_results.append(merged_entry)

    merged_df = pd.DataFrame(merged_results)
    return merged_df

if "__main__" == __name__:
    curr_experiment_ids = [
        "20Minuten", "Klexikon", "Wikinews"
    ]

    for experiment_id in curr_experiment_ids:
        df = load_seahorse_results_by_experiment_id(experiment_id)

        not_main_ideas = df[df["main-ideas"] == False]
        not_concise = df[df["conciseness"] == False]
        not_attributed = df[df["attribution"] == False]

        print("Hello World")

        """
        main ideas: The summary captures the main idea(s) of the source article
        conciseness: The summary concisely represents the information in the source article.
        attribution: All of the information provided by the summary is fully attributable to the source article
        """