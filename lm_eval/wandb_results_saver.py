import numpy as np
import wandb
import pandas as pd
import re

from typing import Dict, List

from pandas import DataFrame


def get_metric_key_name(run_summary: Dict):
    split_key = "_output_table"
    metric_key_name = None
    for key in run_summary._json_dict:
        if split_key in key:
            metric_key_name = key.split(split_key)[0].strip()
            assert "faithfulness_benchmark" in metric_key_name
            return metric_key_name
    return metric_key_name


def extract_number_of_params(model_name: str):
    number = re.search(r'(\d+)b', model_name)
    return int(number.group(1)) if number else 0


# Function to round and remove trailing zeros
def custom_round(value):
    if value is not None:
        # This will convert the number to a string, rounded to 3 decimal places, removing trailing zeros
        return round(value, 3)
        # return float(('{0:.3f}'.format(round(value, 3))).rstrip('0').rstrip('.')
    else:
        return None


def extract_run_info(run_name):
    # Define the regular expressions
    model_regex = r"MODEL_(.*?)_"
    shots_regex = r"_(\d+)-SHOT"
    prompt_version_regex = r"prompt-version-(.*)"
    fewshot_strategy_regex = r"sampling-(.*?)_"

    # Search for matches in the run name
    model_match = re.search(model_regex, run_name)
    shots_match = re.search(shots_regex, run_name)
    prompt_version_match = re.search(prompt_version_regex, run_name)
    fewshot_strategy_match = re.search(fewshot_strategy_regex, run_name)

    # Extract the matching groups
    model = model_match.group(1) if model_match else None
    shots = shots_match.group(1) if shots_match else None
    prompt_version = prompt_version_match.group(1) if prompt_version_match else None
    fewshot_strategy = fewshot_strategy_match.group(1) if fewshot_strategy_match else None

    return model, shots, prompt_version, fewshot_strategy


def get_correct_f1_all_scores_for_run(f1_all_metrics: Dict) -> Dict:
    num_metrics = len(f1_all_metrics)
    if num_metrics == 2:
        if "Intrinsic Hallucination" in f1_all_metrics:
            # Swap faithful and Hallucination values
            return {
                "F1 Faithful": custom_round(f1_all_metrics["Intrinsic Hallucination"]),
                "F1 Hallucination": custom_round(f1_all_metrics["Faithful"])
            }
        else:
            metrics_dict = {}
            for metric_name, metric_value in f1_all_metrics.items():
                metrics_dict[f"F1 {metric_name}"] = custom_round(metric_value)
            return metrics_dict
    elif num_metrics == 3:
        metrics_dict = {}
        metric_order = ["Faithful", "Intrinsic Hallucination", "Extrinsic Hallucination"]
        for metric in metric_order:
            metrics_dict[f"F1 {metric}"] = custom_round(f1_all_metrics[metric])
        return metrics_dict
    else:
        raise ValueError("Currently only 2 or 3 metrics are supported for faithfulness f1 metrics")


def get_correct_prompt_version(prompt_version: str) -> str:
    if prompt_version == "faithfulness":
        return "faithfulness_english"
    elif prompt_version == "faithfulness_only_english":
        return "faithfulness_english"
    # elif prompt_version == "all_labels":
    #     return "all_labels_english"
    # elif prompt_version == "all_labels_german_modified_prompt":
    #     return "all_labels_german"
    # elif prompt_version == "all_labels_german":
    #     return "all_labels_german_finetuned"
    return prompt_version


def get_correct_model_name(model_name: str) -> str:
    split_key = "-one-epoch-qlora-4bit-merged"
    if split_key in model_name:
        model_name = model_name.split(split_key)[0]
        return (f"{model_name}-finetuned")
    return model_name


# Truncate function
def truncate_number(n):
    if isinstance(n, (int, float)):
        return '{0:.3f}'.format(round(n, 3)).rstrip('0').rstrip('.')
    return n


def aggregate_results_by_seeds(results_df: DataFrame) -> DataFrame:
    aggregated_df = results_df.groupby(["Model Name", "Num Shots", "Prompt Version", "Fewshot strategy"]).agg(['mean', 'std']).reset_index()
    aggregated_df = aggregated_df.applymap(truncate_number)
    new_columns = []
    for col in aggregated_df.columns:
        col1, col2 = col
        if col1 == '':
            new_columns.append(col2)
        elif col2 == '':
            new_columns.append(col1)
        else:
            new_columns.append('-'.join(col))
    aggregated_df.columns = new_columns
    return aggregated_df


def save_wandb_faithfulness_runs(wandb_api, groups: List[str]):
    # A dictionary to hold the dataframes for each group
    group_dataframes = {}

    for group in groups:
        # Fetch runs from the group
        runs = wandb_api.runs("background-tool/llm_leaderboard", {"group": group})

        group_metric_key_name = None
        # Extract data from each run
        data = []
        for run in runs:
            if run.state != 'finished':
                continue
            if not group_metric_key_name:
                group_metric_key_name = get_metric_key_name(run.summary)
            # created_at = run.created_at
            # if '2023-10-26' not in created_at:
            #     continue
            metrics = run.summary.get(group_metric_key_name, {})
            f1_all_metrics = metrics.get("f1_all")
            (model, shots, prompt_version, fewshot_strategy) = extract_run_info(run.name)

            # if not fewshot_strategy or fewshot_strategy != "packed":
            #     continue
            if not fewshot_strategy or fewshot_strategy != "stratified" or shots not in ["6", "9"]:
                continue
            if not fewshot_strategy:
                fewshot_strategy = "-"
            elif fewshot_strategy == "packed":
                fewshot_strategy = "multiple sentence per article"

            wandb_results = {
                "Model Name": get_correct_model_name(model_name=model),
                "Num Shots": shots,
                "Prompt Version": get_correct_prompt_version(prompt_version=prompt_version),
                "Fewshot strategy": fewshot_strategy,
                "F1 Macro": custom_round(metrics.get("f1_macro", None))
            }

            wandb_results.update(get_correct_f1_all_scores_for_run(f1_all_metrics=dict(f1_all_metrics._json_dict)))
            wandb_results.update({"BACC": custom_round(metrics.get("bacc", None)),
                                  "Precision_Macro": custom_round(metrics.get("precision_macro", None)),
                                  "Recall_Macro": custom_round(metrics.get("recall_macro", None))
                                  })
            data.append(wandb_results)

        # Create a DataFrame and store it in the dictionary
        df = pd.DataFrame(data)
        df = aggregate_results_by_seeds(results_df=df)
        df['Model_number_of_params'] = df['Model Name'].apply(lambda x: extract_number_of_params(x))
        df = df.sort_values(by=['Model_number_of_params', 'Model Name', 'Prompt Version', 'Num Shots'])
        df.drop(columns=['Model_number_of_params'], inplace=True)
        df.replace({"nan": 0}, inplace=True)
        df.to_csv(f"result_tables/wandb_group_results_averaged_{group}_6_9.csv", index=False)
        group_dataframes[group] = df


def main():
    groups = ["llm_leaderboard_TASK_faithfulness_benchmark_final_swisstext23_multi_label_group"]
    # "llm_leaderboard_TASK_faithfulness_benchmark_final_swisstext23_benchmark_faithful_group",
    # "llm_leaderboard_TASK_faithfulness_benchmark_final_swisstext23_benchmark_faithful_german_group"]
    # llm_leaderboard_TASK_faithfulness_benchmark_final_swisstext23_multi_label_group
    # "llm_leaderboard_TASK_faithfulness_benchmark_final_swisstext23_with_explanation_multi_label_group"

    api = wandb.Api()

    save_wandb_faithfulness_runs(wandb_api=api, groups=groups)


if __name__ == "__main__":
    main()
