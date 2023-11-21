import glob
import json
import os
import re
import shutil
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
import wandb


def get_model_and_task_name_from_csv_file_name(csv_file_name: str) -> Tuple[str, str]:
    # Regex pattern to extract everything after 'TASK_' and before '_group'
    task_name_pattern = r"TASK_(.*?).csv"
    model_name_pattern = r"MODEL_(.*?)_"

    match = re.search(model_name_pattern, csv_file_name)
    extracted_model_name = match.group(1) if match else "No match found"

    # Extracting the required part
    match = re.search(task_name_pattern, csv_file_name)
    extracted_task_name = match.group(1) if match else "No match found"

    return extracted_model_name, extracted_task_name


def domain_adaptation_summary_tables_post_processing(summary_tables_path: str):
    csv_files = glob.glob(f"{summary_tables_path}/part*.csv")

    summary_dfs = {}
    for csv_file in csv_files:
        part = csv_file.split("/")[-1].split("_")[0]
        model_name, task_name = get_model_and_task_name_from_csv_file_name(csv_file)
        if task_name in summary_dfs:
            summary_dfs[task_name][part] = pd.read_csv(csv_file)
        else:
            summary_dfs[task_name] = {part: pd.read_csv(csv_file)}

    for task_name, task_dfs in summary_dfs.items():
        column_name = 'truth'

        # Convert the first dataframe's column to a set
        part1_values = set(task_dfs["part1"][column_name])
        part2_values = set(task_dfs["part2"][column_name])
        common_values = part1_values.intersection(part2_values)

        print(f"Number of common values between part1 and part2: {common_values}")
        # assert len(common_values) == 0
        combined_df = pd.concat([task_dfs["part1"], task_dfs["part2"]], ignore_index=True)
        out_csv_name = f"{summary_tables_path}/combined_MODEL_{model_name}_TASK_{task_name}"
        combined_df.to_csv(out_csv_name, index=False)


def load_wandb_tables_by_project(entity: str, project: str, out_dir: str, selected_run_ids: Optional[List[str]] = None):
    # initialize API client
    api = wandb.Api()

    # make sure the directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get all runs in the project
    runs = api.runs(path=f"{entity}/{project}")

    for run in runs:
        if run.state == "finished":
            if selected_run_ids and run.id not in selected_run_ids:
                continue

            json_config = json.loads(run.json_config)
            model_name = json_config['model_args']['value'].split(",")[0].split("=")[1].replace("/", "-")
            task_name = json_config['task_configs']['value'][0]['task_name']
            table_name = f"{task_name}_output_table"
            filename = f"{run.id}_MODEL_{model_name}_TASK_{task_name}.csv"
            # length_path = f"{target_summary_length}_tokens"
            filepath = os.path.join(out_dir, filename)

            table_artifact = run.logged_artifacts()[0]
            table_dir = table_artifact.download()
            table_path = f"{table_dir}/{table_name}.table.json"
            with open(table_path) as file:
                json_dict = json.load(file)
            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
            df.to_csv(filepath, index=False)

            print(f"Data from run {run.id} saved to {filepath}")

    sleep(3)
    artifacts_root_dir = "artifacts"
    shutil.rmtree(artifacts_root_dir)


def load_wandb_domain_adaptation_summarization_data(out_dir: str):
    entity = "background-tool"
    project = "domain_adaptation"

    load_wandb_tables_by_project(entity=entity, project=project, out_dir=out_dir)


def main():
    out_dir = "generated_summaries"
    # load_wandb_domain_adaptation_summarization_data(out_dir)
    # domain_adaptation_summary_tables_post_processing(out_dir)


if __name__ == "__main__":
    main()
