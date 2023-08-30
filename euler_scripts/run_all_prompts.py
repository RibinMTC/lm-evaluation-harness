import os
import shutil
import yaml
import json
import subprocess

# Definitions
BASE_PROMPT_TEMPLATE = "configs/prompt_templates/summarization_base.json"
DATASET_NAME = "20Minuten"
TASK_BASE_NAME = f"SummarizationTask_{DATASET_NAME}_"

# Make two lists: one for the task names, one for the prompt versions
TASK_NAMES = []
PROMPT_VERSIONS = []

# Loop through indices 1 to 9
for i in range(1, 3): # 1,10

    # Append the task name and the prompt version to the lists
    TASK_NAMES.append(f"{TASK_BASE_NAME}{i}")
    PROMPT_VERSIONS.append(str(i))

    # Generate the new prompt template file by copying the base template
    new_prompt_template = f"../configs/prompt_templates/{TASK_BASE_NAME}{i}.json"
    shutil.copy(f"../{BASE_PROMPT_TEMPLATE}", new_prompt_template)

# print(TASK_NAMES)
# print(PROMPT_VERSIONS)

# Create the new config file
old_config = "configs/eval_config.yaml"
new_config = "configs/eval_config_tmp.yaml"
# Concatenate the two lists into a comma-separated string
new_config_task_names = ",".join(TASK_NAMES)
new_config_prompt_versions = ",".join(PROMPT_VERSIONS)
# Load the old config
with open(f"../{old_config}") as f:
    y = yaml.safe_load(f)
    # Update tasks and prompt_version_per_task values
    y["tasks"] = new_config_task_names
    y["prompt_version_per_task"] = new_config_prompt_versions
    # Write the updated config to the new file
    with open(f"../{new_config}", "w") as new_f:
        yaml.dump(y, new_f, default_flow_style=False, sort_keys=False)

# Create the temporary lm-evaluation-config file using pyjq
with open("lm_eval_euler_config.json") as f:
    old_config_data = json.load(f)
    old_config_data["config_file"] = new_config

with open("tmp_euler_config.json", "w") as new_f:
    json.dump(old_config_data, new_f, indent=4)

# # Run the command with the updated config
os.system("bash run_euler.sh tmp_euler_config.json")

# Remove the temporary config files
os.remove("tmp_euler_config.json")
os.remove(f"../{new_config}")