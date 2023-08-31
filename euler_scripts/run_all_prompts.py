import os
import shutil
import yaml
import json
import subprocess
import random, string

# ["meta-llama/Llama-2-7b-chat-hf", ]
models = ["meta-llama/Llama-2-7b-chat-hf", "bigscience/bloomz-7b1-mt", "tiiuae/falcon-7b-instruct"] # [, "google/flan-t5-xl"]
indices = [10,11,12,13,14,15,16,17,18,19]


for model in models:
    for j in indices:
        print(f"Scheduling {model} with prompt index {j}...\n")

        # Definitions
        BASE_PROMPT_TEMPLATE = "configs/prompt_templates/summarization_base.json"
        DATASET_NAME = "20Minuten"
        TASK_BASE_NAME = f"SummarizationTask_{DATASET_NAME}_"

        # Make two lists: one for the task names, one for the prompt versions
        TASK_NAMES = []
        PROMPT_VERSIONS = []

        # Loop through indices 1 to 9
        # for i in [1,2]:  # range(1, 10): # 1,10

        i = j
        # Append the task name and the prompt version to the lists
        TASK_NAMES.append(f"{TASK_BASE_NAME}{i}")
        PROMPT_VERSIONS.append(str(i))

        # Generate the new prompt template file by copying the base template
        new_prompt_template = f"../configs/prompt_templates/{TASK_BASE_NAME}{i}.json"
        shutil.copy(f"../{BASE_PROMPT_TEMPLATE}", new_prompt_template)

        # print(TASK_NAMES)
        # print(PROMPT_VERSIONS)

        # generate a random 6 character long string
        random_string = ''.join(random.choice(string.ascii_lowercase) for i in range(6))

        # Create the new config file
        old_config = "configs/eval_config.yaml"
        new_config = f"configs/eval_config_{random_string}.yaml"
        # Concatenate the two lists into a comma-separated string
        new_config_task_names = ",".join(TASK_NAMES)
        new_config_prompt_versions = ",".join(PROMPT_VERSIONS)
        # Load the old config
        with open(f"../{old_config}") as f:
            y = yaml.safe_load(f)
            # Update tasks and prompt_version_per_task values
            y["tasks"] = new_config_task_names
            y["prompt_version_per_task"] = new_config_prompt_versions
            y["model_args"] = f"pretrained={model},trust_remote_code=True,use_accelerate=True"
            # Write the updated config to the new file
            with open(f"../{new_config}", "w") as new_f:
                yaml.dump(y, new_f, default_flow_style=False, sort_keys=False)

        # Create the temporary lm-evaluation-config file using pyjq
        with open("lm_eval_euler_config.json") as f:
            old_config_data = json.load(f)
            old_config_data["config_file"] = new_config

        with open("tmp_euler_config.json", "w") as new_f:
            json.dump(old_config_data, new_f, indent=4)

        # Run the command with the updated config
        os.system("bash run_euler.sh tmp_euler_config.json")

        # Remove the temporary config files
        os.remove("tmp_euler_config.json")
        # os.remove(f"../{new_config}")
