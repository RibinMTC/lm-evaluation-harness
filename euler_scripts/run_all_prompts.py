import os
import shutil
from ruamel import yaml
import json
import subprocess
import pprint
import itertools
import random, string

"""
    Experiment Combination Parameters
"""
models = [
    # garage-bAInd/Platypus2-70B-instruct
    "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf",
]
temperature_values = [0] # [0, 0.1, 0.5, 1.0]
precision_values = [""]  # ["", ",load_in_8bit=True"]
dataset_names = ["20Minuten"]
prompt_versions = [1, 2, 3, 5, 20]
task_base_names = ["SummarizationTask_"]  # ["SummLtM_", "SummLtMDe_"]

"""
    Definitions
"""
inferable_args = {
    "model": {
        "default": "hf-causal-experimental",
        "meta-llama/Llama-2-7b-chat-hf": "hf-causal-experimental",
        "meta-llama/Llama-2-13b-chat-hf": "hf-causal-experimental",
        "meta-llama/Llama-2-70b-chat-hf": "hf-causal-experimental",
        "bigscience/bloomz-7b1-mt": "hf-causal-experimental",
        "tiiuae/falcon-7b-instruct": "hf-causal-experimental",
        "tiiuae/falcon-40b-instruct": "hf-causal-experimental",
    },
    "task_temp_suffix": {
        "default": "",
        0: "",
        0.1: "_T01",
        0.5: "_T05",
        1.0: "_T10",
    },
    "run_duration_hours": {
        "default": "04:00",
        "meta-llama/Llama-2-7b-chat-hf": "02:00",
        "meta-llama/Llama-2-13b-chat-hf": "02:00",
        "meta-llama/Llama-2-70b-chat-hf": "04:00",
        "bigscience/bloomz-7b1-mt": "04:00",
        "tiiuae/falcon-7b-instruct": "04:00",
        "tiiuae/falcon-40b-instruct": "08:00",
    },
    "gpu": {
        "default": "rtx_3090",
        "meta-llama/Llama-2-7b-chat-hf": "rtx_3090",
        "meta-llama/Llama-2-13b-chat-hf": "rtx_3090",
        "meta-llama/Llama-2-70b-chat-hf": "a100-pcie-40gb",
        "bigscience/bloomz-7b1-mt": "a100-pcie-40gb",
        "tiiuae/falcon-7b-instruct": "a100-pcie-40gb",
        "tiiuae/falcon-40b-instruct": "a100_80gb",
    },
    "num_gpus": {
        "default": 1,
        "meta-llama/Llama-2-7b-chat-hf": 1,
        "meta-llama/Llama-2-13b-chat-hf": 1,
        "meta-llama/Llama-2-70b-chat-hf": 1,
        "bigscience/bloomz-7b1-mt": 1,
        "tiiuae/falcon-7b-instruct": 1,
        "tiiuae/falcon-40b-instruct": 2,
    }
}
BASE_PROMPT_TEMPLATE = "configs/prompt_templates/summarization_base.json"
TMP_PROMPT_TEMPLATE = "configs/prompt_templates/{name}.json"
BASE_CONFIG = "configs/eval_config.yaml"
NEW_CONFIG_PATTERN = "configs/eval_config_{random_string}.yaml"
BASE_EULER_CONFIG = "lm_eval_euler_config.json"
TMP_EULER_CONFIG = "tmp_euler_config.json"

task_name_schema = "{task_base_name}{dataset_name}{task_temp_suffix}{task_prompt_suffix}"
model_args_schema = "pretrained={model},trust_remote_code=True,use_accelerate=True{temperature_suffix}"

"""
    Build the configurations
"""
# Create a list of all possible combinations of the parameters
combinations = list(itertools.product(models, temperature_values, precision_values, dataset_names, prompt_versions, task_base_names))
config_list = []

# Iterate over each combination and create a config dictionary
for combination in combinations:
    model, tempVal, precision, dataset, promptVersion, taskBaseName = combination

    # prepare the values
    task_temp_suffix = inferable_args["task_temp_suffix"][tempVal] if tempVal in inferable_args["task_temp_suffix"] else inferable_args["task_temp_suffix"]["default"]
    if tempVal == 0:
        temp_suffix_model_args = ""
    else:
        temp_suffix_model_args = f",do_sample=True,temperature={tempVal}"

    # Build the arguments (eval_config)
    model_config = inferable_args["model"][model] if model in inferable_args["model"] else inferable_args["model"]["default"]
    model_args = model_args_schema.format(model=model, temperature_suffix=temp_suffix_model_args)
    task_name = task_name_schema.format(task_base_name=taskBaseName, dataset_name=dataset, task_temp_suffix=task_temp_suffix, task_prompt_suffix=f"_{promptVersion}")
    # Build the arguments (euler_config)
    run_duration_hours = inferable_args["run_duration_hours"][model] if model in inferable_args["run_duration_hours"] else inferable_args["run_duration_hours"]["default"]
    gpu = inferable_args["gpu"][model] if model in inferable_args["gpu"] else inferable_args["gpu"]["default"]
    num_gpus = inferable_args["num_gpus"][model] if model in inferable_args["num_gpus"] else inferable_args["num_gpus"]["default"]

    # Create the config dictionary for this combination
    config = {
        "model": model_config,
        "model_args": model_args,
        "tasks": task_name,
        "prompt_version_per_task": f"{promptVersion}",
        "run_duration_hours": run_duration_hours,
        "gpu": gpu,
        "num_gpus": num_gpus,
    }

    # Append the config dictionary to the list
    config_list.append(config)

# Print configuration combinations
for config in config_list:
    pprint.pprint(config)

# Ask the user if they want to continue
user_input = input(f"\n{len(config_list)} configurations are built. Do you want to continue? (Yes/No): ")

while user_input.lower() not in ["yes", "y", "no", "n"]:
    user_input = input("Please enter a valid input (Yes/No): ")

if user_input.lower() in ["yes", "y"]:
    print("Execution continues.")
else:
    print("Execution stopped.")
    exit(0)

"""
    Schedule the tasks
"""
for config in config_list:
    print(f"Scheduling\n")
    pprint.pprint(config)

    new_prompt_template = TMP_PROMPT_TEMPLATE.format(name=f"{config['tasks']}.json")
    shutil.copy(f"../{BASE_PROMPT_TEMPLATE}", f"../{new_prompt_template}")

    """
    Load the old eval_config and update the values, and write it to the new temporary file
    """
    # generate a random 10 character long string for the config file name
    random_string = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    new_config = NEW_CONFIG_PATTERN.format(random_string=random_string)

    # configurate the yaml library
    yamlPrinter = yaml.YAML()
    yamlPrinter.preserve_quotes = True
    yamlPrinter.default_flow_style = False
    with open(f"../{BASE_CONFIG}") as f:
        y = yamlPrinter.load(f)
    y["model"] = config['model']
    y["model_args"] = config['model_args']
    y["tasks"] = config['tasks']
    y["prompt_version_per_task"] = config["prompt_version_per_task"]

    with open(f"../{new_config}", "w") as new_f:
        yamlPrinter.dump(y, new_f)

    """
    Generate the temporary euler-config-file
    """
    with open(BASE_EULER_CONFIG) as f:
        old_config_data = json.load(f)
    old_config_data["run_duration_hours"] = config["run_duration_hours"]
    old_config_data["gpu"] = config["gpu"]
    old_config_data["num_gpus"] = config["num_gpus"]
    old_config_data["config_file"] = new_config

    with open(TMP_EULER_CONFIG, "w") as new_f:
        json.dump(old_config_data, new_f, indent=4)

    """
    Schedule the experiment
    """
    # Run the command with the updated config
    os.system(f"bash run_euler.sh {TMP_EULER_CONFIG}")

    # Remove the temporary config files
    os.remove(f"tmp_euler_config.json")
    # os.remove(f"../{new_config}")
