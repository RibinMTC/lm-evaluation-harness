import json
import os
import re
from dataclasses import asdict
from typing import Dict, List

import yaml

from lm_eval.utils import TaskConfig

model_names = [
    "LeoLM/leo-hessianai-13b"]
num_fewshots = [8, 16, 24]
few_shot_sampling_strategies = ["stratified"]
seeds = [42]
#selected_tasks_with_prompt_version_names = []
task_config_list: List[TaskConfig] = []
load_in_8bit = False
use_flash_attention_2 = False

model_name_to_type_map = {
    "NousResearch/Llama-2": "hf-causal-experimental",
    "google/flan-ul2": "hf-seq2seq",
    "LeoLM/leo": "hf-causal-experimental",
    "mtc/LeoLM": "hf-causal-experimental",
    "ehartford/dolphin": "hf-causal-experimental",
    "mtc/ehartford-dolphin": "hf-causal-experimental",
    "meta-llama": "hf-causal-experimental",
    "mistralai/Mistral-7B-Instruct-v0.1": "hf-causal-experimental"
}

base_yaml_config_path = ""


# def get_task_name_from_prompt(prompt_name: str) -> str:
#     for prompt, task_name in prompt_to_task_map.items():
#         if prompt in prompt_name:
#             return task_name
#     raise ValueError(f"Prompt: {prompt_name} could not be mapped to a task")


def get_model_type_from_name(model_name: str):
    for model_prefix_name, model_type in model_name_to_type_map.items():
        if model_prefix_name in model_name:
            return model_type
    raise ValueError(f"No model type found for model {model_name}")


def read_yaml(yaml_file_name: str):
    with open(yaml_file_name, 'r') as file:
        yaml_file = yaml.safe_load(file)
    return yaml_file


def write_yaml(data, yaml_out_file_directory: str, yaml_out_file_name: str):
    if not os.path.exists(yaml_out_file_directory):
        os.makedirs(yaml_out_file_directory)
    final_path = os.path.join(yaml_out_file_directory, yaml_out_file_name)
    with open(final_path, 'w') as file:
        yaml.dump(data, file, indent=4)


def update_string(original_string: str, string_pattern: str, string_value: str) -> str:
    replacement_string = f",{string_pattern}{string_value}"
    updated_string = original_string + replacement_string
    return updated_string


def update_model_args(models_args: str) -> str:
    load_in_8bit_string = "load_in_8bit="
    use_flash_attention_2_string = "use_flash_attention_2="
    updated_model_args = update_string(original_string=models_args, string_pattern=load_in_8bit_string,
                                       string_value=str(load_in_8bit))
    updated_model_args = update_string(original_string=updated_model_args, string_pattern=use_flash_attention_2_string,
                                       string_value=str(use_flash_attention_2))
    return updated_model_args


def main():
    base_models_args = "pretrained={model_name},trust_remote_code=False,use_accelerate=True,dtype=bfloat16"
    config_out_yaml_template = ("{prompt_version}_{model_name}_num_fewshot_{num_fewshot}-{fewshot_sampling}_seed_{"
                                "seed}_eval_config.yaml")
    base_config_template_name = os.path.join(base_yaml_config_path, "base_template_eval_config.yaml")
    base_config_yaml = read_yaml(yaml_file_name=base_config_template_name)

    for model_name in model_names:
        cleaned_model_name = model_name.replace("/", "-")
        model_type = get_model_type_from_name(model_name=model_name)

        #for task_name, prompt_version_name in selected_tasks_with_prompt_version_names:
        for task_config in task_config_list:
            for num_fewshot in num_fewshots:
                for seed in seeds:
                    for few_shot_sampling_strategy in few_shot_sampling_strategies:
                        new_config = base_config_yaml.copy()
                        models_args = base_models_args.format(model_name=model_name)
                        updated_model_args = update_model_args(models_args)
                        new_config["model"] = model_type
                        new_config["model_args"] = updated_model_args
                        #new_config["tasks"] = task_name
                        #new_config["prompt_version_per_task"] = prompt_version_name
                        new_config["task_configs"] = asdict(task_config)
                        new_config["num_fewshot"] = num_fewshot
                        new_config["seed"] = seed
                        new_config["fewshot_sampling"] = few_shot_sampling_strategy
                        new_config_yaml_out_file_name = config_out_yaml_template.format(
                            prompt_version=task_config.prompt_version,
                            model_name=cleaned_model_name,
                            num_fewshot=num_fewshot,
                            seed=seed,
                            fewshot_sampling=few_shot_sampling_strategy)
                        new_config_yaml_out_file_path = os.path.join(base_yaml_config_path, cleaned_model_name,
                                                                     f"task_{task_config.task_name}",
                                                                     f"num_fewshot_{num_fewshot}",
                                                                     f"fewshot_strategy_{few_shot_sampling_strategy}",
                                                                     f"seed_{seed}")
                        write_yaml(data=new_config, yaml_out_file_directory=new_config_yaml_out_file_path,
                                   yaml_out_file_name=new_config_yaml_out_file_name)


if __name__ == "__main__":
    main()
