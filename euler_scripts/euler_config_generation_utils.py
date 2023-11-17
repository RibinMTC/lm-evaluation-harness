import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict

import yaml

from lm_eval.utils import TaskConfig

BASE_MODEL_ARGS = "pretrained={model_name},trust_remote_code=False,use_accelerate=True,dtype=bfloat16"
CONFIG_OUT_YAML_TEMPLATE = ("{prompt_version}_{model_name}_num_fewshot_{num_fewshot}-{fewshot_sampling}_seed_{"
                            "seed}_eval_config.yaml")

MODEL_NAME_TO_TYPE_MAP = {
    "NousResearch/Llama-2": "hf-causal-experimental",
    "google/flan-ul2": "hf-seq2seq",
    "LeoLM/leo": "hf-causal-experimental",
    "mtc/LeoLM": "hf-causal-experimental",
    "ehartford/dolphin": "hf-causal-experimental",
    "mtc/ehartford-dolphin": "hf-causal-experimental",
    "meta-llama": "hf-causal-experimental",
    "mistralai/Mistral-7B-Instruct-v0.1": "hf-causal-experimental",
    "lmsys/vicuna-13b-v1.5-16k": "hf-causal-experimental"
}


@dataclass
class UIConfigValues:
    base_config_dir: str
    model_name_values: List[str]
    few_shot_values: List[int]
    tasks_with_prompt_templates: Dict[str, str]
    seeds: List[int]
    few_shot_sampling_strategies: List[str]


@dataclass
class SelectedUIConfigValues:
    base_config_dir: str
    model_name_values: List[str]
    few_shot_values: List[int]
    tasks_config_list: List[TaskConfig]
    seeds: List[int]
    few_shot_sampling_strategies: List[str]
    load_in_8bit: bool
    use_flash_attention_2: bool


def read_json(json_file_name: str):
    with open(json_file_name, 'r') as file:
        json_file = json.load(file)
    return json_file


def get_model_prompt_versions(model_name: str, prompt_template_json: Dict) -> List:
    for prompt_model_name, prompts in prompt_template_json.items():
        if prompt_model_name in model_name:
            return list(prompts.keys())
    print(f"The model {model_name} has no corresponding prompts defined in the template")
    return []


def get_model_type_from_name(model_name: str):
    for model_prefix_name, model_type in MODEL_NAME_TO_TYPE_MAP.items():
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


def update_model_args(models_args: str, load_in_8bit: bool, use_flash_attention_2: bool) -> str:
    load_in_8bit_string = "load_in_8bit="
    use_flash_attention_2_string = "use_flash_attention_2="
    updated_model_args = update_string(original_string=models_args, string_pattern=load_in_8bit_string,
                                       string_value=str(load_in_8bit))
    updated_model_args = update_string(original_string=updated_model_args, string_pattern=use_flash_attention_2_string,
                                       string_value=str(use_flash_attention_2))
    return updated_model_args


def save_config_files(selected_ui_config_values: SelectedUIConfigValues):
    base_config_template_name = os.path.join(selected_ui_config_values.base_config_dir,
                                             "base_template_eval_config.yaml")
    base_config_yaml = read_yaml(yaml_file_name=base_config_template_name)

    for model_name in selected_ui_config_values.model_name_values:
        cleaned_model_name = model_name.replace("/", "-")
        model_type = get_model_type_from_name(model_name=model_name)

        # for task_name, prompt_version_name in selected_tasks_with_prompt_version_names:
        for task_config in selected_ui_config_values.tasks_config_list:
            for num_fewshot in selected_ui_config_values.few_shot_values:
                for seed in selected_ui_config_values.seeds:
                    for few_shot_sampling_strategy in selected_ui_config_values.few_shot_sampling_strategies:
                        new_config = base_config_yaml.copy()
                        models_args = BASE_MODEL_ARGS.format(model_name=model_name)
                        updated_model_args = update_model_args(models_args,
                                                               load_in_8bit=selected_ui_config_values.load_in_8bit,
                                                               use_flash_attention_2=selected_ui_config_values.use_flash_attention_2)
                        new_config["model"] = model_type
                        new_config["model_args"] = updated_model_args
                        # new_config["tasks"] = task_name
                        # new_config["prompt_version_per_task"] = prompt_version_name
                        new_config["task_configs"] = [asdict(task_config)]
                        new_config["num_fewshot"] = num_fewshot
                        new_config["seed"] = seed
                        new_config["fewshot_sampling"] = few_shot_sampling_strategy
                        new_config_yaml_out_file_name = CONFIG_OUT_YAML_TEMPLATE.format(
                            prompt_version=task_config.prompt_version,
                            model_name=cleaned_model_name,
                            num_fewshot=num_fewshot,
                            seed=seed,
                            fewshot_sampling=few_shot_sampling_strategy)
                        new_config_yaml_out_file_path = os.path.join(selected_ui_config_values.base_config_dir,
                                                                     cleaned_model_name,
                                                                     f"task_{task_config.task_name}",
                                                                     f"num_fewshot_{num_fewshot}",
                                                                     f"fewshot_strategy_{few_shot_sampling_strategy}",
                                                                     f"seed_{seed}")
                        write_yaml(data=new_config, yaml_out_file_directory=new_config_yaml_out_file_path,
                                   yaml_out_file_name=new_config_yaml_out_file_name)
