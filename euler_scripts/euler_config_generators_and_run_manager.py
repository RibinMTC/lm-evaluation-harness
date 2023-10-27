import json
import os
from typing import Dict, List

import yaml

model_names = [
    "LeoLM/leo-hessianai-13b"]
num_fewshots = [8, 16, 24]
few_shot_sampling_strategies = ["stratified"]
seeds = [42]
selected_prompt_version_names = ["all_labels_german"]

prompt_to_task_map = {
    "faithfulness": "faithfulness_benchmark_final_swisstext23_benchmark_faithful",
    "intrinsic": "faithfulness_benchmark_final_swisstext23_benchmark_intrinsic",
    "extrinsic": "faithfulness_benchmark_final_swisstext23_benchmark_extrinsic",
    "all_labels": "faithfulness_benchmark_final_swisstext23_multi_label"
}

model_name_to_type_map = {
    "NousResearch/Llama-2": "hf-causal-experimental",
    "google/flan-ul2": "hf-seq2seq",
    "LeoLM/leo": "hf-causal-experimental",
    "mtc/LeoLM": "hf-causal-experimental"
}


def get_task_name_from_prompt(prompt_name: str) -> str:
    for prompt, task_name in prompt_to_task_map.items():
        if prompt in prompt_name:
            return task_name
    raise ValueError(f"Prompt: {prompt_name} could not be mapped to a task")


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


def main():
    base_models_args = "pretrained={model_name},trust_remote_code=False,use_accelerate=True"
    base_yaml_config_path = "configs/faithfulness_benchmark_final_configs"
    config_out_yaml_template = ("{prompt_version}_{model_name}_num_fewshot_{num_fewshot}-{fewshot_sampling}_seed_{"
                                "seed}_eval_config.yaml")
    base_config_template_name = os.path.join(base_yaml_config_path, "base_template_eval_config.yaml")
    base_config_yaml = read_yaml(yaml_file_name=base_config_template_name)

    for model_name in model_names:
        cleaned_model_name = model_name.replace("/", "-")
        model_type = get_model_type_from_name(model_name=model_name)

        for prompt_version_name in selected_prompt_version_names:
            for num_fewshot in num_fewshots:
                for seed in seeds:
                    for few_shot_sampling_strategy in few_shot_sampling_strategies:
                        new_config = base_config_yaml.copy()
                        models_args = base_models_args.format(model_name=model_name)
                        task_name = get_task_name_from_prompt(prompt_name=prompt_version_name)
                        new_config["model"] = model_type
                        new_config["model_args"] = models_args
                        new_config["tasks"] = task_name
                        new_config["prompt_version_per_task"] = prompt_version_name
                        new_config["num_fewshot"] = num_fewshot
                        new_config["seed"] = seed
                        new_config["fewshot_sampling"] = few_shot_sampling_strategy
                        new_config_yaml_out_file_name = config_out_yaml_template.format(
                            prompt_version=prompt_version_name,
                            model_name=cleaned_model_name,
                            num_fewshot=num_fewshot,
                            seed=seed,
                            fewshot_sampling=few_shot_sampling_strategy)
                        new_config_yaml_out_file_path = os.path.join(base_yaml_config_path, cleaned_model_name,
                                                                     f"num_fewshot_{num_fewshot}",
                                                                     f"fewshot_strategy_{few_shot_sampling_strategy}",
                                                                     f"seed_{seed}")
                        write_yaml(data=new_config, yaml_out_file_directory=new_config_yaml_out_file_path,
                                   yaml_out_file_name=new_config_yaml_out_file_name)


if __name__ == "__main__":
    main()
