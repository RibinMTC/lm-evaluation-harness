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
    # "gpt-4",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    # "fangloveskari/ORCA_LLaMA_70B_QLoRA",
    # "garage-bAInd/Platypus2-70B-instruct",
]

# TODO: CHANGE PARAMETERS + NAME
experiment_name = "mds-simple-" + ''.join(random.choice(string.ascii_lowercase) for i in range(5))
dataset_names = ["WikinewsClean"]  # ["20Minuten", "Wikinews"], ["20min0", "20min1", "20min2", "20min3"]
prompt_versions = [52]  # [1, 2, 3, 4, 5]
task_base_names = ["MDSSumm_"]  # ["SummLtM_", "SummLtMDe_", "SummarizationTask_", "SummFewshot{num_fewshot}_", "MDSSumm_", "SummLtM1_", "SummLtM2_"]

temperature_values = [0]  # [0, 0.1, 0.5, 1.0]
precision_values = ["8b"]  # ["", "8b"]
num_fewshot_list = [0]  # [0, 1, 2] # [0] #

# TODO: MDS-split-input-documents
"""
MDS-split-input-documents- (SCHEDULED) # TODO
dataset_names = ["WikinewsSplit"]
prompt_versions = [2]
task_base_names = ["MDS2S_"]

MDS-split-input-documents-stage2- (SCHEDULED)
dataset_names = ["WikinewsSplitS2O", "WikinewsSplitS2S"]
prompt_versions = [52]
task_base_names = ["MDS2S_"]


gpt-4-20min- (FINISHED)
dataset_names = ["20Minuten"]
prompt_versions = [1,2,3,4,5]
task_base_names = ["SummSample_"]

ltm-experiment-stage-1- (SCHEDULED)
prompt_versions = [21, 22, 30, 31, 32, 33]
task_base_names = ["SummLtM1_"]

ltm-experiment-stage-2- => UPLOAD NEW PRE-PROCESSED DATASET FROM OUTPUTS FROM STAGE 1
=> Make sure to only include the main points and not the old prompt as well
=> Make a second version leaving the old prompt in there as well??? (ASK SOMEONE)
-> TODO: dataset_names (for pre-processed dataset)
prompt_versions = [23, 24, 34, 35] # 36?, 37?
task_base_names = ["SummLtM2_"]

mds-simple- (FINISHED)
prompt_versions = [50,51]
task_base_names = ["MDSSumm_"]
dataset_names = ["Wikinews"]

mds-simple-shuffle-annotations-and-variants- (SCHEDULED)
... MDS Experiment: make annotated version stating: "Article 1: ..., Article 2: ..., ..."
... A = article index annotation, S = shuffled
prompt_versions = [52]
task_base_names = ["MDSSumm_"]
dataset_names = ["Wikinews", "WikinewsSimple", "WikinewsSimpleS", "WikinewsSimpleA", "WikinewsSimpleAS", "WikinewsSC32", "WikinewsSC64", "WikinewsSC128", "WikinewsSC256", "WikinewsSC512"]
dataset_names [ "WikinewsSCS2", "WikinewsSCS4", "WikinewsSCS8", "WikinewsSCS16", "WikinewsSCS32"] -> TODO


fewshot-experiment- (SCHEDULED)
prompt_versions = [2] # [1,2,3,4,5]
task_base_names = ["SummFewshot{num_fewshot}_"]
num_fewshot_list = [0,1,2]

versions-experiment- (SCHEDULED)
=> Different llama2 versions
prompt_versions = [1,2,3,4,5]
task_base_names = ["SummarizationTask_"]
NOTE:
- SummarizationTask_20Minuten_1_8b_MODEL_fangloveskari-ORCA_LLaMA_70B_QLoRA_0-SHOT
    - -> requires 54 hours???
- SummarizationTask_20Minuten_2_8b_MODEL_fangloveskari-ORCA_LLaMA_70B_QLoRA_0-SHOT
    - -> requires 54 hours as well???
=> KILLED BOTH FOR NOW

"""

# CUSTOM LtM Prompting
# SummLtM1_20Minuten_21
# SummLtM1_20Minuten_22
# SummLtM2_20Minuten_23
# SummLtM2_20Minuten_24

# Element-aware summarization with LLMs -> CoT Summarization paper using LtM
# SummLtM1_20Minuten_30
# SummLtM1_20Minuten_31
# SummLtM2_20Minuten_32
# SummLtM2_20Minuten_33

# SummFewshot1_20Minuten_1
# SummFewshot1_20Minuten_2
# SummFewshot1_20Minuten_3
# SummFewshot1_20Minuten_4
# SummFewshot1_20Minuten_5

"""
    Definitions
"""
inferable_args = {
    "model": {
        "default": "hf-causal-experimental",
        "gpt-4": "gpt4",
        "meta-llama/Llama-2-7b-chat-hf": "hf-causal-experimental",
        "meta-llama/Llama-2-13b-chat-hf": "hf-causal-experimental",
        "meta-llama/Llama-2-70b-chat-hf": "hf-causal-experimental",
        "bigscience/bloomz-7b1-mt": "hf-causal-experimental",
        "tiiuae/falcon-7b-instruct": "hf-causal-experimental",
        "tiiuae/falcon-40b-instruct": "hf-causal-experimental",
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": "hf-causal-experimental",
        "garage-bAInd/Platypus2-70B-instruct": "hf-causal-experimental",
    },
    "task_temp_suffix": {
        "default": "",
        0: "",
        0.1: "_T01",
        0.5: "_T05",
        1.0: "_T10",
    },
    "run_duration_hours": {
        "default": "24:00",
        "gpt-4": "08:00",
        "meta-llama/Llama-2-7b-chat-hf": "4:00",
        "meta-llama/Llama-2-13b-chat-hf": "18:00",
        "meta-llama/Llama-2-70b-chat-hf": "30:00",
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": "30:00",
        "garage-bAInd/Platypus2-70B-instruct": "30:00",
        "bigscience/bloomz-7b1-mt": "08:00",
        "tiiuae/falcon-7b-instruct": "08:00",
        "tiiuae/falcon-40b-instruct": "24:00",
    },
    "gpu": {
        "default": "rtx_3090",
        "gpt-4": "rtx_3090",
        "meta-llama/Llama-2-7b-chat-hf": "rtx_3090",
        "meta-llama/Llama-2-13b-chat-hf": "rtx_3090",
        "meta-llama/Llama-2-70b-chat-hf": "a100-pcie-40gb",
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": "a100-pcie-40gb",
        "garage-bAInd/Platypus2-70B-instruct": "a100-pcie-40gb",
        "bigscience/bloomz-7b1-mt": "a100-pcie-40gb",
        "tiiuae/falcon-7b-instruct": "a100-pcie-40gb",
        "tiiuae/falcon-40b-instruct": "a100-pcie-40gb",
    },
    "num_gpus": {
        "default": 1,
        "gpt-4": 1,
        "meta-llama/Llama-2-7b-chat-hf": 2,
        "meta-llama/Llama-2-13b-chat-hf": 1,
        "meta-llama/Llama-2-70b-chat-hf": 3,
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": 3,
        "garage-bAInd/Platypus2-70B-instruct": 3,
        "bigscience/bloomz-7b1-mt": 1,
        "tiiuae/falcon-7b-instruct": 1,
        "tiiuae/falcon-40b-instruct": 4,
    },
    "precision": {
        "default": "",
        "8b": ",load_in_8bit=True",
    }
}
BASE_PROMPT_TEMPLATE = "configs/prompt_templates/summarization_base.json"
TMP_PROMPT_TEMPLATE = "configs/prompt_templates/{name}.json"
BASE_CONFIG = "configs/eval_config.yaml"
NEW_CONFIG_PATTERN = "configs/eval_config_{random_string}.yaml"
BASE_EULER_CONFIG = "lm_eval_euler_config.json"
TMP_EULER_CONFIG = "tmp_euler_config.json"

task_name_schema = "{task_base_name}{dataset_name}{task_temp_suffix}{task_prompt_suffix}{precision}"
model_args_schema = "pretrained={model},trust_remote_code=True,use_accelerate=True{temperature_suffix}{precision_suffix}"
model_args_schema_gpt4 = "engine=gpt-4"

"""
    Build the configurations
"""
# Create a list of all possible combinations of the parameters
combinations = list(itertools.product(models, temperature_values, precision_values, dataset_names, prompt_versions, task_base_names, num_fewshot_list))
config_list = []

# Iterate over each combination and create a config dictionary
for combination in combinations:
    model, tempVal, precision, dataset, promptVersion, taskBaseName, num_fewshot = combination

    # prepare the values
    task_temp_suffix = inferable_args["task_temp_suffix"][tempVal] if tempVal in inferable_args["task_temp_suffix"] else inferable_args["task_temp_suffix"]["default"]
    if tempVal == 0:
        temp_suffix_model_args = ""
    else:
        temp_suffix_model_args = f",do_sample=True,temperature={tempVal}"
    precision_suffix = inferable_args["precision"][precision] if precision in inferable_args["precision"] else inferable_args["precision"]["default"]
    precision_task_suffix = "" if precision == "" else f"_{precision}"

    # insert num_fewshot into task string if necessary
    if "num_fewshot" in taskBaseName:
        taskBaseName = taskBaseName.format(num_fewshot=num_fewshot)

    # Build the arguments (eval_config)
    model_config = inferable_args["model"][model] if model in inferable_args["model"] else inferable_args["model"]["default"]
    if inferable_args["model"][model] == "gpt4":
        model_args = model_args_schema_gpt4
    elif inferable_args["model"][model] == "hf-causal-experimental":
        model_args = model_args_schema.format(model=model, temperature_suffix=temp_suffix_model_args, precision_suffix=precision_suffix)
    else:
        raise NotImplementedError(f"Model {model} not implemented yet.")
    task_name = task_name_schema.format(task_base_name=taskBaseName, dataset_name=dataset, task_temp_suffix=task_temp_suffix, task_prompt_suffix=f"_{promptVersion}", precision=f"{precision_task_suffix}")
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
        "num_fewshot": num_fewshot,
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


# Adding null representation to avoid problems with generated yaml config files
class NullRepresenter:
    def __call__(self, repr, data):
        ret_val = repr.represent_scalar(u'tag:yaml.org,2002:null', u'null')
        return ret_val


my_represent_none = NullRepresenter()

# make a logs dir if it doesn't exist
if not os.path.exists("./logs"):
    os.makedirs("./logs")
# open a jsonl file and a log file to append the outputs to
with open(f"./logs/{experiment_name}.jsonl", "a") as jsonl_file, open(f"./logs/{experiment_name}.log", "a") as log_file:
    """
        Schedule the tasks
    """
    for config in config_list:
        print(f"Scheduling\n")
        pprint.pprint(config)

        # make a copy for the output file and print the config also to the log file
        config_out = config.copy()
        pprint.pprint(config_out, log_file)

        new_prompt_template = TMP_PROMPT_TEMPLATE.format(name=f"{config['tasks']}")
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
        yamlPrinter.representer.add_representer(type(None), my_represent_none)
        with open(f"../{BASE_CONFIG}") as f:
            y = yamlPrinter.load(f)
        y["model"] = config['model']
        y["model_args"] = config['model_args']
        y["tasks"] = config['tasks']
        y["prompt_version_per_task"] = config["prompt_version_per_task"]
        y["num_fewshot"] = config["num_fewshot"]

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
        # os.system(f"bash run_euler.sh {TMP_EULER_CONFIG}")
        # cmd = ['bash', 'run_dummy.sh', f"{TMP_EULER_CONFIG}"]
        cmd = ['bash', 'run_euler.sh', f"{TMP_EULER_CONFIG}"]
        output = subprocess.run(cmd, stdout=subprocess.PIPE)
        printable_out = output.stdout.decode('utf-8')

        # Find the line with the sbatch command and the consecutive line with the job id
        sbatch_line = None
        job_id_line = None
        sbatch_line_split = None
        submission_success = False
        for line in printable_out.split("\n"):
            if "sbatch" in line and sbatch_line is None:
                sbatch_line = line
            if "Submitted batch job" in line and job_id_line is None:
                job_id_line = line
                submission_success = True
        if sbatch_line is not None:
            # find the output file path (using -o flag)
            sbatch_line_split = sbatch_line.split(" ")
            out_idx = sbatch_line_split.index("-o")
            output_file_path = sbatch_line_split[out_idx + 1] if out_idx + 1 < len(sbatch_line_split) else None
        else:
            sbatch_line = ""
        if job_id_line is not None:
            # find the job id
            job_id = job_id_line.split(" ")[-1]
        else:
            job_id = ""
            job_id_line = ""

        # print the status to the log file
        log_file.write(f"\n\n{printable_out}")
        # print the status to the jsonl file
        config_out["sbatch_line"] = sbatch_line
        config_out["job_id_line"] = job_id_line
        config_out["job_id"] = job_id
        config_out["submission_success"] = submission_success
        config_out["output_file_path"] = output_file_path
        jsonl_file.write(json.dumps(config_out) + "\n")

        # print the status to the console
        print(f"\n{printable_out}")
        print(f"JOB ID: {job_id}")
        print(f"SUBMISSION: {'SUCCESS' if submission_success else 'FAILED'}")

        # Remove the temporary config files
        os.remove(f"tmp_euler_config.json")
        # os.remove(f"../{new_config}")
