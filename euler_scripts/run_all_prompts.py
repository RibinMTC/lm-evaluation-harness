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
    # "palm2",
    # "bigscience/bloomz-7b1-mt",
    # "tiiuae/falcon-7b-instruct",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "fangloveskari/ORCA_LLaMA_70B_QLoRA",
    # "garage-bAInd/Platypus2-70B-instruct",
    # "LeoLM/leo-hessianai-7b",
    # "LeoLM/leo-hessianai-13b",
    # "LeoLM/leo-hessianai-7b-chat",
    # "LeoLM/leo-hessianai-13b-chat",
    # "mtc/NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged",
    # "mtc/NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged",
    # "mtc/NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged",
    # "google/seahorse-large-q1",
    # "google/seahorse-large-q2",
    # "google/seahorse-large-q3",
    # "google/seahorse-large-q4",
    # "google/seahorse-large-q5",
    # "google/seahorse-large-q6",
    # "google/seahorse-xxl-q1",
    # "google/seahorse-xxl-q2",
    # "google/seahorse-xxl-q3",
    # "google/seahorse-xxl-q4",
    # "google/seahorse-xxl-q5",
    # "google/seahorse-xxl-q6",
]

# TODO: CHANGE PARAMETERS + NAME
experiment_name = "summchain-ablation-" + ''.join(
    random.choice(string.ascii_lowercase) for i in range(5))
dataset_names = [
    # "WikinewsTrunc3584SUBS",
    # "WikinewsTrunc3584",
    # "MultinewsTrunc3584",

    # TODO: NEW START
    # "WikiDi090S1024",
    # "WikiDi090S1536",
    # "WikiDi091SW1024",
    # "WikiDi091SW1536",
    # "WikiDi092SW1024",
    # "WikiDi092SW1536",
    ## "MDSFCO_WikiCl0S2048_100_8b",
    # "WikiCl1SSimDynW2048",
    # "WikiCl2SSimDynW2048",
    # "WikiChN1024",
    # "WikiChN1536",
    # "WikiChN1S1024",
    # "WikiChN1S1536",
    # "WikiChN2S1024",
    # "WikiChN2S1536",

    # "Wikinewsi0",
    # "Wikinewsi1",
    # "Wikinewsi2",
    # "Wikinewsi3",
    # "Wikinewsi4",
    # "Wikinewsi5",
    # "Wikinewsi6",
    # "Wikinewsi7",
    # "Wikinewsi8",
    # "Wikinewsi9", # CURRENTLY RUNNING
    # "Wikinewsi10",
    # "Wikinewsi11",
    # "Wikinewsi12",
    # TODO: NEW END

    # "MultiCD040SSimDyn1024",
    # "MultiCD040SSimDyn1536", # TODO: Wait to see if finishes (39554644)
    #
    # "WikiCl0SSimDyn1536",
    # "WikiCl1SSimDyn1536",
    # "WikiCl2SSimDyn1536",

    # "WikiCD040SSimDyn1024",
    # "WikiCD040SSimDyn1536",
    # "WikiCD041SSimDyn1024",
    # "WikiCD042SSimDyn1024",
    # "WikiCD043SSimDyn1024",
    # "WikiCD041SSimDyn1536",
    # "WikiCD042SSimDyn1536",
    # "WikiCD043SSimDyn1536",
    #
    # "WikiCD050SSimDyn1536",
    # "WikiCD060SSimDyn1536",
    # "WikiCD051SSimDyn1536",
    # "WikiCD061SSimDyn1536",
    # "WikiCD052SSimDyn1536",
    # "WikiCD062SSimDyn1536",
    # "WikiCD053SSimDyn1536",
    # "WikiCD063SSimDyn1536",

    ## Repeat experiments
    # "0_Llama7b",
    # "0_Llama70b",
    # "1_Llama70b",
    # "2_Llama70b",
    # "3_Llama70b",
    # "4_Llama70b",

    ## SEAHORSE evaluation
    # "base_datasets",
    # "fco_experiments",

    ## summarization-chain
    # "WikinewsS2",
    # "WikinewsClustS2",
    # "WikinewsClustDistS2",
    # Actual summarization chain
    # "WikinewsCDS4i0",
    # "WikinewsCDS4i1",
    # "WikinewsCDS4i2",
    # "WikinewsCDS4i3",
    # "WikinewsCDS4i4",
    # "WikinewsCDS4i5",
    # "WikinewsCDS4i6",
    # "WikinewsCDS4i7",
    # "WikinewsCDS4i8",
    # "WikinewsCDS4i9",
    # "WikinewsCDS4i10",
    # "WikinewsCDS4i11", # TODO: CONTINUE HERE
    # "WikinewsCDS4i12",

    "WikiLe1SW1536",

    # "20Minuten"
    # "20min0"
    ## FCO Experiments
    # Cheat
    # "WikiCh1024", "WikiCh1536",
    # Lead
    # "WikiLe1024", "WikiLe1536",
    # Lead - 1-shot (20Minuten examples)
    # "WikiLe1S21024", "WikiLe1S21536",
    # Lead - 1-shot (Wikinews examples)
    # TODO-FUTURE: "WikiLe1SW1024", "WikiLe1SW1536",
    # Random
    # "WikiRa1024", "WikiRa1536",
    # Random - 1-shot (20Minuten examples)
    # "WikiRa1S21024", "WikiRa1S21536",
    # Random - 1-shot (Wikinews examples)
    # "WikiRa1SW1024", "WikiRa1SW1536",
    # Clustering - 0-shot
    # "WikiCl0N1024", "WikiCl0N1536", "WikiCl0N2048",
    # Clustering - 1-shot (20Minuten examples)
    # "WikiCl1N21024", "WikiCl1N21536", "WikiCl1N22048",
    # Clustering - 2-shot (20Minuten examples)
    # "WikiCl2S21024", "WikiCl2S21536", "WikiCl2S22048",
    # Clustering - 1-shot (Wikinews examples)
    # "WikiCl1SW1024", "WikiCl1SW1536", # TODO-FUTURE: "WikiCl1SW2048",
    # Clustering - 2-shot (Wikinews examples)
    # "WikiCl2SW1024", "WikiCl2SW1536",
    # DistanceMMR - 0-shot
    # "WikiDi0S1024", "WikiDi0S1536",
    # DistanceMMR - 1-shot (20Minuten examples)
    # "WikiDi1S21024", "WikiDi1S21536",
    # DistanceMMR - 2-shot (20Minuten examples)
    # "WikiDi2S21024", "WikiDi2S21536",
    # DistanceMMR - 1-shot (Wikinews examples)
    # "WikiDi1SW1024", "WikiDi1SW1536",
    # DistanceMMR - 2-shot (Wikinews examples)
    # "WikiDi2SW1024", # TODO-FUTURE:  "WikiDi2SW1536",

    # "WikinewsSplitS2OP41", "WikinewsSplitS2SP41"
    # "WikinewsClust1R", "WikinewsClust1O", "WikinewsClust1C", "WikinewsClust3R", "WikinewsClust3O", "WikinewsClust3C", "WikinewsClust5R", "WikinewsClust5O", "WikinewsClust5C", "WikinewsClust10R", "WikinewsClust10O", "WikinewsClust10C"
    # "WikinewsSent1L00", "WikinewsSent1L05", "WikinewsSent3L00", "WikinewsSent3L05", "WikinewsSent5L00", "WikinewsSent5L05", "WikinewsSent10L00", "WikinewsSent10L05"

    # "20Minuten",

    # SEAHORSE TASK DATASETS
    # "falcon7b1",
    # "falcon7b2",
    # "bloomz7b1",
    # "bloomz7b2",
    # "palm2",
    # "gpt4",
    # "platypus2",
    # "orcallama2",
    # "leolm7b",
    # "leolm13b",
    # # "llama27b1", # EXTRA
    # "llama27b2",
    # "llama27b3",
    # "llama213b1",
    # "llama213b2",
    # "llama270b1",
    # "llama270b2a",
    # "llama270b2b",
    # "llama270b3",
    # "llama270b4",
    # "llama270b5a",
    # "llama270b5b",
    # "llama270b6a",
    # "llama270b6b",
    # "llama270b7a",
    # "llama270b7b",
    # "llama270b8a",
    # "llama270b8b",
    # "llama270b9a",
    # "llama270b9b",
    # "llama270b10a",
    # "llama270b10b",
    # "llama270b11a",
    # "llama270b11b",
    # "llama270b12a",
    # "llama270b12b",
    # "llama270b13a",
    # "llama270b13b",
    # "llama270b14a",
    # "llama270b14b",
    # "llama270b15",
    # # "llama270b16", # NOT DONE ON PURPOSE
    # "llama270b17",
    # "llama270b18",
    # "missing",
    # "missing2",
    # "missing3",
    # "missing4",
    # "missing5",
    # "missing6", # WAITING ON RESULTS
    # "missing7",
    # "missing8",
    # "missing9",
    # "missing10",
    # TODO: "missing11",

    # "20Minuten",

    # SEAHORSE METRIC EVALUATION ON TEST SET
    # "testq1",
    # "testq2",
    # "testq3",
    # "testq4",
    # "testq5",
    # "testq6",
]  # ["20Minuten", "Wikinews"], ["20min0", "20min1", "20min2", "20min3"]
prompt_versions = [100] # 22, 23, 25, 40, 41, 42
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
# 22,23,40,41,42,43,44,45,46,47,48,49,
#
# 23, 5, 7, 9, 11, 13, 15, 17, 19, 22, 42, 2, 40, 41, 4
task_base_names = [
    # "seahorse_",
    # "RepeatExperimentBugfix_",

    # "SummarizationTask_",
    # "SummSample_",
    # "MDSChain_",
    # "MDSChainAbl_",
    # "MDS_",
    "MDSFCO_",
]  # ["SummLtM_", "SummLtMDe_", "SummarizationTask_", "SummFewshot{num_fewshot}_", "MDSSumm_", "SummLtM1_", "SummLtM2_"]

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


gpt-4-20min- (FINISHED) / PALM2
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
PromptVersions LtM1: 22, 31, 33 -> because others predicted too much non-german text
All LtM2 Prompt Versions: 34, 35, 36, 37 -> but only use 35 (suffix), 37 (prefix) (because german prompts)
prompt_versions = [35, 37] # 35: summary instruction at end, 37: summary instruction at beginning
task_base_names = ["SummLtM2_"]
dataset_names = [
    '20minLtm2p22S', '20minLtm2p22E'
    '20minLtm2p31S', '20minLtm2p31E'
    '20minLtm2p33S', '20minLtm2p33E'
]


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
dataset_names = ["20minTS250"]
prompt_versions = [1,2] # [1,2,3,4,5]
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
        "palm2": "palm2",
        "meta-llama/Llama-2-7b-chat-hf": "hf-causal-experimental",
        "meta-llama/Llama-2-13b-chat-hf": "hf-causal-experimental",
        "meta-llama/Llama-2-70b-chat-hf": "hf-causal-experimental",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "hf-causal-experimental",
        "bigscience/bloomz-7b1-mt": "hf-causal-experimental",
        "tiiuae/falcon-7b-instruct": "hf-causal-experimental",
        "tiiuae/falcon-40b-instruct": "hf-causal-experimental",
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": "hf-causal-experimental",
        "garage-bAInd/Platypus2-70B-instruct": "hf-causal-experimental",
        "LeoLM/leo-hessianai-7b": "hf-causal-experimental",
        "LeoLM/leo-hessianai-13b": "hf-causal-experimental",
        "LeoLM/leo-hessianai-7b-chat": "hf-causal-experimental",
        "LeoLM/leo-hessianai-13b-chat": "hf-causal-experimental",
        "mtc/NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged": "hf-causal-experimental",
        "mtc/NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged": "hf-causal-experimental",
        "mtc/NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged": "hf-causal-experimental",
        "google/seahorse-large-q1": "hf-seq2seq",
        "google/seahorse-large-q2": "hf-seq2seq",
        "google/seahorse-large-q3": "hf-seq2seq",
        "google/seahorse-large-q4": "hf-seq2seq",
        "google/seahorse-large-q5": "hf-seq2seq",
        "google/seahorse-large-q6": "hf-seq2seq",
        "google/seahorse-xxl-q1": "hf-seq2seq",
        "google/seahorse-xxl-q2": "hf-seq2seq",
        "google/seahorse-xxl-q3": "hf-seq2seq",
        "google/seahorse-xxl-q4": "hf-seq2seq",
        "google/seahorse-xxl-q5": "hf-seq2seq",
        "google/seahorse-xxl-q6": "hf-seq2seq",
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
        "palm2": "08:00",
        "meta-llama/Llama-2-7b-chat-hf": "4:00",
        "meta-llama/Llama-2-13b-chat-hf": "18:00",
        "meta-llama/Llama-2-70b-chat-hf": "24:00",  # 24
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "12:00",  # 24
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": "30:00",
        "garage-bAInd/Platypus2-70B-instruct": "50:00",
        "LeoLM/leo-hessianai-7b": "08:00",
        "LeoLM/leo-hessianai-13b": "12:00",
        "LeoLM/leo-hessianai-7b-chat": "08:00",
        "LeoLM/leo-hessianai-13b-chat": "12:00",
        "bigscience/bloomz-7b1-mt": "04:00",
        "tiiuae/falcon-7b-instruct": "04:00",
        "tiiuae/falcon-40b-instruct": "24:00",
        "mtc/NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged": "24:00",
        "mtc/NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged": "24:00",
        "mtc/NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged": "24:00",
        "google/seahorse-large-q1": "24:00",
        "google/seahorse-large-q2": "24:00",
        "google/seahorse-large-q3": "24:00",
        "google/seahorse-large-q4": "24:00",
        "google/seahorse-large-q5": "24:00",
        "google/seahorse-large-q6": "24:00",
        "google/seahorse-xxl-q1": "24:00",
        "google/seahorse-xxl-q2": "24:00",
        "google/seahorse-xxl-q3": "24:00",
        "google/seahorse-xxl-q4": "24:00",
        "google/seahorse-xxl-q5": "24:00",
        "google/seahorse-xxl-q6": "24:00",
    },
    "gpu": {
        "default": "rtx_3090",
        "gpt-4": "rtx_3090",
        "palm2": "rtx_2080_ti",
        "meta-llama/Llama-2-7b-chat-hf": "rtx_3090",
        "meta-llama/Llama-2-13b-chat-hf": "rtx_3090",
        "meta-llama/Llama-2-70b-chat-hf": "rtx_4090",  # 3x a100-pcie-40gb, 4-6x rtx_3090 or rtx_4090
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "rtx_3090",
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": "a100-pcie-40gb",
        "garage-bAInd/Platypus2-70B-instruct": "rtx_3090",
        "LeoLM/leo-hessianai-7b": "rtx_3090",
        "LeoLM/leo-hessianai-13b": "rtx_3090",
        "LeoLM/leo-hessianai-7b-chat": "rtx_3090",
        "LeoLM/leo-hessianai-13b-chat": "rtx_3090",
        "bigscience/bloomz-7b1-mt": "rtx_4090",
        "tiiuae/falcon-7b-instruct": "rtx_4090",
        "tiiuae/falcon-40b-instruct": "a100-pcie-40gb",
        "mtc/NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged": "rtx_4090",
        "mtc/NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged": "rtx_4090",
        "mtc/NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged": "rtx_4090",
        "google/seahorse-large-q1": "rtx_3090",
        "google/seahorse-large-q2": "rtx_3090",
        "google/seahorse-large-q3": "rtx_3090",
        "google/seahorse-large-q4": "rtx_3090",
        "google/seahorse-large-q5": "rtx_3090",
        "google/seahorse-large-q6": "rtx_3090",
        "google/seahorse-xxl-q1": "rtx_3090",
        "google/seahorse-xxl-q2": "rtx_3090",
        "google/seahorse-xxl-q3": "rtx_3090",
        "google/seahorse-xxl-q4": "rtx_3090",
        "google/seahorse-xxl-q5": "rtx_3090",
        "google/seahorse-xxl-q6": "rtx_3090",
    },
    "num_gpus": {
        "default": 1,
        "gpt-4": 1,
        "palm2": 1,
        "meta-llama/Llama-2-7b-chat-hf": 1,
        "meta-llama/Llama-2-13b-chat-hf": 2,
        "meta-llama/Llama-2-70b-chat-hf": 6,  # 3x a100-pcie-40gb, 6x rtx_3090 or rtx_4090
        "mistralai/Mixtral-8x7B-Instruct-v0.1": 4,
        "fangloveskari/ORCA_LLaMA_70B_QLoRA": 3,
        "garage-bAInd/Platypus2-70B-instruct": 8,
        "LeoLM/leo-hessianai-7b": 1,
        "LeoLM/leo-hessianai-13b": 3,
        "LeoLM/leo-hessianai-7b-chat": 1,
        "LeoLM/leo-hessianai-13b-chat": 3,
        "bigscience/bloomz-7b1-mt": 2,
        "tiiuae/falcon-7b-instruct": 1,
        "tiiuae/falcon-40b-instruct": 4,
        "mtc/NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged": 1,
        "mtc/NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged": 1,
        "mtc/NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged": 1,
        "google/seahorse-large-q1": 1,
        "google/seahorse-large-q2": 1,
        "google/seahorse-large-q3": 1,
        "google/seahorse-large-q4": 1,
        "google/seahorse-large-q5": 1,
        "google/seahorse-large-q6": 1,
        "google/seahorse-xxl-q1": 2,
        "google/seahorse-xxl-q2": 2,
        "google/seahorse-xxl-q3": 2,
        "google/seahorse-xxl-q4": 2,
        "google/seahorse-xxl-q5": 2,
        "google/seahorse-xxl-q6": 2,
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
model_args_schema = "pretrained={model},max_gen_toks=512,trust_remote_code=True,use_accelerate=True{temperature_suffix}{precision_suffix}"
model_args_schema_gpt4 = "engine=gpt-4"
model_args_schema_palm2 = "engine=text-bison@001"  # "engine=models/text-bison-001"
model_args_schema_seahorse = "pretrained={model},trust_remote_code=True,use_accelerate=True"

"""
    Build the configurations
"""
# Create a list of all possible combinations of the parameters
combinations = list(
    itertools.product(models, temperature_values, precision_values, dataset_names, prompt_versions, task_base_names,
                      num_fewshot_list))
config_list = []

# Iterate over each combination and create a config dictionary
for combination in combinations:
    model, tempVal, precision, dataset, promptVersion, taskBaseName, num_fewshot = combination

    # prepare the values
    task_temp_suffix = inferable_args["task_temp_suffix"][tempVal] if tempVal in inferable_args["task_temp_suffix"] else \
        inferable_args["task_temp_suffix"]["default"]
    if tempVal == 0:
        temp_suffix_model_args = ""
    else:
        temp_suffix_model_args = f",do_sample=True,temperature={tempVal}"
    precision_suffix = inferable_args["precision"][precision] if precision in inferable_args["precision"] else \
        inferable_args["precision"]["default"]
    precision_task_suffix = "" if precision == "" else f"_{precision}"

    # insert num_fewshot into task string if necessary
    if "num_fewshot" in taskBaseName:
        taskBaseName = taskBaseName.format(num_fewshot=num_fewshot)

    # Build the arguments (eval_config)
    model_config = inferable_args["model"][model] if model in inferable_args["model"] else inferable_args["model"][
        "default"]
    if inferable_args["model"][model] == "gpt4":
        model_args = model_args_schema_gpt4
    elif inferable_args["model"][model] == "palm2":
        model_args = model_args_schema_palm2
    elif inferable_args["model"][model] == "hf-causal-experimental" and "seahorse" in taskBaseName:
        model_args = model_args_schema_seahorse.format(model=model)
    elif inferable_args["model"][model] == "hf-causal-experimental":
        model_args = model_args_schema.format(model=model, temperature_suffix=temp_suffix_model_args,
                                              precision_suffix=precision_suffix)
    else:
        raise NotImplementedError(f"Model {model} not implemented yet.")
    task_name = task_name_schema.format(task_base_name=taskBaseName, dataset_name=dataset,
                                        task_temp_suffix=task_temp_suffix, task_prompt_suffix=f"_{promptVersion}",
                                        precision=f"{precision_task_suffix}")
    # Build the arguments (euler_config)
    run_duration_hours = inferable_args["run_duration_hours"][model] if model in inferable_args[
        "run_duration_hours"] else inferable_args["run_duration_hours"]["default"]
    gpu = inferable_args["gpu"][model] if model in inferable_args["gpu"] else inferable_args["gpu"]["default"]
    num_gpus = inferable_args["num_gpus"][model] if model in inferable_args["num_gpus"] else inferable_args["num_gpus"][
        "default"]

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
