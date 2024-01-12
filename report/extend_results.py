import collections
import itertools
import numpy as np
import pandas as pd
import random
import os
import datasets
from datasets import DatasetDict, Dataset
import tensorflow_datasets as tfds
# import lm_eval.metrics
# import lm_eval.models
# import lm_eval.tasks
# import lm_eval.base
# from lm_eval.utils import positional_deprecated, run_task_tests
import summ_eval as SummEval
from transformers import pipeline
import json
import pathlib
from typing import List, Dict, Union, Callable, Tuple
import sys
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from somajo import SoMaJo

somajo_tokenizer = SoMaJo("de_CMC", split_camel_case=True, split_sentences=True)

# BAD PRACTICE, I KNOW
sys.path.insert(0, os.path.abspath('..'))
from report import report

report.extract_dataset_and_task_name(
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_1_write_out_info.json")

SRC_FOLDER = "../results"
INT_FOLDER = "../results_bag_extended_input"
DST_FOLDER = "../results_bag_extended"

# google_prompt_suffix = "_google"

seahorse_prompt_template = """### System:
You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can.

### User: Given an article and its corresponding summary in {lang}, reply with True if all information in the summary originates from the article and reply False otherwise. 
Article: {article}
Summary: {summary}

### Assistant:"
"""
seahorse_prompt_empty = seahorse_prompt_template.format(lang="German", article="", summary="")
seahorse_prompt_empty_len = len(report.text_to_llama_bpe(seahorse_prompt_empty))

# seahorse_google_prompt_template = """premise: {article} hypothesis: {summary}"""
# seahorse_google_prompt_empty = seahorse_prompt_template.format(lang="German", article="", summary="")
# seahorse_google_prompt_empty_len = len(report.text_to_llama_bpe(seahorse_prompt_empty))

base_dataset_paths = {
    "20Minuten": "./resources/Datasets/20Minuten/test.json",
    "Wikinews": "./resources/Datasets/Wikinews/test.json",
    "Klexikon": "./resources/Datasets/Klexikon/test.json",
}

base_datasets_dst_file_name = "SEAHORSE_base_datasets"
process_files_falcon7b_1 = [
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_10_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_11_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_12_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_13_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_14_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_15_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_16_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_17_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_18_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_19_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_1_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_20_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_22_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_23_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_25_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_2_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_3_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_40_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_41_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_42_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_43_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_44_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_45_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_46_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_47_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_48_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_49_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_4_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_5_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_6_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_7_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_8_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_9_write_out_info.json",
]
process_files_falcon7b_2 = [
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_19_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_23_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_2_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_45_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_49_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_4_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T01_5_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_19_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_23_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_2_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_45_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_49_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_4_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T05_5_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_19_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_23_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_2_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_45_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_49_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_4_write_out_info.json",
    "tiiuae-falcon-7b-instruct/SummarizationTask_20Minuten_T10_5_write_out_info.json",
]
process_files_bloomz7b_1 = [
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_10_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_11_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_12_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_13_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_14_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_15_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_16_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_17_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_18_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_19_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_1_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_20_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_22_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_23_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_25_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_2_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_3_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_40_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_41_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_42_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_43_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_44_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_45_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_46_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_47_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_48_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_49_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_4_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_5_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_6_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_7_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_8_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_9_write_out_info.json",
]
process_files_bloomz7b_2 = [
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_19_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_1_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_20_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_23_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_2_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_3_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_45_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_49_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_4_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T01_5_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_19_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_1_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_20_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_23_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_2_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_3_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_45_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_49_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_4_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T05_5_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_19_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_1_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_20_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_23_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_2_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_3_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_45_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_49_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_4_write_out_info.json",
    "bigscience-bloomz-7b1-mt/SummarizationTask_20Minuten_T10_5_write_out_info.json",
]
process_files_palm2 = [
    "text-bison@001/SummSample_20Minuten_1_write_out_info.json",
    "text-bison@001/SummSample_20Minuten_2_write_out_info.json",
    "text-bison@001/SummSample_20Minuten_3_write_out_info.json",
    "text-bison@001/SummSample_20Minuten_4_write_out_info.json",
    "text-bison@001/SummSample_20Minuten_5_write_out_info.json",
    "text-bison@001/SummSmolSample_20Minuten_1_write_out_info.json",
]
process_files_gpt4 = [
    "gpt-4/SummSample_20Minuten_1_write_out_info.json",
    "gpt-4/SummSample_20Minuten_2_write_out_info.json",
    "gpt-4/SummSample_20Minuten_3_write_out_info.json",
    "gpt-4/SummSample_20Minuten_4_write_out_info.json",
    "gpt-4/SummSample_20Minuten_5_write_out_info.json",
]
process_files_platypus2 = [
    "garage-bAInd-Platypus2-70B-instruct/SummarizationTask_20min0_1_8b_write_out_info.json",
    "garage-bAInd-Platypus2-70B-instruct/SummarizationTask_20min0_2_8b_write_out_info.json",
    "garage-bAInd-Platypus2-70B-instruct/SummarizationTask_20min0_3_8b_write_out_info.json",
    "garage-bAInd-Platypus2-70B-instruct/SummarizationTask_20min0_4_8b_write_out_info.json",
]
process_files_orcallama2 = [
    "fangloveskari-ORCA_LLaMA_70B_QLoRA/SummarizationTask_20min0_1_8b_write_out_info.json",
    "fangloveskari-ORCA_LLaMA_70B_QLoRA/SummarizationTask_20min0_2_8b_write_out_info.json",
    "fangloveskari-ORCA_LLaMA_70B_QLoRA/SummarizationTask_20min0_3_8b_write_out_info.json",
    "fangloveskari-ORCA_LLaMA_70B_QLoRA/SummarizationTask_20min0_4_8b_write_out_info.json",
    "fangloveskari-ORCA_LLaMA_70B_QLoRA/SummarizationTask_20min0_5_8b_write_out_info.json",
]
process_files_leolm7b = [
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_1_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_22_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_23_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_25_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_2_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_40_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_41_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_42_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_43_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_44_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_45_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_46_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_47_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_48_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_49_write_out_info.json",
    "LeoLM-leo-hessianai-7b-chat/SummarizationTask_20Minuten_5_write_out_info.json",
]
process_files_leolm13b = [
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_1_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_22_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_23_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_25_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_2_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_40_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_41_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_42_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_43_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_44_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_45_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_46_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_47_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_48_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_49_write_out_info.json",
    "LeoLM-leo-hessianai-13b-chat/SummarizationTask_20Minuten_5_write_out_info.json",
]

process_files_llama2_7b_1 = [
    "meta-llama-Llama-2-7b-chat-hf/SummEmpty_20minSmol_1_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummEmpty_20minSmol_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummEmpty_20minSmol_4_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummFewshot_20Minuten_1_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummFewshot_20Minuten_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummFewshot_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummLtMDe_20Minuten_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummLtM_20Minuten_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummNonEmpty_20minSmol_1_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummNonEmpty_20minSmol_4_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummSample_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_10_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_11_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_12_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_13_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_14_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_15_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_16_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_17_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_18_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_19_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_1_write_out_info.json",
]
process_files_llama2_7b_2 = [
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_22_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_23_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_25_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_40_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_41_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_42_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_43_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_44_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_45_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_46_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_47_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_48_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_49_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_4_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_6_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_7_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_8_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_9_write_out_info.json",
]
process_files_llama2_7b_3 = [
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_19_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_23_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_45_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_49_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_4_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_19_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_23_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_45_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_49_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_4_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_19_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_23_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_45_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_49_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_4_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_5_write_out_info.json",
]
process_files_llama2_13b_1 = [
    "meta-llama-Llama-2-13b-chat-hf/MDSSumm_Wikinews_50_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/MDSSumm_Wikinews_51_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_10_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_11_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_12_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_13_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_14_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_15_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_16_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_17_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_18_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_19_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_1_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_20_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_22_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_23_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_25_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_3_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_40_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_41_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_42_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_43_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_44_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_45_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_46_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_47_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_48_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_49_write_out_info.json",
]
process_files_llama2_13b_2 = [
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_4_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_5_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_6_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_7_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_8_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_9_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_19_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_1_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_20_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_23_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_2_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_3_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_45_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_49_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_4_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T01_5_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_19_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_1_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_20_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_23_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_2_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_3_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_45_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_49_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_4_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T05_5_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_19_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_1_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_20_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_23_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_2_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_3_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_45_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_49_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_4_write_out_info.json",
    "meta-llama-Llama-2-13b-chat-hf/SummarizationTask_20Minuten_T10_5_write_out_info.json",
]
process_files_llama2_70b_1 = [
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2OP41_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2O_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2SP41_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2S_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplit_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplit_41_8b_write_out_info.json",
]
process_files_llama2_70b_2_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i3_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i4_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i5_100_8b_write_out_info.json",
]
process_files_llama2_70b_2_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS2i0_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i1_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i2_100_8b_write_out_info.json",
]
process_files_llama2_70b_3 = [
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i6_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i7_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i8_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i9_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i10_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsCDS4i11_100_8b_write_out_info.json",
]
# "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsS2_100_8b_write_out_info.json",
# "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsClustDistS2_100_8b_write_out_info.json",

process_files_llama2_70b_4 = [
    # "meta-llama-Llama-2-70b-chat-hf/MDSFCO_MultiCD040SSimDyn1024_100_8b_write_out_info.json",
    # "meta-llama-Llama-2-70b-chat-hf/MDSFCO_MultiCD040SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD040SSimDyn1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD040SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD041SSimDyn1024_100_8b_write_out_info.json"
]
process_files_llama2_70b_5_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD041SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD042SSimDyn1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD042SSimDyn1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_5_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD043SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD050SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD051SSimDyn1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_6_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD052SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD053SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD060SSimDyn1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_6_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD061SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD062SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCD063SSimDyn1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_7_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCh1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCh1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0N1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0N1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0N2048_100_8b_write_out_info.json"
]
process_files_llama2_70b_7_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1N21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1N21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1N22048_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1SSimDyn1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_8_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1SW1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2S21536_100_8b_write_out_info.json",
]
process_files_llama2_70b_8_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2S22048_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2SSimDyn1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2SW1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_9_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi0S1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi0S1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1SW1024_100_8b_write_out_info.json",
]
process_files_llama2_70b_9_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1SW1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi2S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi2S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi2SW1024_100_8b_write_out_info.json"
]
process_files_llama2_70b_10_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1024_100_8b_write_out_info.json",
]
process_files_llama2_70b_10_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1SW1536_100_8b_write_out_info.json"
]
process_files_llama2_70b_11_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsClean_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC128_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC256_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC32_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC512_52_8b_write_out_info.json",
]
process_files_llama2_70b_11_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC64_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS16_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS2_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS32_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS4_52_8b_write_out_info.json"
]
process_files_llama2_70b_12_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS8_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSLR_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSL_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimpleAS_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimpleA_52_8b_write_out_info.json",
]
process_files_llama2_70b_12_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimpleS_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimple_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_Wikinews_50_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_Wikinews_51_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_Wikinews_52_8b_write_out_info.json"
]
process_files_llama2_70b_13_a = [
    # "meta-llama-Llama-2-70b-chat-hf/MDS_MultinewsTrunc3584_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust10C_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust10O_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust10R_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust1C_42_8b_write_out_info.json",
]
process_files_llama2_70b_13_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust1O_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust1R_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust5C_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust5O_42_8b_write_out_info.json",
]
process_files_llama2_70b_14_a = [
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent10L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent10L05_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent1L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent1L05_42_8b_write_out_info.json",
]
process_files_llama2_70b_14_b = [
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent3L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent3L05_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent5L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent5L05_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsTrunc3584_52_8b_write_out_info.json",
]
process_files_llama2_70b_15 = [
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot0_20Minuten_1_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot0_20Minuten_4_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot0_20min0_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot0_20minTS250_1_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot0_20minTS250_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot1_20minTS250_1_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot1_20minTS250_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot2_20minTS250_1_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot2_20minTS250_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummFewshot8_20Minuten_2_write_out_info.json",
]
process_files_llama2_70b_16 = [
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20Minuten_21_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20Minuten_22_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20Minuten_30_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20Minuten_31_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20min0_21_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20min0_22_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20min0_30_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20min0_31_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20min0_32_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtM1_20min0_33_8b_write_out_info.json",
]
process_files_llama2_70b_17 = [
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p22E_35_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p22E_37_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p22S_35_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p22S_37_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p31E_35_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p31E_37_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p31S_35_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p33E_35_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p33E_37_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p33S_35_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummLtm2_20minLtm2p33S_37_8b_write_out_info.json",
]
process_files_llama2_70b_18 = [
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_11_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_13_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_15_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_17_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_19_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_22_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_23_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_40_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_41_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_43_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_44_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_45_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_46_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_47_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_48_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_49_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_4_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_5_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_7_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummSample_20Minuten_9_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20Minuten_25_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20Minuten_3_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20min0_1_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20min0_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20min0_3_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20min0_4_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/SummarizationTask_20min0_5_8b_write_out_info.json",
]

"""
Llama-2-7b Notes: 
Document 990, "Der zweite Silvester in Folge, der unter dem Einfluss der Corona-Pand..." is too long
"""
process_queue = [
    # ("SEAHORSE_falcon7b_1", process_files_falcon7b_1),
    # ("SEAHORSE_falcon7b_2", process_files_falcon7b_2),
    # ("SEAHORSE_bloomz7b_1", process_files_bloomz7b_1),
    # ("SEAHORSE_bloomz7b_2", process_files_bloomz7b_2),
    # ("SEAHORSE_palm2", process_files_palm2),
    # ("SEAHORSE_gpt4", process_files_gpt4),
    # ("SEAHORSE_platypus2", process_files_platypus2),
    # ("SEAHORSE_orcallama2", process_files_orcallama2),
    # ("SEAHORSE_leolm7b", process_files_leolm7b),
    # ("SEAHORSE_leolm13b", process_files_leolm13b),
    # ("SEAHORSE_llama2_7b_1", process_files_llama2_7b_1),
    # ("SEAHORSE_llama2_7b_2", process_files_llama2_7b_2),
    # ("SEAHORSE_llama2_7b_3", process_files_llama2_7b_3),
    # ("SEAHORSE_llama2_13b_1", process_files_llama2_13b_1),
    # ("SEAHORSE_llama2_13b_2", process_files_llama2_13b_2),
    # ("SEAHORSE_llama2_70b_1", process_files_llama2_70b_1),
    ("SEAHORSE_llama2_70b_2_a", process_files_llama2_70b_2_a),
    ("SEAHORSE_llama2_70b_2_b", process_files_llama2_70b_2_b),
    ("SEAHORSE_llama2_70b_3", process_files_llama2_70b_3),
    # ("SEAHORSE_llama2_70b_4", process_files_llama2_70b_4),
    # ("SEAHORSE_llama2_70b_5_a", process_files_llama2_70b_5_a),
    # ("SEAHORSE_llama2_70b_5_b", process_files_llama2_70b_5_b),
    # ("SEAHORSE_llama2_70b_6_a", process_files_llama2_70b_6_a),
    # ("SEAHORSE_llama2_70b_6_b", process_files_llama2_70b_6_b),
    # ("SEAHORSE_llama2_70b_7_a", process_files_llama2_70b_7_a),
    # ("SEAHORSE_llama2_70b_7_b", process_files_llama2_70b_7_b),
    # ("SEAHORSE_llama2_70b_8_a", process_files_llama2_70b_8_a),
    # ("SEAHORSE_llama2_70b_8_b", process_files_llama2_70b_8_b),
    # ("SEAHORSE_llama2_70b_9_a", process_files_llama2_70b_9_a),
    # ("SEAHORSE_llama2_70b_9_b", process_files_llama2_70b_9_b),
    # ("SEAHORSE_llama2_70b_10_a", process_files_llama2_70b_10_a),
    # ("SEAHORSE_llama2_70b_10_b", process_files_llama2_70b_10_b),
    # ("SEAHORSE_llama2_70b_11_a", process_files_llama2_70b_11_a),
    # ("SEAHORSE_llama2_70b_11_b", process_files_llama2_70b_11_b),
    # ("SEAHORSE_llama2_70b_12_a", process_files_llama2_70b_12_a),
    # ("SEAHORSE_llama2_70b_12_b", process_files_llama2_70b_12_b),
    # ("SEAHORSE_llama2_70b_13_a", process_files_llama2_70b_13_a),
    # ("SEAHORSE_llama2_70b_13_b", process_files_llama2_70b_13_b),
    # ("SEAHORSE_llama2_70b_14_a", process_files_llama2_70b_14_a),
    # ("SEAHORSE_llama2_70b_14_b", process_files_llama2_70b_14_b),
    # ("SEAHORSE_llama2_70b_15", process_files_llama2_70b_15),
    # ("SEAHORSE_llama2_70b_16", process_files_llama2_70b_16),
    # ("SEAHORSE_llama2_70b_17", process_files_llama2_70b_17),
    # ("SEAHORSE_llama2_70b_18", process_files_llama2_70b_18),
]

worker_lang = "de"
skip_processing_base_datasets = True
"""
Old filenames:
- llama-2-7b-chat-hf.json
"""

size_threshold = 4000  # ... leaving some space just in case


def sub_identifier(cmp_gt_summary=False, cmp_article_list=False, cmp_article_list_idx=0):
    prefix = "gt_summary" if cmp_gt_summary else "pred_summary"

    if not cmp_article_list:
        return f"{prefix},full_article"
    else:
        return f"{prefix},article_list_{cmp_article_list_idx}"


def upload_dataset_to_hf(dataset_name, dataset_dict):
    # load the hf-token from the environment variables
    hf_token = os.getenv("HF_TOKEN")
    hub_save_name = f"roysc/{dataset_name}"
    dataset_dict.push_to_hub(hub_save_name, private=True, token=hf_token)


def somajo_sentence_splitting(text: str, n_sent_per_chunk=1) -> List[str]:
    # split the text into sentences
    paragraphs = text.split("\n")
    # sentences = list(somajo_tokenizer.tokenize_text([text]))
    sentences = list(somajo_tokenizer.tokenize_text(paragraphs))
    # make chunks of 1 sentence
    chunk_tokens = [list(itertools.chain.from_iterable(el)) for el in
                    [sentences[i:i + n_sent_per_chunk] for i in range(0, len(sentences), n_sent_per_chunk)]]
    sentence_str_list = [" ".join([el.text for el in x]) for x in chunk_tokens]

    # # if it contains a sentence longer than 512 characters -> edge case
    # if any([len(x) > 1024 for x in sentence_str_list]):
    #     # try to reapply the tokenization, but beforehand splitting the text by newline characters into "paragraphs"
    #     paragraphs = text.split("\n")
    #     sentences = list(somajo_tokenizer.tokenize_text(paragraphs))
    #     chunk_tokens = [list(itertools.chain.from_iterable(el)) for el in [sentences[i:i + n_sent_per_chunk] for i in range(0, len(sentences), n_sent_per_chunk)]]
    #     sentence_str_list = [" ".join([el.text for el in x]) for x in chunk_tokens]

    # filter to only keep sentences with at least 2 characters
    short_sentences = [x for x in sentence_str_list if len(x.strip()) <= 2]
    if len(short_sentences) > 0:
        print(f"\nWarning: short sentences found! Short sentences: {short_sentences}\n")
    sentence_str_list = [x for x in sentence_str_list if len(x.strip()) > 2]
    # Make sure every sentence ends in any kind of punctuation
    sentence_str_list = [x.strip() + "." if x.strip()[-1] not in [".", "!", "?", ":", ";", "\""] else x.strip() for x in
                         sentence_str_list]

    return sentence_str_list


def truncate_article(article, max_article_length, sentence_splitter):
    sentences = sentence_splitter(article)
    sentence_sizes = [len(report.text_to_llama_bpe(sentence)) for sentence in sentences]
    cumsum = np.cumsum(sentence_sizes)
    trunc_idx = np.argmax(cumsum > max_article_length * 1.5)
    sentences = sentences[:min(len(sentences), trunc_idx)]
    # remove sentences 1-by-1 until the total size is smaller than the truncation size
    while len(report.text_to_llama_bpe(" ".join(sentences))) > max_article_length:
        sentences.pop()

    article = " ".join(sentences)
    return article


def prepare_files(src_folder, files_list, dst_file_name, dst_folder):
    print(f"Preparing files for {dst_file_name}")
    sentence_splitter_by_lang = {
        "de": lambda x: somajo_sentence_splitting(x, 1),
        "en": lambda x: sent_tokenize(x, language="english"),
    }
    sentence_splitter = sentence_splitter_by_lang['de']

    # load the base datasets
    base_datasets = {}
    for dataset_name, dataset_path in base_dataset_paths.items():
        with open(dataset_path, 'r') as f:
            base_datasets[dataset_name] = pd.read_json(f, orient="records", lines=True)

    if not skip_processing_base_datasets:
        base_dataset_entries = []
        for dataset_name, dataset in base_datasets.items():
            for index, row in dataset.iterrows():
                article = row['article']
                summary = row['summary']

                article_len = len(report.text_to_llama_bpe(article))
                summary_len = len(report.text_to_llama_bpe(summary))
                full_len = seahorse_prompt_empty_len + article_len + summary_len
                if full_len > size_threshold:
                    max_article_length = size_threshold - (seahorse_prompt_empty_len + summary_len)
                    article = truncate_article(article, max_article_length, sentence_splitter)

                base_dataset_entries.append({
                    "article": article,
                    "summary": summary,
                    "gt_summary": summary,
                    "worker_lang": worker_lang,
                    "id": index,
                    "experiment_id": dataset_name,
                    "sub_id": sub_identifier(cmp_gt_summary=False, cmp_article_list=False)
                })
                # if 'article_list' in row:
                #     for cmp_article_list_idx in range(len(row['article_list'])):
                #         base_dataset_entries.append({
                #             "article": row['article_list'][cmp_article_list_idx],
                #             "summary": row['summary'],
                #             "gt_summary": row['summary'],
                #             "id": index,
                #             "experiment_id": dataset_name,
                #             "sub_id": sub_identifier(cmp_gt_summary=False, cmp_article_list=True,
                #                                      cmp_article_list_idx=cmp_article_list_idx)
                #         })

        base_dataset_entries = pd.DataFrame(base_dataset_entries)
        base_dataset_entries.to_json(os.path.join(dst_folder, f"{base_datasets_dst_file_name}.json"), orient="records",
                                     lines=True)

        dataset_dict = DatasetDict({
            "test": Dataset.from_pandas(base_dataset_entries),
        })
        upload_dataset_to_hf(base_datasets_dst_file_name, dataset_dict)

    # Load all files from the files_list, append all to a single dataframe
    all_data_entries = []
    for idx, file in enumerate(files_list):
        print(f"Processing file {idx + 1}/{len(files_list)}")
        with open(os.path.join(src_folder, file), 'r') as f:
            # data = pd.read_json(f, orient="records", lines=True)
            json_entries = json.load(f)

            just_filename = os.path.basename(file)
            _, _, dataset_id, _, _, _ = report.extract_dataset_and_task_name(just_filename)
            if dataset_id in report.datasetNameMap:
                dataset_name = report.datasetNameMap[dataset_id]
            else:
                raise ValueError(f"Unknown dataset id: {dataset_id}")

            for entry in json_entries:

                gt_summary = entry['truth']
                gt_entry = base_datasets[dataset_name][base_datasets[dataset_name]['summary'].apply(
                    lambda x: x == gt_summary or x in gt_summary or gt_summary in x)]
                if len(gt_entry) != 1:
                    gt_summary_stripped = gt_summary.strip()
                    candidate_rows = base_datasets[dataset_name][
                        base_datasets[dataset_name]["summary"].apply(lambda x: gt_summary_stripped[:100] in x)]
                    if len(candidate_rows) == 1:
                        gt_entry = candidate_rows
                    else:
                        raise ValueError(f"Could not find the ground truth entry for the summary: {gt_summary}")
                article = gt_entry['article'].values[0]
                article_list = [article]
                if 'article_list' in gt_entry:
                    # article_list = entry['article_list']
                    article_list = gt_entry['article_list'].values[0]
                    if isinstance(article_list, str):
                        used_apostrophe = article_list[1]
                        start_str = f"[{used_apostrophe}"
                        end_str = f"{used_apostrophe}]"
                        split_article_list = article_list.split(f"{used_apostrophe}, {used_apostrophe}")
                        split_article_list[0] = split_article_list[0].replace(start_str, "")
                        split_article_list[-1] = split_article_list[-1].replace(end_str, "")
                        article_list = split_article_list

                pred_summary = entry['logit_0']
                orig_doc_id = entry['doc_id']

                # if article is too long, cut it
                article_len = len(report.text_to_llama_bpe(article))
                summary_len = len(report.text_to_llama_bpe(pred_summary))
                full_len = seahorse_prompt_empty_len + article_len + summary_len
                if full_len > size_threshold:
                    max_article_length = size_threshold - (seahorse_prompt_empty_len + summary_len)

                    article = truncate_article(article, max_article_length, sentence_splitter)

                    # full_len_now = seahorse_prompt_empty_len + len(report.text_to_llama_bpe(article)) + summary_len
                    # print(
                    #     f"Cut article, total size before: {full_len}, length now:: {full_len_now}\n(original doc_id: {orig_doc_id}, filename: {file},\ngt-summary: {gt_summary[:70]})")

                all_data_entries.append({
                    "article": article,
                    "summary": pred_summary,
                    "gt_summary": gt_summary,
                    "worker_lang": worker_lang,
                    "id": orig_doc_id,
                    "experiment_id": file,
                    "sub_id": sub_identifier(cmp_gt_summary=False, cmp_article_list=False)
                })

                if len(article_list) > 1:
                    # only process articles that are not empty
                    article_list = [x for x in article_list if len(x.strip()) > 2]

                    for cmp_article_list_idx in range(len(article_list)):
                        article = article_list[cmp_article_list_idx]

                        article_len = len(report.text_to_llama_bpe(article))
                        summary_len = len(report.text_to_llama_bpe(pred_summary))
                        full_len = seahorse_prompt_empty_len + article_len + summary_len
                        if full_len > size_threshold:
                            max_article_length = size_threshold - (seahorse_prompt_empty_len + summary_len)

                            article = truncate_article(article, max_article_length, sentence_splitter)

                        all_data_entries.append({
                            "article": article,
                            "summary": pred_summary,
                            "gt_summary": gt_summary,
                            "worker_lang": worker_lang,
                            "id": orig_doc_id,
                            "experiment_id": file,
                            "sub_id": sub_identifier(cmp_gt_summary=False, cmp_article_list=True,
                                                     cmp_article_list_idx=cmp_article_list_idx)
                        })

    all_data = pd.DataFrame(all_data_entries)
    all_data.to_json(os.path.join(dst_folder, f"{dst_file_name}.json"), orient="records", lines=True)

    dataset_dict = DatasetDict({
        "test": Dataset.from_pandas(all_data),
    })
    upload_dataset_to_hf(dst_file_name, dataset_dict)

    return


# def extend_basic_metrics(src_file, dst_file):
#     # Load the src_file into a dataframe
#     with open(src_file, 'r') as f:
#         src_data = json.load(f)
#     data = pd.DataFrame(src_data)
#
#     """
#     Prepare all the data -> copy article (needed for SEAHORSE Metrics)
#     """
#     out_data = []
#     for index, row in enumerate(src_data):
#         # prepare all the text for calculating metrics
#         gt_summary = row['truth']
#         article = \
#         base_dataset[base_dataset['summary'].apply(lambda x: x == gt_summary or x in gt_summary or gt_summary in x)][
#             'article'].values[0]
#         row['article'] = article
#         out_data.append(row)
#
#     """
#     Calculate more SummEval Metrics
#     """
#     for index, row in enumerate(out_data):
#         # prepare all the text for calculating metrics
#         gt_summary = row['truth']
#         article = row['article']
#         pred_summary = row['logit_0']
#
#         # calculate SummEval metrics
#         # TODO ...
#         # summ_eval_metrics = SummEval
#
#         # save the metrics
#
#     # delete the "article" column
#
#     return

def strip_MLSum_summary(summary):
    # Removes the following characters as well as duplicate whitespaces
    # `` '' , . : ( ) ? !

    chars_to_remove = ['``', '\'\'', ',', '.', ':', '(', ')', '?', '!']
    for char_to_remove in chars_to_remove:
        summary = summary.replace(char_to_remove, '')

    summary = summary.strip()

    return summary


def seahorse_gem_identifier(gem_id):
    fields = gem_id.split("-")
    assert len(fields) == 3

    return fields[0], fields[1], fields[2]


def prepare_seahorse_testset(dst_folder):
    SEAHORSE_test_location = "./resources/seahorse_data/test.tsv"
    SEAHORSE_MLSum_location_src = "./resources/seahorse_data/MLSum/test.txt.src"
    SEAHORSE_MLSum_location_tgt = "./resources/seahorse_data/MLSum/test.txt.tgt"
    SEAHORSE_Wikilingua_location = "./resources/seahorse_data/Wikilingua/german.pkl"

    # Load all the data
    SEAHORSE_test = pd.read_csv(SEAHORSE_test_location, sep='\t', header=0)

    MLSum_entries = []
    with open(SEAHORSE_MLSum_location_src) as srcF, open(SEAHORSE_MLSum_location_tgt) as tgtF:
        for src, tgt in zip(srcF, tgtF):
            MLSum_entries.append({
                "article": src,
                "summary": tgt,
                "summary_stripped": strip_MLSum_summary(tgt),
            })

    # NOTE: according to seahorse github, need to be careful with Wikilingua dataset w.r.t. gem_ids?

    Wikilingua = pd.read_pickle(SEAHORSE_Wikilingua_location)

    Wikilingua_hf_dataset = datasets.load_dataset("gem", "wiki_lingua_german_de", split="test")
    MLSum_hf_dataset = datasets.load_dataset("gem", "mlsum_de", split="test")
    MLSum_hf_dataset2 = datasets.load_dataset("GEM/mlsum", "de", split="test")

    Wikilingua_hf_df = pd.DataFrame(Wikilingua_hf_dataset)
    MLSum_hf_df = pd.DataFrame(MLSum_hf_dataset)
    MLSum_hf_df2 = pd.DataFrame(MLSum_hf_dataset2)

    lang = 'german_de'
    orig_split = 'test'
    ds, info = tfds.load(f'huggingface:gem/wiki_lingua_{lang}', split=orig_split, with_info=True)
    Wikilingua_tf_df = tfds.as_dataframe(ds, info)

    # Filter the Seahorse data
    # Filters: worker_lang="de", question1="Yes"
    SEAHORSE_test = SEAHORSE_test[SEAHORSE_test['worker_lang'] == "de"]
    SEAHORSE_test = SEAHORSE_test[SEAHORSE_test['question1'] == "Yes"]

    # Extend the test data with the correct articles, at the same time, form the dataset for seahorse prediction
    seahorse_gt_dataset_name = "SEAHORSE_TEST"
    sentence_splitter_by_lang = {
        "de": lambda x: somajo_sentence_splitting(x, 1),
        "en": lambda x: sent_tokenize(x, language="english"),
    }
    sentence_splitter = sentence_splitter_by_lang['de']
    base_dataset_entries = []
    for index, row in SEAHORSE_test.iterrows():
        gem_id = row['gem_id']
        gem_dataset, gem_split, gem_idx = seahorse_gem_identifier(gem_id)
        article = ""
        if gem_dataset == "mlsum_de":
            candidate_rows = MLSum_hf_df[MLSum_hf_df['gem_id'] == gem_id]
            if len(candidate_rows) == 1:
                article = candidate_rows['text'].values[0]
            else:
                # try the other MLSum dataset
                candidate_rows = MLSum_hf_df2[MLSum_hf_df2['gem_id'] == gem_id]
                if len(candidate_rows) == 1:
                    article = candidate_rows['text'].values[0]
                else:
                    raise ValueError(f"Could not find the article for the gem_id: {gem_id}")
        elif gem_dataset == "wiki_lingua_german_de":
            candidate_rows = Wikilingua_tf_df[Wikilingua_tf_df['gem_id'] == gem_id.encode('UTF-8')]
            if len(candidate_rows) == 1:
                article = candidate_rows['source'].values[0].decode('UTF-8')
            else:
                raise ValueError(f"Could not find the article for the gem_id: {gem_id}")
        else:
            raise ValueError(f"Unknown gem dataset: {gem_dataset}")

        q1 = row['question1']
        q2 = row['question2']
        q3 = row['question3']
        q4 = row['question4']
        q5 = row['question5']
        q6 = row['question6']
        summary = row['summary']
        model = row['model']

        article_len = len(report.text_to_llama_bpe(article))
        summary_len = len(report.text_to_llama_bpe(summary))
        full_len = seahorse_prompt_empty_len + article_len + summary_len
        if full_len > size_threshold:
            max_article_length = size_threshold - (seahorse_prompt_empty_len + summary_len)
            article = truncate_article(article, max_article_length, sentence_splitter)

        base_dataset_entries.append({
            "gem_id": gem_id,
            "model": model,
            "article": article,
            "summary": summary,
            "worker_lang": "de",
            "question1": q1,
            "question2": q2,
            "question3": q3,
            "question4": q4,
            "question5": q5,
            "question6": q6
        })

    base_dataset_entries = pd.DataFrame(base_dataset_entries)
    base_dataset_entries.to_json(os.path.join(dst_folder, f"{seahorse_gt_dataset_name}.json"), orient="records",
                                 lines=True)

    dataset_dict = DatasetDict({
        "test": Dataset.from_pandas(base_dataset_entries),
    })
    upload_dataset_to_hf(seahorse_gt_dataset_name, dataset_dict)

    # OLD NOTES: Make function that reads in output from seahorse and extends the results
    #   Load the seahorse test set (load the tsv file)
    #   Filter by worker_lang="de"
    #   Filter by qestion1="Yes"
    #   Load the original articles from the datasets: Wikilingua and MLSum
    #   - Wikilingua:
    #       import tensorflow_datasets as tfds
    #       lang = 'english_en'
    #       orig_split = 'validation'
    #       ds, info = tfds.load(f'huggingface:gem/wiki_lingua_{lang}', split=orig_split, with_info=True)
    #       hfdf = tfds.as_dataframe(ds,info)
    #   - MLSum -> retreive through GEM on huggingface, or directly through the original test file (with lines as indices?)
    #       import datasets
    #       data = datasets.load_dataset('GEM/mlsum')
    #   ALTERNATIVE: Load the datasets, and try to match the reference summaries to the original summaries to get the appropriate articles
    #     MLSum -> make sure to remove `` '' , . : ( ) ? ! from both summaries, because the MLSum dataset has weird whitespaces next to them
    #

    return


def postprocess_seahorse_results():
    # NOTE: The code assumes that all seahorse results are performed for all the keys, so each file must be present
    # for each key below.
    seahorse_paths = {
        "attribution": "../results/mtc-NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged",
        "conciceness": "../results/mtc-NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged",
        "main-ideas": "../results/mtc-NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged",
    }
    exclude_filenames = [
        "seahorse_testq1_100_write_out_info.json",
        "seahorse_testq2_100_write_out_info.json",
        "seahorse_testq3_100_write_out_info.json",
        "seahorse_testq4_100_write_out_info.json",
        "seahorse_testq5_100_write_out_info.json",
        "seahorse_testq6_100_write_out_info.json",
    ]
    DST_FOLDER = "./results_seahorse"  # folder to put all results in from experiments

    if len(seahorse_paths) == 0:
        raise ValueError("No seahorse paths specified!")

    # read in all the filenames, and process them grouped, such that only the content of files with the same name is processed at once
    filenames = {}
    for key, path in seahorse_paths.items():
        filenames[key] = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        filenames[key] = [f for f in filenames[key] if f.endswith(".json")]
        filenames[key] = sorted(filenames[key])

        # remove the files that should be excluded
        filenames[key] = [f for f in filenames[key] if f not in exclude_filenames]


    # process the files
    seahorse_keys = list(seahorse_paths.keys())
    generic_seahorse_key = list(seahorse_keys)[0]
    for filename in filenames[generic_seahorse_key]:
        print(f"Processing file {filename}")

        # load the data into a map, only load the relevant fields, and already group the data by the experiment_id,
        # because all the data with the same experiment_id will be put into the same file in the end
        seahorse_data = {}
        experiment_ids = []
        for key, path in seahorse_paths.items():
            if key not in seahorse_data:
                seahorse_data[key] = {}
            with open(os.path.join(path, filename), 'r') as f:
                result = json.load(f)
                entries = {}
                for entry in result:
                    experiment_id = entry["experiment_id"]
                    if experiment_id not in experiment_ids:
                        experiment_ids.append(experiment_id)
                    if experiment_id not in entries:
                        entries[experiment_id] = []
                    truth_value = True if entry["prediction"] == "1" else False
                    entries[experiment_id].append({
                        "doc_id": entry["id"],
                        "score": entry["true_prediction_probability"],
                        "truth_value": truth_value,
                        "experiment_id": experiment_id,
                        "sub_id": entry["sub_id"],
                        "article": entry["article"],
                        "summary": entry["summary"],
                        "gt_summary": entry["gt_summary"],
                    })
                # need to aggregate entries that have the same doc_id, these are MDS predictions and need to be further processed
                agg_entries = {}
                for experiment_id, experiment_entries in entries.items():
                    processed_doc_ids = []
                    for entry in experiment_entries:
                        doc_id = entry["doc_id"]
                        if doc_id in processed_doc_ids:
                            continue

                        # find all entries with the same doc_id
                        base_entries = [e for e in experiment_entries if
                                        e["doc_id"] == doc_id and ",full_article" in e["sub_id"]]
                        if len(base_entries) != 1:
                            raise ValueError(f"Expected exactly 1 base entry, but found {len(base_entries)}")
                        base_entry = base_entries[0]
                        # get the mds_entries and sort them by sub_id (since it contains the index of the article)
                        mds_entries = [e for e in experiment_entries if
                                       e["doc_id"] == doc_id and ",full_article" not in e["sub_id"]]
                        mds_entries = sorted(mds_entries, key=lambda x: int(x["sub_id"].split("_")[-1]))

                        # aggregate the entries (if necessary)
                        if len(mds_entries) > 0:
                            if experiment_id not in agg_entries:
                                agg_entries[experiment_id] = []

                            # calculate basic aggregations and add them to the base_entry
                            mds_scores = [e["score"] for e in mds_entries]
                            mds_truth_values = [e["truth_value"] for e in mds_entries]
                            avg_score = np.mean(mds_scores)
                            avg_truth_value = np.mean([1.0 if x else 0.0 for x in mds_truth_values])

                            base_entry["mds_scores"] = mds_scores
                            base_entry["mds_truth_values"] = mds_truth_values
                            base_entry["mds_avg_score"] = avg_score
                            base_entry["mds_avg_truth_value"] = avg_truth_value

                            agg_entries[experiment_id].append(base_entry)
                        else:
                            if experiment_id not in agg_entries:
                                agg_entries[experiment_id] = []
                            agg_entries[experiment_id].append(base_entry)

                        processed_doc_ids.append(doc_id)

                # sort the agg_entries by doc_id and save them in the seahorse_data
                for experiment_id, experiment_entries in agg_entries.items():
                    experiment_entries = sorted(experiment_entries, key=lambda x: int(x["doc_id"]))
                    seahorse_data[key][experiment_id] = experiment_entries
                seahorse_data[key] = agg_entries
        # Now, all the data is loaded into seahorse_data, and grouped by experiment_id
        # Next: put all the data of the same experiment_id into a single file, and save it for the report code to use
        for experiment_id in experiment_ids:
            # create the output folder if it does not exist yet
            if experiment_id.endswith(".json"):
                save_folder = os.path.join(DST_FOLDER, os.path.dirname(experiment_id))
                save_filename = os.path.basename(experiment_id)
            else:
                save_folder = os.path.join(DST_FOLDER, experiment_id)
                save_filename = f"{experiment_id}.json"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Aggregate the data that has the same doc_id but different seahorse-key into a single file
            all_doc_ids = [entry["doc_id"] for key in seahorse_keys for entry in seahorse_data[key][experiment_id]]
            all_doc_ids = [int(x) for x in all_doc_ids]
            all_doc_ids = list(set(all_doc_ids))
            all_doc_ids.sort()

            experiment_entries = []
            for doc_id in all_doc_ids:
                doc_id_entries = {}
                # find the relevant entries (i know that its not very efficient)
                for key in seahorse_keys:
                    for entry in seahorse_data[key][experiment_id]:
                        if entry["doc_id"] == doc_id or entry["doc_id"] == str(doc_id):
                            doc_id_entries[key] = entry
                # aggregate the entries
                agg_entry = {
                    "doc_id": doc_id,
                    "experiment_id": doc_id_entries[generic_seahorse_key]["experiment_id"],
                    "article": doc_id_entries[generic_seahorse_key]["article"],
                    "summary": doc_id_entries[generic_seahorse_key]["summary"],
                    "gt_summary": doc_id_entries[generic_seahorse_key]["gt_summary"],
                }
                for key in seahorse_keys:
                    agg_entry[f"{key}_score"] = doc_id_entries[key]["score"]
                    agg_entry[f"{key}_truth_value"] = doc_id_entries[key]["truth_value"]
                    if "mds_scores" in doc_id_entries[key]:
                        agg_entry[f"{key}_mds_scores"] = doc_id_entries[key]["mds_scores"]
                        agg_entry[f"{key}_mds_truth_values"] = doc_id_entries[key]["mds_truth_values"]
                        agg_entry[f"{key}_mds_avg_score"] = doc_id_entries[key]["mds_avg_score"]
                        agg_entry[f"{key}_mds_avg_truth_value"] = doc_id_entries[key]["mds_avg_truth_value"]

                experiment_entries.append(agg_entry)

            # save the data to the file
            with open(os.path.join(save_folder, save_filename), 'w') as f:
                # write the data to the file
                json.dump(experiment_entries, f, indent=4)

    return


if __name__ == "__main__":
    # # if argument --google pass the instruction to use the other prompt
    # beluga_prompt = True
    # if "--google" in sys.argv:
    #     beluga_prompt = False

    if "--postprocess" in sys.argv:
        postprocess_seahorse_results()
    else:
        for dst_file_name, process_files in process_queue:
            prepare_files(SRC_FOLDER, process_files, dst_file_name, INT_FOLDER)
        # extend_basic_metrics(os.path.join(INT_FOLDER, "all_data.json"), os.path.join(DST_FOLDER, "all_data.json"))

        # prepare the test set
        prepare_seahorse_testset(INT_FOLDER)
