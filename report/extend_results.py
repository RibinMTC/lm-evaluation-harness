import collections
import itertools
import numpy as np
import pandas as pd
import random
import os
from datasets import DatasetDict, Dataset
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

seahorse_prompt_template = """### System:
You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can.

### User: Given an article and its corresponding summary in {lang}, reply with True if all information in the summary originates from the article and reply False otherwise. 
Article: {article}
Summary: {summary}

### Assistant:"
"""
seahorse_prompt_empty = seahorse_prompt_template.format(lang="German", article="", summary="")
seahorse_prompt_empty_len = len(report.text_to_llama_bpe(seahorse_prompt_empty))

base_dataset_paths = {
    "20Minuten": "./resources/Datasets/20Minuten/test.json",
    "Wikinews": "./resources/Datasets/Wikinews/test.json",
    "Klexikon": "./resources/Datasets/Klexikon/test.json",
}

base_datasets_dst_file_name = "SEAHORSE_base_datasets"
process_files_llama_2_7b = [
    "meta-llama-Llama-2-7b-chat-hf/SummEmpty_20minSmol_1_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_15_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummEmpty_20minSmol_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_16_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummEmpty_20minSmol_4_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_17_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummFewshot_20Minuten_1_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_18_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummFewshot_20Minuten_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_19_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummFewshot_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummLtMDe_20Minuten_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummLtM_20Minuten_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummNonEmpty_20minSmol_1_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T05_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummNonEmpty_20minSmol_4_8b_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_4_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_1_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummSample_20Minuten_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_20_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_10_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_6_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_2_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_11_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_7_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_3_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_12_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_8_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T10_5_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_13_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_9_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_14_write_out_info.json",
    "meta-llama-Llama-2-7b-chat-hf/SummarizationTask_20Minuten_T01_1_write_out_info.json",
]
process_files_MDS_Basic = [
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2OP41_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2O_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2SP41_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplitS2S_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplit_2_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS2S_WikinewsSplit_41_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsClean_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC128_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC256_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC32_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC512_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSC64_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS16_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS2_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS32_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS4_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSCS8_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSLR_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSL_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimpleAS_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimpleA_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimpleS_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_WikinewsSimple_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_Wikinews_50_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_Wikinews_51_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSSumm_Wikinews_52_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust10C_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust10O_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust10R_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust1C_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust1O_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust1R_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust5C_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsClust5O_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent10L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent10L05_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent1L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent1L05_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent3L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent3L05_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent5L00_42_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDS_WikinewsSent5L05_42_8b_write_out_info.json",
]
process_files_MDS_FCO = [
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCh1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCh1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0N1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0N1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl0N2048_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1N21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1N21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1N22048_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl1SW1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2S22048_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiCl2SW1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi0S1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi0S1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi1SW1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi2S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi2S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiDi2SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiLe1S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1S21024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1S21536_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1SW1024_100_8b_write_out_info.json",
    "meta-llama-Llama-2-70b-chat-hf/MDSFCO_WikiRa1SW1536_100_8b_write_out_info.json",
]
process_files_MDS_Chain = [
    "meta-llama-Llama-2-70b-chat-hf/MDSChain_WikinewsS2_100_8b_write_out_info.json",
]
process_files_SDS = [
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

worker_lang = "de"
process_files = process_files_SDS
dst_file_name = "SEAHORSE_mds_fco_experiments"
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


def prepare_files(src_folder, files_list, dst_folder):
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
        print(f"Processing file {idx}/{len(files_list)}")
        with open(os.path.join(src_folder, file), 'r') as f:
            # data = pd.read_json(f, orient="records", lines=True)
            json_entries = json.load(f)

            _, _, dataset_id, _, _, _ = report.extract_dataset_and_task_name(file)
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
                if 'article_list' in entry:
                    article_list = entry['article_list']

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
                # if len(article_list) > 1:
                #     for cmp_article_list_idx in range(len(article_list)):
                #         all_data_entries.append({
                #             "article": article_list[cmp_article_list_idx],
                #             "summary": pred_summary,
                #             "gt_summary": gt_summary,
                #             "id": orig_doc_id,
                #             "experiment_id": file,
                #             "sub_id": sub_identifier(cmp_gt_summary=False, cmp_article_list=True,
                #                                      cmp_article_list_idx=cmp_article_list_idx)
                #         })

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


# TODO: Make function that reads in output from seahorse and extends the results


if __name__ == "__main__":
    prepare_files(SRC_FOLDER, process_files, INT_FOLDER)
    # extend_basic_metrics(os.path.join(INT_FOLDER, "all_data.json"), os.path.join(DST_FOLDER, "all_data.json"))
