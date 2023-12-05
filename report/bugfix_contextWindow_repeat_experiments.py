from tqdm import tqdm
import json
import os
import sys
import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset
import pathlib
from typing import List, Dict, Union, Callable, Tuple
from transformers import LlamaTokenizer, LlamaTokenizerFast

llama_tokenizer = LlamaTokenizerFast(vocab_file="llama2-tokenizer.model", tokenizer_file="llama2-tokenizer.json")
text_to_llama_bpe = lambda x: llama_tokenizer.encode(x)

base_folder = "../results"
output_folder = "repeat_results_bugfix"
model_folders = [
    "meta-llama-Llama-2-7b-chat-hf",
    "meta-llama-Llama-2-70b-chat-hf",
]
model_folder_to_id = {
    "meta-llama-Llama-2-7b-chat-hf": "Llama_2_7b",
    "meta-llama-Llama-2-70b-chat-hf": "Llama_2_70b",
}

exclude_experiments = [
    "SummFewshot8_20Minuten_2_write_out_info.json",

    "SummEmpty_20minSmol_1_8b_write_out_info.json",
    "SummEmpty_20minSmol_1_write_out_info.json",
    "SummEmpty_20minSmol_4_8b_write_out_info.json",
    "SummFewshot_20Minuten_1_8b_write_out_info.json",
    "SummFewshot_20Minuten_1_write_out_info.json",
    "SummFewshot_20Minuten_2_write_out_info.json",
    "SummLtMDe_20Minuten_write_out_info.json",
    "SummLtM_20Minuten_write_out_info.json",
    "SummNonEmpty_20minSmol_1_8b_write_out_info.json",
    "SummNonEmpty_20minSmol_4_8b_write_out_info.json",
    "SummSample_20Minuten_2_write_out_info.json",
]

model_folder = model_folders[1]


def upload_dataset_to_hf(dataset_name, dataset_dict):
    # load the hf-token from the environment variables
    hf_token = os.getenv("HF_TOKEN")
    hub_save_name = f"roysc/{dataset_name}"
    dataset_dict.push_to_hub(hub_save_name, private=True, token=hf_token)


def identifier(doc_id, filename):
    return f"{filename}/{doc_id}"


def filename_from_identifier(identifier):
    return identifier.split("/")[0]


def doc_id_from_identifier(identifier):
    return identifier.split("/")[1]


def needs_repetition(prompt: str, response: str):
    size_threshold = 3450

    # simple rule (can also be extended to check if response starts with lowercase character etc.
    if len(text_to_llama_bpe(prompt)) > size_threshold:
        return True


def load_results_to_be_repeated():
    results = []
    base_path = os.path.join(base_folder, model_folder)
    for filename in os.listdir(base_path):
        if filename in exclude_experiments:
            print(f"Skipping {filename}")
            continue
        if not filename.endswith(".json"):
            continue
        print(f"Processing {filename}")
        with open(os.path.join(base_path, filename), "r") as f:
            experiment = json.load(f)

            for entry in experiment:
                prompt = entry["prompt_0"]
                response = entry["logit_0"]
                gt_response = entry["truth"]
                doc_id = entry["doc_id"]

                if needs_repetition(prompt, response):
                    results.append({
                        "article": prompt,
                        "summary": gt_response,
                        "prev_prediction": response,
                        "identifier": identifier(doc_id, filename),
                    })

    print("Done loading results")
    print(f"Founds {len(results)} entries that need repetition")

    # loop over the results and count the number of entries that need repetition per filename
    filename_to_num_entries = {}
    for entry in results:
        filename = filename_from_identifier(entry["identifier"])
        if filename not in filename_to_num_entries:
            filename_to_num_entries[filename] = 0
        filename_to_num_entries[filename] += 1
    longest_filename_length = max([len(filename) for filename in os.listdir(base_path)])
    for filename in filename_to_num_entries:
        print(f"\t{filename.ljust(longest_filename_length)}:\t{filename_to_num_entries[filename]}")

    # save results and upload to huggingface
    output_path = os.path.join(output_folder, f"repeated_results_{model_folder_to_id[model_folder]}.json")
    # make the directory if it does not exist
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    # split into multiple chunks of 500 entries each
    chunk_size = 500
    num_chunks = len(results) // chunk_size + 1
    for i in range(num_chunks):
        chunk = results[i * chunk_size: (i + 1) * chunk_size]
        output_path = os.path.join(output_folder, f"repeated_results_{model_folder_to_id[model_folder]}_{i}.json")
        with open(output_path, "w") as f:
            json.dump(chunk, f)

        chunk_dataset_name = f"repeated_results_{model_folder_to_id[model_folder]}_{i}"
        df_chunk = pd.DataFrame(chunk)
        dataset_dict = DatasetDict({
            "test": Dataset.from_pandas(df_chunk),
        })
        upload_dataset_to_hf(chunk_dataset_name, dataset_dict)


def insert_repeated_experiments():
    raise NotImplementedError("Not implemented yet")


if __name__ == '__main__':
    if "--load" in sys.argv:
        load_results_to_be_repeated()
    elif "--insert" in sys.argv:
        insert_repeated_experiments()
    else:
        raise ValueError("Invalid argument")
