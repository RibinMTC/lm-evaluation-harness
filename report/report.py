import re
import shutil

from tqdm import tqdm
import json
import os
import sys
import math
from scipy.stats import bootstrap
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib
# import matplotlib.patches as mpatches
import gc

tqdm.pandas()

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import jinja2
from typing import List, Dict, Union, Callable, Tuple
import fasttext
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

import nltk
from sentence_transformers import SentenceTransformer, util
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from somajo import SoMaJo
import itertools
import tiktoken
from lm_eval.fragments import Fragments

from transformers import LlamaTokenizer, LlamaTokenizerFast

llama_tokenizer = LlamaTokenizerFast(vocab_file="llama2-tokenizer.model", tokenizer_file="llama2-tokenizer.json")
text_to_llama_bpe = lambda x: llama_tokenizer.encode(x)

DEFAULT_THEME = {
    "style": "darkgrid", "rc": {"figure.figsize": (27, 7), "font.size": 10, },
}

pd.set_option("display.precision", 4)
sns.set_theme(**DEFAULT_THEME)
sns_num_xticks_rotation_threshold = 12
# sns.set(font_scale=1.05)

sns.despine(bottom=True, left=True, offset=5)
# plt.tight_layout()

ROUND_TO_DECIMALS = 4
pd.options.mode.chained_assignment = None  # default='warn'

# To supress: Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
fasttext.FastText.eprint = lambda x: None

""" TODO: Plot minor ticks
src: https://www.reddit.com/r/learnpython/comments/pm948a/cant_find_a_way_to_add_minor_yticks_to_my_plot/

import numpy as np from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
t = np.arange(0.0, 100.0, 0.01) s = np.sin(2 * np.pi * t) * np.exp(-t * 0.01)
fig, ax = plt.subplots() ax.plot(t, s)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2) ax.tick_params(which='major', length=7) ax.tick_params(which='minor', length=4, color='r')
plt.show()
"""


def rename_hf_model(model_name):
    # replace / with -
    model_name = model_name.replace("/", "-")
    return model_name


# Make a function extracting the prompt version and the dataset name from a filename
# Example: SummarizationTask_20Minuten_2_write_out_info.json -> 20Minuten, 2
def extract_dataset_and_task_name(filename):
    # Split the filename by underscores
    split_filename = filename.split("_")
    # Extract the prompt version and the task name
    task_name = split_filename[0]
    prompt_version_idx = 2
    dataset_name = split_filename[1]

    # extract number of shots
    nshot = 0
    if "Fewshot" in task_name:
        nshotRegex = re.compile(r"Fewshot[0-9]+")
        nshotMatch = nshotRegex.search(task_name)
        if nshotMatch:
            nshot = int(nshotMatch.group(0)[len("Fewshot"):])

        # get location of Fewshot in the task name, and extract the number after the Fewshot string
        # nshot = int(task_name[task_name.find("Fewshot") + len("Fewshot"):])
    # TODO

    # find an element of the form 'T[0-9]*' in the split_filename
    # if there is no such element, the temperature is 0.0
    temperature = 0.0
    for element in split_filename:
        if element.startswith("T") and len(element) >= 3:
            temperature = float(element[1:]) / 10
            prompt_version_idx = 3
    # if found temperature -> prompt-version is the element after the temperature
    prompt_version = split_filename[prompt_version_idx]

    # Check if there is a precision flag after the prompt version
    precision = "full"
    if split_filename[prompt_version_idx + 1].endswith("b"):
        precision = "8bit"

    return task_name, nshot, dataset_name, prompt_version, temperature, precision


# Function to load results from JSON files to a dataframe
def load_results(results_path_base, model_name, return_concat_df=True):
    model_name = rename_hf_model(model_name)
    results_path = f"../{results_path_base}/{model_name}"
    results_path_read = f"../results/{model_name}"
    seahorse_path = f"./results_seahorse/{model_name}"
    result_files = [f for f in os.listdir(results_path) if f.endswith(".json")]

    # Load the results from all JSON files and extend their fields as needed
    all_results = []
    model_names = []
    filenames = []
    for file in result_files:
        with open(os.path.join(results_path_read, file), "r") as f:
            result = json.load(f)

            task_name, nshot, dataset, promptVersion, temperature, precision = extract_dataset_and_task_name(file)
            for entry in result:
                entry["Task Name"] = task_name
                entry["N-Shot"] = nshot
                entry["Dataset"] = dataset
                entry["Prompt ID"] = promptVersion
                entry["Model"] = model_name
                entry['Temperature'] = temperature
                entry['Model Precision'] = precision

            # join with seahorse results based on doc_id
            include_seahorse_columns = ["attribution_score", "attribution_truth_value", "conciceness_score",
                                        "conciceness_truth_value", "main-ideas_score", "main-ideas_truth_value"]
            seahorse_mds_base = ["attribution", "conciceness", "main-ideas"]
            seahorse_mds_extensions = ["_mds_scores", "_mds_truth_values", "_mds_avg_score", "_mds_avg_truth_value", ]
            seahorse_list_exts = [f"{base}{extension}" for base in seahorse_mds_base for extension in
                                  ["_mds_scores", "_mds_truth_values"]]
            include_seahorse_mds_columns = [f"{base}{extension}" for base in seahorse_mds_base for extension in
                                            seahorse_mds_extensions]
            if os.path.exists(os.path.join(seahorse_path, file)):
                with open(os.path.join(seahorse_path, file), "r") as f:
                    seahorse_result = json.load(f)
                    seahorse_result_df = pd.DataFrame(seahorse_result)
                    for entry in result:
                        doc_id = entry["doc_id"]
                        seahorse_row = seahorse_result_df[seahorse_result_df["doc_id"] == doc_id]
                        if len(seahorse_row) == 0:
                            print(
                                f"Warning: No Seahorse row found for model/doc_id {model_name}/{doc_id}  --  ({results_path} / {file})")
                            continue
                        seahorse_row = seahorse_row.iloc[0]
                        for column in include_seahorse_columns:
                            entry[column] = seahorse_row[column]
                        for column in include_seahorse_mds_columns:
                            if column in seahorse_row:
                                entry[column] = seahorse_row[column]
            else:
                print(f"Warning: No Seahorse file found for file {model_name}/{file}, initializing to 0")
                for entry in result:
                    for column in include_seahorse_columns:
                        entry[column] = 0.0
                    for column in include_seahorse_mds_columns:
                        if column in seahorse_list_exts:
                            entry[column] = [0.0]
                        else:
                            entry[column] = 0.0

            all_results.append(result)
            model_names.append(model_name)
            filenames.append(file)

    if return_concat_df:
        # Concatenate the results from all JSON files
        df = pd.concat([pd.DataFrame(result) for result in all_results])
        return [df], [model_name], ["concat"]
    else:
        all_results_df = [pd.DataFrame(result) for result in all_results]
        return all_results_df, model_names, filenames


# function receiving a list of model names calling load_results for each model and concatenating the results
def load_all_results(results_path, model_names, shortNames, reload_preprocessed_dataset=False,
                     no_cache_reload_all=False):
    dfs = []
    model_names_list = []
    filenames_list = []
    for model_name in model_names:
        df_list, model_names, filenames = load_results(results_path, model_name, return_concat_df=False)
        dfs.extend(df_list)
        model_names_list.extend(model_names)
        filenames_list.extend(filenames)
    df = pd.concat(dfs)

    # prepare shotNames map -> replace / with -
    shortNames = {rename_hf_model(model_name): shortNames[model_name] for model_name in shortNames}

    in__dataset_path = os.path.join("resources", "Datasets")
    out_dataset_path = os.path.join("resources", "Datasets-Prepared")
    pathlib.Path(out_dataset_path).mkdir(parents=True, exist_ok=True)

    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    text_to_bpe = lambda x: tiktoken_encoder.encode(x)
    text_to_tokens = lambda x: list(itertools.chain.from_iterable(list(somajo_tokenizer.tokenize_text([x]))))

    # Load the datasets to calculate the number of input documents and number of input tokens
    datasets = {}
    if reload_preprocessed_dataset:
        # List directories in the folder, load the datasets, use the filename as Split Idenfitier
        for dataset in os.listdir(in__dataset_path):
            # skip if not a directory
            if not os.path.isdir(os.path.join(in__dataset_path, dataset)):
                continue
            dataset_path = os.path.join(in__dataset_path, dataset)
            dataset_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
            dataset_dfs = []
            for file in dataset_files:
                with open(file, "r") as f:
                    dataset_df = pd.read_json(f, orient="records", lines=True)
                    dataset_df['Split'] = os.path.splitext(os.path.basename(file))[0]
                    dataset_dfs.append(dataset_df)
            dataset_df = pd.concat(dataset_dfs)
            datasets[dataset] = dataset_df

        # Calculate the number of input documents and number of input tokens
        # Pre-calculate for all relevant datasets, store in a map, use map to fill df
        dataset_names = df["Dataset"].unique()
        dataset_names = [datasetNameMap[x] if x in datasetNameMap else x for x in dataset_names]
        supported_names = ["20Minuten", "Wikinews", "Klexikon", "Multinews"]
        mds_names = ["Wikinews", "Multinews"]
        supported_name = [True if dataset_name in supported_names else False for dataset_name in dataset_names]
        assert all(supported_name), "All datasets must be supported in load_all_results"

        for dataset_name in datasets:
            dataset_df = datasets[dataset_name]
            isMDS = "article_list" in dataset_df.columns

            # calculate the number of input documents per row (MDS)
            # and the number of input tokens per row (full input size)
            for idx, row in datasets[dataset_name].iterrows():
                if not isMDS:
                    n_input_docs = 1
                else:
                    n_input_docs = len(row["article_list"])

                article = row["article"]
                summary = row["summary"]
                article_bpe = text_to_bpe(article)
                article_tokens = text_to_tokens(article)
                summary_bpe = text_to_bpe(summary)
                summary_tokens = text_to_tokens(summary)
                datasets[dataset_name].at[idx, "num_input_docs"] = n_input_docs
                datasets[dataset_name].at[idx, "num_article_bpe"] = len(article_bpe)
                datasets[dataset_name].at[idx, "num_article_tokens"] = len(article_tokens)
                datasets[dataset_name].at[idx, "num_summary_bpe"] = len(summary_bpe)
                datasets[dataset_name].at[idx, "num_summary_tokens"] = len(summary_tokens)

        # save the prepared datasets
        for dataset_name in datasets:
            dataset_df = datasets[dataset_name]
            dataset_df.to_csv(os.path.join(out_dataset_path, f"{dataset_name}.csv"), index=False)
    else:
        # load the prepared datasets
        dataset_files = os.listdir(out_dataset_path)
        dataset_names = [os.path.splitext(dataset_file)[0] for dataset_file in dataset_files]
        for dataset_name in dataset_names:
            dataset_df = pd.read_csv(os.path.join(out_dataset_path, f"{dataset_name}.csv"))

            # if "index" column in dataset_df -> rename to "id"
            if "index" in dataset_df.columns:
                dataset_df.rename(columns={"index": "id"}, inplace=True)

            datasets[dataset_name] = dataset_df

    # make a length-statistic plot for all datasets
    dataset_insights_path = os.path.join("reports", "_dataset_insights")
    pathlib.Path(dataset_insights_path).mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        metric_plt_theme = {
            "style": "darkgrid", "rc": {"figure.figsize": (15, 18), "font.size": 24, },
        }
        sns.set_theme(**metric_plt_theme)
        # Box-Plot
        fig, axes = plt.subplots(1, 2)
        box_plot_1 = sns.boxplot(data=datasets[dataset_name], y="num_article_bpe", flierprops={"marker": "x"},
                                 notch=True, bootstrap=1000, ax=axes[0])
        box_plot_1.set_xlabel('Article', fontsize=18)
        box_plot_1.set_ylabel("Length (BPE)", fontsize=18)
        box_plot_1.tick_params(labelsize=16)
        box_plot_2 = sns.boxplot(data=datasets[dataset_name], y="num_summary_bpe", flierprops={"marker": "x"},
                                 notch=True, bootstrap=1000, ax=axes[1])
        box_plot_2.set_xlabel('Summary', fontsize=18)
        box_plot_2.set_ylabel("Length (BPE)", fontsize=18)
        box_plot_2.tick_params(labelsize=16)
        # save
        box_plot_path = os.path.join(dataset_insights_path, f"{dataset_name}_length_box.pdf")
        plt.savefig(box_plot_path)
        plt.close()

        # Violin-Plot
        fig, axes = plt.subplots(1, 2)
        violin_plot_1 = sns.violinplot(data=datasets[dataset_name], y="num_article_bpe", flierprops={"marker": "x"},
                                       notch=True, bootstrap=1000, ax=axes[0])
        violin_plot_1.set_xlabel('Article', fontsize=18)
        violin_plot_1.set_ylabel("Length (BPE)", fontsize=18)
        violin_plot_1.tick_params(labelsize=16)
        violin_plot_2 = sns.violinplot(data=datasets[dataset_name], y="num_summary_bpe", flierprops={"marker": "x"},
                                       notch=True, bootstrap=1000, ax=axes[1])
        violin_plot_2.set_xlabel('Summary', fontsize=18)
        violin_plot_2.set_ylabel("Length (BPE)", fontsize=18)
        violin_plot_2.tick_params(labelsize=16)
        # save
        violin_plot_path = os.path.join(dataset_insights_path, f"{dataset_name}_length_violin.pdf")
        plt.savefig(violin_plot_path)
        plt.close()
        sns.set_theme(**DEFAULT_THEME)

        # make a histogram plot for the size of the input documents
        keys_to_histogram_binwidth = {
            'num_article_bpe': 500,
            'num_summary_bpe': 100,
            'num_input_docs': 1,
        }
        keys_to_xlabels = {
            'num_article_bpe': 'Article Length (BPE)',
            'num_summary_bpe': 'Summary Length (BPE)',
            'num_input_docs': 'Number of Input Documents',
        }
        metric_plt_theme = {
            "style": "darkgrid", "rc": {"figure.figsize": (15, 10), "font.size": 24, },
        }
        sns.set_theme(**metric_plt_theme)
        for key in keys_to_xlabels.keys():
            hist_binwidth = keys_to_histogram_binwidth[key]
            if dataset_name == "20Minuten" and key == "num_summary_bpe":
                hist_binwidth = 25

            hist_xlabel = keys_to_xlabels[key]
            max_val = max(datasets[dataset_name][key])
            max_val_rounded = math.ceil(max_val / hist_binwidth) * hist_binwidth
            binrange = (0, max_val_rounded)
            sns.set(rc={'figure.figsize': (15, 10)}, font_scale=1.5, style='darkgrid', font={'family': 'serif'})
            histplot = sns.histplot(data=datasets[dataset_name], x=key, binwidth=hist_binwidth, binrange=binrange)
            histplot.set_xlabel(hist_xlabel, fontsize=27)
            histplot.set_ylabel("Count", fontsize=27)
            histplot.tick_params(labelsize=22)
            plt.tight_layout()
            # save
            hist_plot_path = os.path.join(dataset_insights_path, f"{dataset_name}_{key}_count_histogram.pdf")
            plt.savefig(hist_plot_path)
            plt.close()

            # also save the histogram as a table (latex)
            hist_table_path = os.path.join(dataset_insights_path, f"{dataset_name}_{key}_count_histogram.tex")
            hist_bins = np.arange(binrange[0], binrange[1] + hist_binwidth, hist_binwidth)
            hist_table = pd.cut(datasets[dataset_name][key], bins=hist_bins, include_lowest=True).value_counts(
                sort=False).reset_index(name='Count')
            # save as latex table
            hist_table.to_latex(hist_table_path, index=False)

        # Make an additional table for the numbr of input documents
        # ... group by the number of input documents, and calculate the following: number of entries, average article length, average summary length
        # ... also save this as a table (latex)
        grouped_data = datasets[dataset_name].groupby("num_input_docs")
        num_input_docs_table = grouped_data.agg(
            num_entries=("num_input_docs", "count"),
            avg_article_length=("num_article_bpe", "mean"),
            avg_summary_length=("num_summary_bpe", "mean"),
        ).reset_index()
        num_input_docs_table["Num. Input Documents"] = num_input_docs_table["num_input_docs"]
        num_input_docs_table["Avg. Total Input Size"] = num_input_docs_table["avg_article_length"]
        num_input_docs_table["Avg. Article Length"] = num_input_docs_table["avg_article_length"] / num_input_docs_table[
            "num_input_docs"]
        num_input_docs_table["Avg. Summary Length"] = num_input_docs_table["avg_summary_length"]
        num_input_docs_table["Count"] = num_input_docs_table["num_entries"]
        # Format the floats to specific precisions
        num_input_docs_table["Num. Input Documents"] = num_input_docs_table["Num. Input Documents"].apply(
            lambda x: f"{x:.0f}")
        num_input_docs_table["Avg. Total Input Size"] = num_input_docs_table["Avg. Total Input Size"].apply(
            lambda x: f"{x:.2f}")
        num_input_docs_table["Avg. Article Length"] = num_input_docs_table["Avg. Article Length"].apply(
            lambda x: f"{x:.2f}")
        num_input_docs_table["Avg. Summary Length"] = num_input_docs_table["Avg. Summary Length"].apply(
            lambda x: f"{x:.2f}")
        # delete the old columns
        num_input_docs_table.drop(columns=["num_input_docs", "avg_article_length", "avg_summary_length", "num_entries"],
                                  inplace=True)
        num_input_docs_table_path = os.path.join(dataset_insights_path, f"{dataset_name}_num_input_docs_table.tex")
        num_input_docs_table.to_latex(num_input_docs_table_path, index=False)

        # make some statistics on the sizes of the articles
        avg_article_length = np.mean(datasets[dataset_name]["num_article_bpe"])
        avg_summary_length = np.mean(datasets[dataset_name]["num_summary_bpe"])
        avg_num_input_docs = np.mean(datasets[dataset_name]["num_input_docs"])
        num_entries = len(datasets[dataset_name])

        # make a text file with those statistics
        stats_text_path = os.path.join(dataset_insights_path, f"{dataset_name}_stats.txt")
        with open(stats_text_path, "w") as f:
            f.write(f"Average Article Length (BPE): {avg_article_length}\n")
            f.write(f"Average Summary Length (BPE): {avg_summary_length}\n")
            f.write(f"Average Number of Input Documents: {avg_num_input_docs}\n")
            f.write(f"Number of Entries: {num_entries}\n")

        sns.set_theme(**DEFAULT_THEME)

        # for MDS datasets, make an additional plot for the number of input documents within the dataset
        # ... also save this as a table (latex)
        #     mds_names

    preprocessed_cache_path = os.path.join("resources", "preprocessed_experiments_cache")
    pathlib.Path(preprocessed_cache_path).mkdir(parents=True, exist_ok=True)

    # Purge the cache of already preprocessed files if specified
    if no_cache_reload_all:
        # Check the cache for the number of files that are already in the cache but will be purged
        purge_files = []
        for _, model_name, filename in zip(dfs, model_names_list, filenames_list):
            cache_file_path = os.path.join(preprocessed_cache_path, model_name, filename)
            if os.path.exists(cache_file_path):
                print(f"Warning: Cache file will be purged: {cache_file_path}")
                purge_files.append(cache_file_path)

        print(f"Purging preprocessing cache... purging {len(purge_files)} files")
        for purge_file in purge_files:
            os.remove(purge_file)

    # Preprocess the result files 1-by-1 if not already present
    all_processed_dfs = []
    for df, model_name, filename in zip(dfs, model_names_list, filenames_list):
        # ensure directory exstis
        curr_dir_path = os.path.join(preprocessed_cache_path, model_name)
        pathlib.Path(curr_dir_path).mkdir(parents=True, exist_ok=True)

        # check if the file is already preprocessed -> if not, preprocess and save
        cache_file_path = os.path.join(preprocessed_cache_path, model_name, filename)
        if os.path.exists(cache_file_path):
            print(f"Loading preprocessed file from cache: {cache_file_path}")
            # load the json file into a dataframe
            cached_df = pd.read_json(cache_file_path, orient="records", lines=True)
            all_processed_dfs.append(cached_df)
        else:
            print(f"Preprocessing file: {cache_file_path}")
            preprocessed_df = process_results_for_load(df, shortNames, datasets)
            preprocessed_df.to_json(cache_file_path, orient="records", lines=True)
            all_processed_dfs.append(preprocessed_df)

    out_df = pd.concat(all_processed_dfs)

    # apply some basic extension of the already existing data
    out_df["Prompt Name"] = out_df["Prompt ID"].apply(
        lambda x: prompt_name_map[f"{x}"] if f"{x}" in prompt_name_map else "[ERROR]")

    def mds_custom_prompt_name(row):
        # prompt_name_MDS_override[Task Name][Prompt ID] -> Prompt Name
        if row["Task Name"] not in prompt_name_MDS_override:
            return row["Prompt Name"]
        else:
            if row["Prompt ID"] not in prompt_name_MDS_override[row["Task Name"]]:
                return row["Prompt Name"]
            else:
                return prompt_name_MDS_override[row["Task Name"]][row["Prompt ID"]]

        # return row["Preprocessing + N-Shot"]

    out_df["Prompt Name"] = out_df.apply(lambda row: mds_custom_prompt_name(row), axis=1)
    # out_df["Prompt Name"] = out_df.apply(
    #     lambda row: row["Preprocessing + N-Shot"] if row["Prompt ID"] == "100" else row["Prompt Name"], axis=1)

    # make sure seahorse scores are floats
    seahorse_float_cols = ["Attribution", "Conciseness", "Main Ideas"]
    for seahorse_float_col in seahorse_float_cols:
        out_df[seahorse_float_col] = out_df[seahorse_float_col].astype(float, copy=True)

    # Process the Seahorse scores (also for MDS specific reports)
    seahorse_metrics = ["attribution", "conciceness", "main-ideas"]
    seahorse_metric_to_name = {
        "attribution": "Attribution",
        "conciceness": "Conciseness",
        "main-ideas": "Main Ideas"
    }
    seahorse_base_column_subname = " (MDS Scores)"
    seahorse_aggregate_funcs = {
        " (St. Dev.)": np.std,
        " (Mean)": np.mean,
        " (Median)": np.median,
        " (Min)": np.min,
        " (Max)": np.max,
        " (90th Percentile)": lambda x: np.percentile(x, 90),
        " (75th Percentile)": lambda x: np.percentile(x, 75),
        " (25th Percentile)": lambda x: np.percentile(x, 25),
        " (10th Percentile)": lambda x: np.percentile(x, 10),
        " (First)": lambda x: x[0],
        " (Last)": lambda x: x[-1],
        # averaging all but the first and the last
        " (Middle)": lambda x: np.mean(x[1:-1]) if len(x) > 2 else np.mean(x),
    }
    # For MDS Data -> if only 1 input article -> the columns are nan now -> fill with values from base seahorse cols
    expect_nan_col_extensions = [" (Truth Values)", " (MDS Scores)", " (Avg. Score)", " (Agreement)"]
    expect_nan_col_names = [f"{seahorse_metric_to_name[seahorse_metric]}{extension}" for seahorse_metric in
                            seahorse_metrics for extension in expect_nan_col_extensions]

    def fill_mds_n_input_docs_one(row):
        # if expect_nan_col_names in the dataframe -> continue
        if not all([mds_colname in row.keys() for mds_colname in expect_nan_col_names]):
            return row

        if row["N-Input Docs"] == 1:
            for seahorse_metric in seahorse_metrics:
                seahorse_metric_name = seahorse_metric_to_name[seahorse_metric]

                row[f"{seahorse_metric_name} (MDS Scores)"] = [row[f"{seahorse_metric_name}"]]
                row[f"{seahorse_metric_name} (Truth Values)"] = [row[f"{seahorse_metric_name} (Truth Value)"]]
                row[f"{seahorse_metric_name} (Avg. Score)"] = row[f"{seahorse_metric_name}"]
                row[f"{seahorse_metric_name} (Agreement)"] = 1.0
        return row

    out_df = out_df.apply(lambda row: fill_mds_n_input_docs_one(row), axis=1)

    # For MDS Data -> calculate some more aggregates based on the scores themselves
    for seahorse_metric in seahorse_metrics:
        seahorse_base_data_col = f"{seahorse_metric_to_name[seahorse_metric]}{seahorse_base_column_subname}"
        if seahorse_base_data_col not in out_df.columns:
            continue
        # First make sure the base-column contains a list of floats
        out_df[seahorse_base_data_col] = out_df[seahorse_base_data_col].apply(lambda x: [float(y) for y in x])

        for name, agg_func in seahorse_aggregate_funcs.items():
            seahorse_dst_col = f"{seahorse_metric_to_name[seahorse_metric]}{name}"
            out_df[seahorse_dst_col] = out_df[seahorse_base_data_col].apply(lambda x: agg_func(x))

        # Specific annotations for the Summarization-Chain analysis
        # TODO: Extend to apply to other comparison methods depending on requirements

    # Prompt Language annotation
    prompts = get_prompts_by_experiment_name(experiment_name)
    prompt_langs = {}
    lang_detector = get_de_lang_detector()
    threshold = 0.5
    empty_val = "other"
    for promptID in prompts:
        prompt_langs[promptID] = get_lang_from_detector(lang_detector, prompts[promptID], threshold, empty_val)
    # calculate the effect of the prompt being in the target language (german) on the target language being the same (german)
    out_df['Prompt Language'] = out_df.apply(lambda row: prompt_langs[f"{row['Prompt ID']}"], axis=1)
    out_df['Prompt Lang. (Separator)'] = out_df.apply(
        lambda row: f"{row['Prompt Language']} ({row['Has Prompt-Summary Separator']})", axis=1)

    out_df["Model Type"] = out_df["Model"].apply(lambda x: model_type_by_shortname[x])

    # Fill the column "Is Bulletpoint Summary" based on a heuristic
    def is_bulletpoint_summary(prediction):
        if prediction is None or len(prediction) == 0:
            return False

        # Remove "Zusammenfassung:" from prefix
        if prediction.startswith("Zusammenfassung:"):
            prediction = prediction[len("Zusammenfassung:"):]
        # strip from whitespace
        prediction = prediction.strip()

        # Make regex to identify whether the multiple lines in the summary start with the following characters:
        # •, -, *
        # or they are an enumeration (start with number and ., e.g. 1. 2. etc.)
        # NOTE: Assume at least two lines starting with that pattern to be considered a bulletpoint summary)
        # is_bulletpoint_regex = re.compile(r"^(•|\-|\*|(\d+\.)).*")
        # bulletpoint_pattern = r"^(?:\s*(?:•|-|\*)|\d+\.)"
        bulletpoint_pattern2 = r"^(•|\-|\*|(\d+\.)).*"
        lines = prediction.split("\n")
        is_bullet_line = [re.match(bulletpoint_pattern2, line) for line in lines]
        # if at least two lines are bulletpoints (i.e. two entries that are not None) -> return True
        if len(list(filter(None, is_bullet_line))) >= 2:
            return True
        return False

    out_df["Is Bulletpoint Summary"] = out_df["Prediction"].apply(lambda x: is_bulletpoint_summary(x))
    out_df["Dataset Annotation (Is Bullet Point Summary)"] = out_df.apply(
        lambda row: f"{row['Dataset Annotation']} {'(bullet-point)' if row['Is Bulletpoint Summary'] else ''}", axis=1)

    def method_annotation(row, short=False):
        preprocessing_method = row["Preprocessing Method"]
        preprocessing_method_abbr = preprocessing_abbr(preprocessing_method)
        preprocessing_params = row["Preprocessing Parameters"]
        dataset_ann = row["Dataset Annotation"]
        prompt_id = row["Prompt ID"]
        task_name = row["Task Name"]

        prompt_name = prompt_name_map[f"{prompt_id}"]
        if task_name in prompt_name_MDS_override:
            if prompt_id in prompt_name_MDS_override[task_name]:
                prompt_name = prompt_name_MDS_override[task_name][f"{prompt_id}"]

        if task_name == "MDS2S":  # 2-stage summarization
            if "mds-summarization-chain" in experiment_name:
                replace_orders = [",\noriginal order", ",\nrandom order"]
                for replace_order in replace_orders:
                    dataset_ann = dataset_ann.replace(replace_order, "")
                return dataset_ann
            return f"2-Stage ({preprocessing_params})"
        elif task_name == "MDS":
            if preprocessing_method == "Summarization Chain" or dataset_ann == "Summarization Chain":  # Summarization-Chain
                return "Chain"
            elif preprocessing_method == "Summarization Chain (Ablation)" or dataset_ann == "Summarization Chain (Ablation)":  # Summarization-Chain (Ablation)
                return "Chain Abl."
            elif "truncated" in dataset_ann:  # truncated baseline
                if short:
                    return "Baseline"
                else:
                    return "Baseline (trunc.)"
            else:
                return "ERROR!?"
        elif task_name == "MDSSumm":
            if "cleaning" in dataset_ann:
                return "Baseline (cleaned)"
            else:
                return f"Baseline ({prompt_name})"
        elif task_name == "MDSFCO":
            return preprocessing_method_abbr

        return preprocessing_method

    out_df["Method"] = out_df.apply(lambda row: method_annotation(row, short=True), axis=1)
    out_df["Preprocessing Method"] = out_df["Dataset ID"].apply(
        lambda x: preprocessing_method[x] if x in preprocessing_method else "")

    # add an additional column to distinguish the size of the input article
    size_threshold = 3500
    in_group_append = f"(>= {size_threshold} Tokens)"
    out_group_append = f"(< {size_threshold} Tokens)"

    size_subset_big = "\u2265 3500 Tokens"
    size_subset_small = "< 3500 Tokens"

    out_df["Large Article"] = out_df["#Input Article Tokens"].apply(lambda x: True if x >= size_threshold else False)
    out_df["Size subset"] = out_df["Large Article"].apply(lambda x: size_subset_big if x else size_subset_small)
    out_df["Dataset Annotation (incl. article size)"] = out_df.apply(
        lambda row: f"{row['Dataset Annotation']}\n{in_group_append}" if row["Large Article"] else
        f"{row['Dataset Annotation']}\n{out_group_append}", axis=1)

    out_df["Dataset Annotation (Prompt Name)"] = out_df.apply(
        lambda row: f"{row['Dataset Annotation']}\n{row['Prompt Name']}", axis=1)

    # add identifiers for the few-shot experiments
    out_df["Preprocessing (N-Shot) (#Tokens)"] = out_df.apply(
        lambda row: f"{row['Preprocessing Method']} ({row['N-Shot']}-shot) ({row['Preprocessing Parameters']})",
        axis=1)
    remove_token_str = lambda x: x.replace(" Tokens", "")
    out_df["Method (N-Shot)"] = out_df.apply(
        lambda row: f"{method_annotation(row, short=True)}\n({row['N-Shot']}-shot)", axis=1)
    out_df["Method (N-Shot) (#Tokens)"] = out_df.apply(
        lambda
            row: f"{preprocessing_abbr(row['Preprocessing Method'])} ({row['N-Shot']}-shot) ({remove_token_str(row['Preprocessing Parameters'])})",
        axis=1)
    out_df["Method (N-Shot) (Source)"] = out_df.apply(
        lambda
            row: f"{preprocessing_abbr(row['Preprocessing Method'])} ({row['N-Shot']}-shot) ({example_source_abbreviation(row['Dataset Annotation'])})",
        axis=1)
    out_df["Method (#Tokens) (Source)"] = out_df.apply(
        lambda
            row: f"{preprocessing_abbr(row['Preprocessing Method'])} ({remove_token_str(row['Preprocessing Parameters'])}) ({example_source_abbreviation(row['Dataset Annotation'])})",
        axis=1)
    out_df["(N-Shot) (#Tokens) (Source)"] = out_df.apply(
        lambda
            row: f"({row['N-Shot']}-shot) ({remove_token_str(row['Preprocessing Parameters'])}) ({example_source_abbreviation(row['Dataset Annotation'])})",
        axis=1)

    # Annotation specifically for comparing example source with number of shots
    def example_source_annotation(row):
        dataset_ann = row["Dataset Annotation"]
        task_name = row["Task Name"]
        num_shots = row["N-Shot"]
        example_source = example_source_abbreviation(row['Dataset Annotation'])

        if task_name == "MDS":
            if "truncated" in dataset_ann:  # truncated baseline
                return "Baseline (truncated)\n(0-shot)"
            else:
                return "NOT IMPLEMENTED"
        elif task_name == "MDSFCO":
            if num_shots == "0":
                return "(0-shot)"
            else:
                return f"{example_source} ({num_shots}-shot)"

        return "Not Implemented"

    out_df["Example Source + N-Shot"] = out_df.apply(lambda row: example_source_annotation(row), axis=1)

    # NOTE: using "Dataset ID", can re-assign annotations when necessary here

    return out_df


def process_results_for_load(df, shortNames, datasets):
    df.rename(columns=column_rename_map, inplace=True)
    df.rename(columns={"Model": "Model-Identifier"}, inplace=True)
    df["Model"] = df["Model-Identifier"].map(shortNames)
    df["Dataset ID"] = df["Dataset"].apply(lambda x: x)
    df["Dataset Annotation"] = df["Dataset"].apply(
        lambda x: datasetAnnotationMap[x] if x in datasetAnnotationMap else "")
    df["Preprocessing Method"] = df["Dataset"].apply(
        lambda x: preprocessing_method[x] if x in preprocessing_method else "")
    df["Preprocessing Parameters"] = df["Dataset"].apply(
        lambda x: preprocessing_parameters[x] if x in preprocessing_parameters else "")
    df["Preprocessing Order"] = df["Dataset"].apply(
        lambda x: preprocessing_order[x] if x in preprocessing_order else "")
    df["N-Shot"] = df.apply(lambda row: dataset_n_fewshot_annotation_map[row["Dataset"]] if row[
                                                                                                "Dataset"] in dataset_n_fewshot_annotation_map else
    row["N-Shot"], axis=1)
    df["Preprocessing + N-Shot"] = df.apply(lambda row: f"{row['Preprocessing Method']} ({row['N-Shot']}-shot)", axis=1)
    df["Dataset"] = df["Dataset"].apply(lambda x: x if x not in datasetNameMap else datasetNameMap[x])
    df["Prompt Description"] = df["Prompt ID"].apply(
        lambda x: prompt_description[x] if x in prompt_description else "[ERROR]")
    df["Prompt Variant"] = df["Prompt ID"].apply(lambda x: prompt_variant[x] if x in prompt_variant else "[ERROR]")
    df["Has Prompt-Summary Separator"] = df["Prompt ID"].apply(
        lambda x: prompt_annotation[x] if x in prompt_annotation else "[ERROR]")
    df["Prompt Desc. [ID]"] = df.apply(lambda row: f"{row['Prompt Description']} [{row['Prompt ID']}]", axis=1)

    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    text_to_bpe = lambda x: tiktoken_encoder.encode(x)
    text_to_tokens = lambda x: list(itertools.chain.from_iterable(list(somajo_tokenizer.tokenize_text([x]))))

    df["#Predicted Tokens"] = df["Prediction"].apply(lambda x: len(text_to_bpe(x)))
    df["#Predicted Full Tokens"] = df["Prediction"].apply(lambda x: len(text_to_tokens(x)))
    df["#Prompt Tokens"] = df["Prompt"].apply(lambda x: len(text_to_bpe(x)))
    # df["#Prompt Full Tokens"] = df["Prompt"].apply(lambda x: len(text_to_tokens(x)))

    # change the type of the following columns to string
    str_cols = ["N-Shot", "Prompt ID"]
    for col in str_cols:
        df[col] = df[col].astype(str)
    # change "Temperature" column to string (format 1 decimal point)
    df["Temperature"] = df["Temperature"].apply(lambda x: f"{x:.1f}")

    # Fill the df with the calculated values
    for idx, row in df.iterrows():
        dataset_name = row["Dataset"]
        summary = row["GT-Summary"]
        summary_stripped = summary.strip()

        dataset_df = datasets[dataset_name]
        candidate_rows = dataset_df[dataset_df["summary"].apply(
            lambda x: x == summary or x in summary or summary in x or x in summary_stripped or summary_stripped in x)]
        if len(candidate_rows) == 0:
            print(f"Warning: No dataset row found for dataset {dataset_name} and summary {summary}")
            candidate_rows = dataset_df[dataset_df["summary"].apply(lambda x: summary_stripped[:100] in x)]
            if len(candidate_rows) == 1:
                print(f"Using a prefix of the stripped summary")
            else:
                print(f"Warning: No dataset row found for dataset {dataset_name} and summary {summary_stripped[:100]}")
                exit(1)
        elif len(candidate_rows) > 1:
            print(f"Warning: Multiple dataset rows found for dataset {dataset_name} and summary {summary}")

        df.at[idx, 'doc_id'] = candidate_rows.iloc[0]['id']
        df.at[idx, "N-Input Docs"] = candidate_rows.iloc[0]['num_input_docs']
        df.at[idx, "#Input Article Tokens"] = candidate_rows.iloc[0]['num_article_bpe']
        df.at[idx, "#Input Article Full Tokens"] = candidate_rows.iloc[0]['num_article_tokens']
        df.at[idx, "#GT-Summary Tokens"] = candidate_rows.iloc[0]['num_summary_bpe']
        df.at[idx, "#GT-Summary Full Tokens"] = candidate_rows.iloc[0]['num_summary_tokens']

        # Recalculate coverage and density
        article = candidate_rows.iloc[0]['article']
        prediction = row["Prediction"]
        language = shortLangToLanguage[datasetLanguage[dataset_name]]
        fragment = Fragments(article, prediction, language=language)
        df.at[idx, 'Coverage'] = fragment.coverage()
        df.at[idx, 'Density'] = fragment.density()
        df.at[idx, 'Compression (Article)'] = fragment.compression()

        # recalculate the compression ratio
        if df.iloc[idx]["#Predicted Tokens"] == 0:
            compression = 0
        else:
            compression = df.iloc[idx]["#Input Article Tokens"] / df.iloc[idx]["#Predicted Tokens"]
        if df.iloc[idx]["#Predicted Full Tokens"] == 0:
            compression_full = 0
        else:
            compression_full = df.iloc[idx]["#Input Article Full Tokens"] / df.iloc[idx]["#Predicted Full Tokens"]
        df.at[idx, "Compression"] = compression
        df.at[idx, "Compression (Full)"] = compression_full

    return df


# Function to save DataFrame as CSV under the experiment name
def save_dataframe(df, experiment_name):
    report_folder = "data"
    pathlib.Path(report_folder).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(report_folder, f"{experiment_name}.csv")
    df.to_csv(csv_path, index=False)


def load_dataframe(experiment_name):
    report_folder = "data"
    csv_path = os.path.join(report_folder, f"{experiment_name}.csv")
    df = pd.read_csv(csv_path)
    return df


# Function to make plots and create an overview report
def create_preprocessed_report(df, experiment_name, metric_names, prompts, skip_lang=True):
    # groupByList = ["Model", "Prompt ID"]

    experiment_path = os.path.join("reports", experiment_name)

    # templateLoader = jinja2.FileSystemLoader(searchpath="./templates")
    # templateEnv = jinja2.Environment(loader=templateLoader)
    # TEMPLATE_FILE = "report_template.html"
    # template = templateEnv.get_template(TEMPLATE_FILE)
    # outputText = template.render()
    # {% include ['my_template.html', 'another_template.html'] %}

    # prepare the subfolder for the given report to save the plots and the overview report to
    report_folder = "reports"
    pathlib.Path(os.path.join(report_folder, experiment_name)).mkdir(parents=True, exist_ok=True)

    # Calculate the languages
    if not skip_lang:
        df = lang_detect(df, "Prediction")
    # before filtering out any rows -> save
    df.to_csv(os.path.join(experiment_path, "df_all.csv"), index=False)

    # create the statistics for the empty predictions
    # ... calculate the success rate per prompt
    # ... calculate the top 5 documents with the worst success rate
    df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions = empty_statistics(
        df, groupbyList=groupByList)
    # Filter out the empty predictions
    df_empty = df[df["Prediction"] == ""]
    df = df[df["Prediction"] != ""]

    # Calculate the language of the predicted text using spacy language detection
    # ... create a new column in the dataframe containing the predicted language
    # ... make a plot showing the distribution of the predicted languages per model and prompt
    if not skip_lang:
        # df_lang_stat, df_prompt_lang_effect = language_statistics(df, experiment_path, prompts, groupbyList=groupByList)
        # Filter out the non-german predictions
        tgt_lang = datasetLanguage[df["Dataset"].iloc[0]]  # should all be the same dataset
        df_non_german = df[df["Language"] != tgt_lang]
        df = df[df["Language"] == tgt_lang]

    # Save all stuff
    print("Saving results...")

    df.to_csv(os.path.join(experiment_path, "df_filtered.csv"), index=False)
    df_empty.to_csv(os.path.join(experiment_path, "df_empty.csv"), index=False)
    df_empty_stat.to_csv(os.path.join(experiment_path, "df_empty_stat.csv"), index=False)
    df_num_empty_docs.to_csv(os.path.join(experiment_path, "df_num_empty_docs.csv"), index=False)
    df_num_empty_prompts.to_csv(os.path.join(experiment_path, "df_num_empty_prompts.csv"), index=False)
    worst_empty_docs.to_csv(os.path.join(experiment_path, "worst_empty_docs.csv"), index=False)
    worst_empty_promptVersions.to_csv(os.path.join(experiment_path, "worst_empty_promptVersions.csv"), index=False)

    if not skip_lang:
        # df_lang_stat.to_csv(os.path.join(experiment_path, "df_lang_stat.csv"), index=False)
        # df_prompt_lang_effect.to_csv(os.path.join(experiment_path, "df_prompt_lang_effect.csv"), index=False)
        df_non_german.to_csv(os.path.join(experiment_path, "df_non_german.csv"), index=False)


def empty_statistics(df, groupbyList=["Model", "Prompt ID"], text_col="Prediction", topN=5, docCol="doc_id",
                     promptCol="Prompt ID") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.shape[0] == 0:
        return None, None, None, None, None
    print("Calculating empty-prediction statistics...")
    # Get the count of empty predictions
    df_empty_stat = df[df[text_col] == ""].groupby(groupbyList).agg(
        {"Prediction": "count"}).reset_index()
    df_empty_stat = df_empty_stat.rename(columns={"Prediction": "empty_count"})

    # Calculate the success rate per document
    df_num_empty_docs = df[df[text_col] == ""].groupby(docCol).agg(
        {"Prediction": "count"}).reset_index()
    df_num_empty_docs = df_num_empty_docs.rename(columns={"Prediction": "empty_count"})

    # Calculate the success rate per prompt
    df_num_empty_prompts = df[df[text_col] == ""].groupby(promptCol).agg(
        {"Prediction": "count"}).reset_index()
    df_num_empty_prompts = df_num_empty_prompts.rename(columns={"Prediction": "empty_count"})

    # Get the worst documents and prompts (with the highest number of empty predictions)
    worst_empty_docs_IDs = df_num_empty_docs.sort_values(by="empty_count", ascending=False).head(topN)[docCol].tolist()
    worst_empty_promptVersions_IDs = df_num_empty_prompts.sort_values(by="empty_count", ascending=False).head(topN)[
        promptCol].tolist()
    worst_empty_docs = df[df[docCol].isin(worst_empty_docs_IDs)]
    worst_empty_promptVersions = df[df[promptCol].isin(worst_empty_promptVersions_IDs)]

    # df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions
    return df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions


def failure_statistics_plot(df_all, experiment_path, groupbyList=["Model", "Prompt ID"], groupByListExtension=[],
                            groupByIterator=[], x_group="Temperature", text_col="Prediction", langCol="Language",
                            docCol="doc_id", promptCol="Prompt ID"):
    if df_all.shape[0] == 0:
        return
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"
    assert x_group not in groupbyList, "x_group must not be in groupbyList"

    subfolder_name = "failure_statistics"
    pathlib.Path(os.path.join(experiment_path, subfolder_name)).mkdir(parents=True, exist_ok=True)

    # map all float nan values in logit_0 to empty strings
    df_all[text_col] = df_all[text_col].apply(lambda x: "" if isinstance(x, float) and math.isnan(x) else x)

    def get_failure(row):
        if row[text_col] == "":
            return "empty"
        elif row[langCol] != "de":
            return "non-german"
        else:
            return "ok"

    x_order = ["empty", "non-german", "ok"]

    # Calculate the failures
    df_failures = df_all.copy()
    df_failures["failure"] = df_failures.apply(lambda x: get_failure(x), axis=1)
    df_failures['failure'] = df_failures['failure'].astype('category')

    # make a table for each group-by variation, and save as csv and latex table
    for gbl in [[groupByList[0]], [groupByList[1]], groupByList] + groupByListExtension:
        df_failures_grouped = df_failures.groupby(gbl + ["failure"]).agg({"Prediction": "count"}).reset_index()
        df_failures_grouped = df_failures_grouped.rename(columns={"Prediction": "Count"})

        df_failures_grouped.to_csv(os.path.join(experiment_path, subfolder_name,
                                                f"failure_statistics_group_by__{'_'.join(gbl)}.csv"), index=False)
        # reorganize the dataframe to have the different failure types as columns
        df_failures_grouped_pivot = df_failures_grouped.pivot_table(index=gbl, columns='failure', values='Count',
                                                                    fill_value=math.nan).reset_index()
        df_failures_grouped_pivot.to_csv(os.path.join(experiment_path, subfolder_name,
                                                      f"failure_statistics_group_by__{'_'.join(gbl)}_pivot.csv"),
                                         index=False)
        # also save as latex table into a txt file
        df_failures_grouped.to_latex(os.path.join(experiment_path, subfolder_name,
                                                  f"failure_statistics_group_by__{'_'.join(gbl)}.tex"), index=False)
        df_failures_grouped_pivot.to_latex(os.path.join(experiment_path, subfolder_name,
                                                        f"failure_statistics_group_by__{'_'.join(gbl)}_pivot.tex"),
                                           index=False)

        # also calculate the failure rate
        df_failures_grouped_rates = df_failures_grouped_pivot.copy()
        if "empty" in df_failures_grouped_rates.columns:
            empty_failures = df_failures_grouped_rates["empty"]
        else:
            empty_failures = [0] * len(df_failures_grouped_rates)
        if "non-german" in df_failures_grouped_rates.columns:
            non_german_failures = df_failures_grouped_rates["non-german"]
        else:
            non_german_failures = [0] * len(df_failures_grouped_rates)
        if "ok" in df_failures_grouped_rates.columns:
            ok_failures = df_failures_grouped_rates["ok"]
        else:
            ok_failures = [0] * len(df_failures_grouped_rates)
        total_entries = empty_failures + non_german_failures + ok_failures
        total_failures = empty_failures + non_german_failures

        df_failures_grouped_rates["empty rate"] = (empty_failures / total_entries) * 100
        df_failures_grouped_rates["non-german rate"] = (non_german_failures / total_entries) * 100
        df_failures_grouped_rates["ok rate"] = (ok_failures / total_entries) * 100
        df_failures_grouped_rates["total failure rate"] = (total_failures / total_entries) * 100
        # round the rates to 2 decimal places
        df_failures_grouped_rates = df_failures_grouped_rates.round(2)

        df_failures_grouped_rates.to_csv(os.path.join(experiment_path, subfolder_name,
                                                      f"failure_statistics_group_by__{'_'.join(gbl)}_rates.csv"),
                                         index=False)

    # Aggregate the success rates per model and prompt (by category)
    df_failure_stat = df_failures.groupby(groupbyList + ["failure", x_group]).agg({"Prediction": "count"}).reset_index()
    df_failure_stat = df_failure_stat.rename(columns={"Prediction": "count"})
    df_failure_stat = df_failure_stat.reset_index(drop=True)

    # Make a facet-grid plot, make 1 plot per x_group value
    for x_group_val in df_failures[x_group].unique():
        df_failure_stat_x_group = df_failures[df_failures[x_group] == x_group_val]
        failure_plot = sns.FacetGrid(df_failure_stat_x_group, col=groupbyList[0], row=groupbyList[1], height=3,
                                     aspect=1.5)
        failure_plot.map(sns.countplot, 'failure', order=x_order)
        failure_plot_path = os.path.join(experiment_path, subfolder_name,
                                         f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}_{x_group}_{x_group_val}.pdf")
        plt.savefig(failure_plot_path)
        plt.close()

    # Create all possible triplets from groupByIterator list
    groupByIteratorTriplets = list(itertools.combinations(groupByIterator, 3))
    for (a, b, c) in groupByIteratorTriplets:
        # make sure labels are strings
        df_failures[a] = df_failures[a].astype(str)
        df_failures[b] = df_failures[b].astype(str)
        df_failures[c] = df_failures[c].astype(str)

        failure_plot = sns.catplot(df_failures, x='failure', order=x_order, col=a, row=b, hue=c, kind="count", height=3,
                                   aspect=1.5)
        # annotate the values
        for ax in failure_plot.axes.flat:
            for p in ax.patches:
                ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x(), p.get_height() + 0.02), fontsize='xx-small',
                            rotation='vertical')
        # failure_plot = sns.FacetGrid(df_failures, col=groupbyList[0], row=groupbyList[1], height=3, aspect=1.5)
        # failure_plot.map(sns.countplot, 'failure')
        failure_plot_path = os.path.join(experiment_path, subfolder_name,
                                         f"failure_statistics_exploration__{a}_{b}_{c}.pdf")
        plt.savefig(failure_plot_path)
        plt.close()

    # also do a similar plot for all passed combinations of groupbyList entries
    for gbl in [groupByList] + groupByListExtension:
        a = gbl[0]
        b = gbl[1]
        # make sure labels are strings
        df_failures[a] = df_failures[a].astype(str)
        df_failures[b] = df_failures[b].astype(str)
        row_order = get_sorted_labels(df_failures, b)
        col_order = get_sorted_labels(df_failures, a)

        failure_plot = sns.catplot(df_failures, x='failure', order=x_order, col=a, row=b, kind="count",
                                   height=3, aspect=1.5, row_order=row_order, col_order=col_order)

        failure_plot_path = os.path.join(experiment_path, subfolder_name,
                                         f"failure_statistics_group_by_exploration__{a}_{b}.pdf")
        plt.savefig(failure_plot_path)

        # annotate the values
        for ax in failure_plot.axes.flat:
            for p in ax.patches:
                ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x(), p.get_height() + 0.02),
                            fontsize='xx-small',
                            rotation='vertical')
        # failure_plot = sns.FacetGrid(df_failures, col=groupbyList[0], row=groupbyList[1], height=3, aspect=1.5)
        # failure_plot.map(sns.countplot, 'failure')
        failure_plot_path = os.path.join(experiment_path, subfolder_name,
                                         f"failure_statistics_group_by_exploration__{a}_{b}_annotated.pdf")
        plt.savefig(failure_plot_path)
        plt.close()

    # also save the failure statistics as csv
    df_failure_stat.to_csv(os.path.join(experiment_path, subfolder_name,
                                        f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}_{x_group}.csv"),
                           index=False)
    # reorganize the dataframe to have the different failure types as columns
    df_failure_stat_pivot = df_failure_stat.pivot_table(index=groupbyList + [x_group], columns='failure',
                                                        values='count',
                                                        fill_value=math.nan).reset_index()
    df_failure_stat_pivot.to_csv(os.path.join(experiment_path, subfolder_name,
                                              f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}_{x_group}_pivot.csv"),
                                 index=False)
    # also save as latex table into a txt file
    df_failure_stat_pivot.to_latex(os.path.join(experiment_path, subfolder_name,
                                                f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}_{x_group}_pivot.txt"),
                                   index=False)


def get_lang_detector(nlp, name):
    return LanguageDetector()


def get_de_lang_detector():
    nlp = spacy.load("de_core_news_sm")
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe("language_detector", last=True)
    return nlp


def eval_lang_detector(lang_detector, text) -> Tuple[str, float]:
    doc = lang_detector(text)
    return doc._.language['language'], doc._.language['score']


def get_lang_from_detector(lang_detector, text, threshold, empty_val) -> str:
    if text == "":
        return empty_val

    lang, score = eval_lang_detector(lang_detector, text)
    if score < threshold:
        return empty_val
    else:
        return lang


def lang_detect(df, col_name, fast=True):
    """
    Detect the language of the text in the given column of the dataframe and add a new column containing the language
    :param df: dataframe containing the text to be analyzed
    :param col_name: name of the column containing the text to be analyzed
    :param fast: if True, use fasttext language detection, otherwise use spacy language detection
    """
    print(f"Detecting Languages... [{'fasttext' if fast else 'spacy'}]")

    if fast:
        # Use Fasttext Language Detection
        pretrained_model_path = './resources/lid.176.bin'
        fast_lang_detector = fasttext.load_model(pretrained_model_path)

        # Predict
        # remove any newline characters from the text
        text_list = [text.replace("\n", " ") if isinstance(text, str) else "" for text in df[col_name].tolist()]
        predictions = fast_lang_detector.predict(text_list)
        post_preds = [(t[0][0].replace('__label__', ''), t[1][0]) for t in zip(predictions[0], predictions[1])]

        # Postprocess
        threshold = 0.5
        empty_val = "other"
        # compute the predicted language and assign it to the dataframe
        df["Language"] = [lang if score >= threshold else empty_val for lang, score in post_preds]
        df["Language Score"] = [score for lang, score in post_preds]
        return df

    else:
        # Use Spacy Language Detection
        lang_detector = get_de_lang_detector()
        threshold = 0.5
        empty_val = "other"

        tqdm.pandas()

        # generate dataframe column by df.apply
        df["Language"] = df[col_name].progress_apply(
            lambda x: get_lang_from_detector(lang_detector, x, threshold, empty_val))
        return df


def language_statistics(df, experiment_path, prompts, groupbyList=["Model", "Prompt ID"], promptVersionCol="Prompt ID",
                        modelCol="Model"):
    if df.shape[0] == 0:
        return None, None
    print("Calculating language statistics...")

    # get the languages of each prompt
    prompt_langs = {}
    lang_detector = get_de_lang_detector()
    threshold = 0.5
    empty_val = "other"
    for promptID in prompts:
        prompt_langs[promptID] = get_lang_from_detector(lang_detector, prompts[promptID], threshold, empty_val)

    # calculate the language distribution per model and prompt
    df_lang_stat = df.groupby(groupbyList + ["Language"]).agg({"Prediction": "count"}).reset_index()

    # calculate the effect of the prompt being in the target language (german) on the target language being the same (german)
    df_prompt_langs = df.copy()
    df_prompt_langs['prompt_lang'] = df_prompt_langs.apply(lambda row: prompt_langs[f"{row[promptVersionCol]}"], axis=1)
    groupbyListLangEffect = list(set([col for col in groupbyList if col != promptVersionCol] + [
        modelCol]))  # exclude promptVersion from groupbyList
    df_prompt_lang_effect = df_prompt_langs.groupby(groupbyListLangEffect + ["prompt_lang", "Language"]).agg(
        {"Prediction": "count"}).reset_index()

    # Make a language effect plot, showing the effect of the prompt language on the predicted language (for each model)
    # Grouping all other languages together
    df_prompt_lang_effect_plot = df_prompt_lang_effect.copy()
    df_prompt_lang_effect_plot["Language"] = df_prompt_lang_effect_plot["Language"].apply(
        lambda x: "other" if x != "de" else x)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.groupby([modelCol, "prompt_lang", "Language"]).agg(
        {"Prediction": "sum"}).reset_index()
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.rename(columns={"Prediction": "count"})
    df_prompt_lang_effect_plot["count"] = df_prompt_lang_effect_plot["count"].astype(int)
    # Make the actual plot
    lang_effect_plot = sns.FacetGrid(df_prompt_lang_effect_plot, col="prompt_lang", row=modelCol, height=3, aspect=1.5)
    x_order = get_sorted_labels(df_prompt_lang_effect_plot, "Language")
    lang_effect_plot.map(sns.barplot, "Language", "count", order=x_order)
    token_distr_plot_path = os.path.join(experiment_path, "prompt_lang_effect_plot.pdf")
    plt.savefig(token_distr_plot_path)
    plt.close()
    #
    # sns.set_theme(style="whitegrid")
    # g = sns.catplot(
    #     data=df_prompt_lang_effect_plot, kind="bar",
    #     x="prompt_lang", y="count", hue="Language",
    #     ci="sd", palette="dark", alpha=.6, height=6
    # )
    # g.despine(left=True)
    # g.set_axis_labels("", "Count")
    # g.legend.set_title("Predicted Language")
    # g.savefig(os.path.join(experiment_path, "prompt_lang_effect_plot.pdf"))

    # Language effect plot, showing the distribution of the predicted languages per model and prompt (with the prompt language as hue) ->
    df_prompt_lang_effect_plot = df_prompt_lang_effect.copy()
    plot_langs = ["de", "en", "af", "other"]
    # map languages not in plot_langs to "other"
    df_prompt_lang_effect_plot["Language"] = df_prompt_lang_effect_plot["Language"].apply(
        lambda x: "other" if x not in plot_langs else x)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.groupby(["Model", "prompt_lang", "Language"]).agg(
        {"Prediction": "sum"}).reset_index()
    # fill missing combinations with 0 value (issue with plotting)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.groupby(["Model", "prompt_lang", "Language"]).agg(
        {"Prediction": "sum"}).reset_index()
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.pivot(index=["Model", "prompt_lang"], columns="Language",
                                                                  values="Prediction").reset_index()
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.fillna(0)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.melt(id_vars=["Model", "prompt_lang"], var_name="Language",
                                                                 value_name="Prediction")
    df_prompt_lang_effect_plot["Prediction"] = df_prompt_lang_effect_plot["Prediction"].astype(int)
    # plot
    lang_effect_distr_plot = sns.FacetGrid(df_prompt_lang_effect_plot, col="prompt_lang", row=modelCol, height=3,
                                           aspect=1.5, sharex=True)
    x_order = get_sorted_labels(df_prompt_lang_effect_plot, "Language")
    lang_effect_distr_plot.map(sns.barplot, "Language", "Prediction", order=x_order)
    token_distr_plot_path = os.path.join(experiment_path, "prompt_lang_effect_lang_distr_plot.pdf")
    plt.savefig(token_distr_plot_path)
    plt.close()

    return df_lang_stat, df_prompt_lang_effect


WORD = re.compile('r\w+')


def approxTokenize(text):
    words = WORD.findall(text)
    return words


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def approx_split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    # https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def get_nlp_model():
    nlp = spacy.load("de_core_news_sm")
    return nlp


def sentence_splitting(text: str, approximation: bool = True) -> Tuple[List[str], List[str]]:
    if approximation:
        approxTokens = approxTokenize(text)
        approxSents = approx_split_into_sentences(text)
        return approxSents, approxTokens

    nlp = get_nlp_model()  # disable=["ner", "parser", "tagger"]
    # nlp.add_pipe('sentencizer')
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    tokens = [token.text for token in doc]
    return sentences, tokens


def length_statistics(df, save_base_path, groupbyList=["Model", "Prompt ID"], gblExtension=[], approximation=True,
                      file_suffix=""):
    if df.shape[0] == 0:
        return None
    print("Calculating length statistics...")

    pathlib.Path(os.path.join(save_base_path, "length-statistics")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(save_base_path, "length-dimension-statistics")).mkdir(parents=True, exist_ok=True)

    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"

    # calculate the number of tokens and sentences per prediction
    df_len = df.copy()
    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

    def calculate_lengths(text: str) -> Tuple[int, int, int]:
        text_bpe = tiktoken_encoder.encode(text)
        newline_split_txt = text.split("\n")
        sent_somajo = list(somajo_tokenizer.tokenize_text(newline_split_txt))
        text_somajo = list(itertools.chain.from_iterable(sent_somajo))
        return len(sent_somajo), len(text_somajo), len(text_bpe)

    num_sentences_colname = "# Sentences"
    num_full_tokens_colname = "# Full Tokens"
    num_bpe_tokens_colname = "# BPE Tokens"
    num_bpe_tokens_sumLen_colname = "Summary Length (BPE)"
    avg_sentlen_colname = "Avg. Sentence Length"

    df_len[num_sentences_colname], df_len[num_full_tokens_colname], df_len[num_bpe_tokens_colname] = zip(
        *df_len["Prediction"].progress_apply(lambda x: calculate_lengths(x)))
    df_len[avg_sentlen_colname] = df_len.apply(
        lambda row: row[num_full_tokens_colname] / row[num_sentences_colname] if row[num_sentences_colname] > 0 else 0,
        axis=1)
    full_bckt_size = 25
    bpe__bckt_size = 50
    full_string_size = 4
    bpe__string_size = 4
    bckt_full_colname = f"{num_full_tokens_colname} Bucket"
    bckt_bpe___colname = f"{num_bpe_tokens_colname} Bucket"
    df_len[bckt_full_colname] = df_len[num_full_tokens_colname].apply(
        lambda x: f"{int(math.floor(x / full_bckt_size)) * full_bckt_size}".rjust(
            full_string_size) + "-" + f"{int(math.ceil(x / full_bckt_size)) * full_bckt_size}".rjust(full_string_size))
    df_len[bckt_bpe___colname] = df_len[num_bpe_tokens_colname].apply(
        lambda x: f"{int(math.floor(x / bpe__bckt_size)) * bpe__bckt_size}".rjust(
            bpe__string_size) + "-" + f"{int(math.ceil(x / bpe__bckt_size)) * bpe__bckt_size}".rjust(bpe__string_size))

    # calculate the impact of the prompt on the number of tokens and sentences
    if df_len is None or df_len.groupby(groupbyList) is None:
        print("Unable to create length statistics, ...")
        return
    df_prompt_length_impact = df_len.groupby(groupbyList).agg(
        {num_sentences_colname: "mean", num_full_tokens_colname: "mean", num_bpe_tokens_colname: "mean",
         avg_sentlen_colname: "mean"}).reset_index()

    # add a table with two counters indicating how many entries per group use up more than 4096 tokens in total
    #  and how many use up less than 4096 tokens in total

    df_len['used tokens'] = df_len['#Prompt Tokens'] + df_len['#Predicted Tokens']
    df_len['exceeds_4096'] = df_len['used tokens'].apply(lambda x: 1 if x >= 4096 else 0)
    df_len['below_4096'] = df_len['used tokens'].apply(lambda x: 1 if x < 4096 else 0)
    df_max_length_table = df_len.groupby(groupbyList).agg(
        {'exceeds_4096': 'sum', 'below_4096': 'sum'}).reset_index()
    # save
    df_max_length_table.to_csv(os.path.join(save_base_path, "length-statistics", f"max_length_table{file_suffix}.csv"),
                               index=False)
    # also save as latex table into a txt file
    df_max_length_table.to_latex(os.path.join(save_base_path, "length-statistics",
                                              f"max_length_table{file_suffix}.tex"), index=False)

    # Calculate the distribution statistics in several dimensions
    distr_dimensions = [num_sentences_colname, num_full_tokens_colname, num_bpe_tokens_colname]
    for gbl in [groupbyList] + gblExtension:
        for distr_dim in distr_dimensions:
            df_len_copy = df_len.copy()
            df_len_copy[distr_dim] = df_len_copy[distr_dim].astype(float, copy=True)
            df_len_agg = df_len_copy.groupby(gbl).agg(
                {distr_dim: [
                    percentile_agg(1),
                    percentile_agg(5),
                    percentile_agg(25),
                    np.median,
                    percentile_agg(75),
                    percentile_agg(95),
                    percentile_agg(99),
                    np.std,
                    np.var,
                ]}).reset_index()
            new_colnames = gbl + ["1st Perc.", "5th Perc.", "25th Perc.", "Median", "75th Perc.", "95th Perc.",
                                  "99th Perc.",
                                  "Std. Dev.", "Variance"]
            df_len_agg.columns = new_colnames
            df_len_agg = df_len_agg.round(ROUND_TO_DECIMALS)
            # save as csv and latex table
            df_len_agg.to_csv(
                os.path.join(save_base_path, "length-dimension-statistics",
                             f"Distribution_{distr_dim}__{gbl[0]}_{gbl[1]}.csv"),
                index=False)
            df_len_agg.to_latex(
                os.path.join(save_base_path, "length-dimension-statistics",
                             f"Distribution_{distr_dim}__{gbl[0]}_{gbl[1]}.tex"),
                index=False)

    for gbl in [groupbyList] + gblExtension:
        for agg_func in ['mean', 'median', 'min', 'max']:
            df_prompt_length_impact_gbl = df_len.groupby(gbl).agg(
                {num_sentences_colname: agg_func, num_full_tokens_colname: agg_func,
                 num_bpe_tokens_colname: agg_func, avg_sentlen_colname: agg_func}).reset_index()
            length_impact_path = os.path.join(save_base_path, f"length-statistics",
                                              f"length_impact_{gbl[0]}_{gbl[1]}_{agg_func}.csv")
            df_prompt_length_impact_gbl.to_csv(length_impact_path, index=False)
            # also save as latex table
            df_prompt_length_impact_gbl.to_latex(os.path.join(save_base_path, "length-statistics",
                                                              f"length_impact_{gbl[0]}_{gbl[1]}_{agg_func}.tex"),
                                                 index=False)

            # if median -> plot the median points
            # make a plot showing the median of the number of tokens (per group by combination)
            if agg_func in ['median', 'mean']:
                for y_colname in [num_sentences_colname, num_full_tokens_colname, num_bpe_tokens_colname,
                                  avg_sentlen_colname]:
                    x_order = get_sorted_labels(df_prompt_length_impact_gbl, gbl[0])
                    hue_order = get_sorted_labels(df_prompt_length_impact_gbl, gbl[1])
                    sns.pointplot(data=df_prompt_length_impact_gbl, x=gbl[0], y=y_colname, hue=gbl[1], order=x_order,
                                  hue_order=hue_order,
                                  linestyles='')  # , err_kws={"markersize": 10, "capsize": 0.1, "errwidth": 1.5, "palette": "colorblind"}
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    save_path = os.path.join(save_base_path, "length-statistics",
                                             f"length_impact_{gbl[0]}_{gbl[1]}_{agg_func}_plot_{y_colname}.pdf")
                    plt.savefig(save_path)
                    plt.close()

    for lbl, (a, b) in zip(["", "_flipped"], [(0, 1), (1, 0)]):
        if lbl == "_flipped":
            aspect_ratio = 0.75
        else:
            aspect_ratio = 6
        num_bins = 10

        # subplot grid: one-for each dimension in group-by-list
        # show num_full_tokens, num_bpe_tokens, num_sentences
        # for each, make a histplot, violinplot, boxplot
        tgt_label_list = [
            (num_sentences_colname, ["box", "count", "violin"]),
            (num_full_tokens_colname, ["box", "violin"]),
            (num_bpe_tokens_colname, ["box", "violin"]),
            (bckt_full_colname, ["count"]),
            (bckt_bpe___colname, ["count"]),
        ]
        # for tgt_label, plot_type_list in tgt_label_list:
        #     for plt_type in plot_type_list:
        #         for gbl in gblExtension:
        #             row_order = get_sorted_labels(df_len, gbl[a])
        #             col_order = get_sorted_labels(df_len, gbl[b])
        #             sns.set(font_scale=3)
        #
        #             try:
        #                 if len(row_order) > 8 or len(col_order) > 8:
        #                     raise ValueError("Too much data to plot, ...")
        #
        #                 len_plt = sns.catplot(df_len, row=gbl[a], col=gbl[b], row_order=row_order, col_order=col_order,
        #                                       x=tgt_label, height=5, aspect=aspect_ratio, kind=plt_type)
        #                 sns.set(font_scale=3)
        #                 # len_plt.set_xlabel(gbl[0], fontsize=18)
        #                 # len_plt.set_ylabel(metric_name, fontsize=18)
        #                 # len_plt.tick_params(labelsize=16)
        #                 metrics_n_in_docs_plt_path = os.path.join(save_base_path, f"length-statistics",
        #                                                           f"length_statistics_{tgt_label}_{plt_type}_{gbl[0]}_{gbl[1]}{lbl}.pdf")
        #                 plt.savefig(metrics_n_in_docs_plt_path)
        #                 plt.close()
        #             except:
        #                 plt.close()
        #                 print(
        #                     "Unable to create catplot for lengths, creating catplot for individual dimensions instead...")
        #                 continue
        #
        #                 # aspect_ratio = 1.5
        #                 # if len(row_order) > len(col_order):
        #                 #     larger_order = row_order
        #                 #     a_prime = b
        #                 #     b_prime = a
        #                 # else:
        #                 #     larger_order = col_order
        #                 #     a_prime = a
        #                 #     b_prime = b
        #                 #
        #                 # for larger_cat in larger_order:
        #                 #     col_order_prime = [larger_cat]
        #                 #     # filter to only contain the current larger category
        #                 #     df_len_filtered = df_len[df_len[gbl[b_prime]] == larger_cat]
        #                 #     row_order_prime = get_sorted_labels(df_len_filtered, gbl[a_prime])
        #                 #
        #                 #     len_plt = sns.catplot(df_len_filtered, row=gbl[a_prime], col=gbl[b_prime],
        #                 #                           row_order=row_order_prime, col_order=col_order_prime,
        #                 #                           x=tgt_label, height=5, aspect=aspect_ratio, kind=plt_type)
        #                 #     len_plt_save_path = os.path.join(save_base_path, f"length-statistics",
        #                 #                                      f"length_statistics_{tgt_label}_{plt_type}_{gbl[a_prime]}_{gbl[b_prime]}__{larger_cat}.pdf")
        #                 #     plt.savefig(len_plt_save_path)
        #                 #     print(f"saved {len_plt_save_path}")
        #                 #     # if plt is list_iterator object
        #                 #     if plt is None or isinstance(plt, list):
        #                 #         print("WHY?")
        #                 #     else:
        #                 #         plt.close()

        sns.set_theme(**DEFAULT_THEME)
        #
        # # make plots showing the two distributions (with a subplot grid, one for each dimension in the group-by list) (just showing the number of tokens)
        # # make a subplot grid with one plot for each dimension in the group-by list
        # token_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        # token_distr_plot.map(sns.histplot, "num_tokens", bins=20)
        # token_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"token_distr_plot_hist{lbl}{file_suffix}.pdf")
        # plt.savefig(token_distr_plot_path)
        # plt.close()
        #
        # # make the same plot as violin plot
        # token_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        # token_distr_plot.map(sns.violinplot, "num_tokens", bins=20)
        # token_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"token_distr_plot_violin{lbl}{file_suffix}.pdf")
        # plt.savefig(token_distr_plot_path)
        # plt.close()
        #
        # # make a subplot grid with one plot for each dimension in the group-by list
        # sent_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        # sent_distr_plot.map(sns.histplot, "num_sentences", bins=20)
        # sent_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"sent_distr_plot_hist{lbl}{file_suffix}.pdf")
        # plt.savefig(sent_distr_plot_path)
        # plt.close()
        #
        # # make the same plot as violin plot
        # sent_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        # sent_distr_plot.map(sns.violinplot, "num_sentences", bins=20)
        # sent_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"sent_distr_plot_violin{lbl}{file_suffix}.pdf")
        # plt.savefig(sent_distr_plot_path)
        # plt.close()

    df_len[num_bpe_tokens_sumLen_colname] = df_len[num_bpe_tokens_colname]
    # make a boxplot showing the distribution of tokens (similar to the median plot above, but for the full distribution)
    for y_colname in [num_bpe_tokens_sumLen_colname, num_sentences_colname, num_full_tokens_colname,
                      num_bpe_tokens_colname, avg_sentlen_colname]:
        for gbl in [groupByList] + gblExtension:
            sorted_x_axis_labels = get_sorted_labels(df_len, gbl[0], prioritize_string_order=True)
            sorted_hue_labels = get_sorted_labels(df_len, gbl[1], prioritize_string_order=True)

            sns.set(rc={'figure.figsize': (18, 8)}, font_scale=1.5)
            len_plt = sns.boxplot(df_len, x=gbl[0], y=y_colname, hue=gbl[1], order=sorted_x_axis_labels,
                                  hue_order=sorted_hue_labels, flierprops={"marker": "x"}, notch=True, bootstrap=1000)
            # set the font sizes of the x- and y-labels as well as the ticks
            len_plt.set_xlabel(gbl[0], fontsize=20)
            len_plt.set_ylabel(y_colname, fontsize=20)
            len_plt.tick_params(labelsize=18)
            # set the legend appropriately
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.5, fontsize=20)
            plt.tight_layout()

            # plt.gcf().subplots_adjust(bottom=0.4)
            metrics_n_in_docs_plt_path = os.path.join(save_base_path, f"length-statistics",
                                                      f"length_statistics_{y_colname}_box_{gbl[0]}_{gbl[1]}.pdf")
            plt.savefig(metrics_n_in_docs_plt_path)

            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', borderaxespad=0.5, ncols=2)
            plt.setp(len_plt.get_legend().get_texts(), fontsize='20')
            plt.setp(len_plt.get_legend().get_title(), fontsize='20')
            plt.tight_layout()
            metrics_n_in_docs_plt_path = os.path.join(save_base_path, f"length-statistics",
                                                      f"length_statistics_{y_colname}_box_{gbl[0]}_{gbl[1]}_TopLegend.pdf")
            plt.savefig(metrics_n_in_docs_plt_path)

            plt.close()

    return df_prompt_length_impact


def percentile_agg(percentile):
    # return lambda x: np.percentile(x, percentile)

    def percentile_func(x, **kwargs):
        return np.percentile(x, q=percentile, **kwargs)

    return percentile_func


def bootstrap_CI(statistic, confidence_level=0.95, num_samples=10000):
    def CI(x):
        if len(x) <= 2:
            return math.nan, math.nan
        data = (np.array(x),)  # convert to sequence
        confidence_interval = bootstrap(data, statistic, n_resamples=num_samples, confidence_level=confidence_level,
                                        method='basic', vectorized=True).confidence_interval
        LOW_CI = round(confidence_interval.low, ROUND_TO_DECIMALS)
        HIGH_CI = round(confidence_interval.high, ROUND_TO_DECIMALS)
        return LOW_CI, HIGH_CI

    return CI


def statistics_overview(experiment_path, df, metric_names, groupbyList=["Model", "Prompt ID"], gblExtension=[]):
    if df.shape[0] == 0:
        return None, None, None
    confidence_level = 0.95

    print("Calculating statistics overview tables...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"
    agg_funcs = ["median", "std", "min", "max", percentile_agg(5), percentile_agg(25), percentile_agg(50),
                 percentile_agg(75), percentile_agg(95),
                 bootstrap_CI(np.median, confidence_level=confidence_level)]
    agg_names = ["Median", "St.Dev.", "Min", "Max", "5th Perc.", "25th Perc.", "50th Perc.", "75th Perc.", "95th Perc.",
                 f"Median {int(confidence_level * 100)}% CI"]
    agg_dict = {agg_name: agg_func for agg_name, agg_func in zip(agg_names, agg_funcs)}

    out_overview = []
    out_detail = []
    ci_level_str = f"{int(confidence_level * 100)}% CI"
    ci_colname = f"Median {ci_level_str}"

    # make sure the directories exist
    pathlib.Path(os.path.join(experiment_path, "overview_plots")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(experiment_path, "overview_table")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(experiment_path, "overview_merged")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(experiment_path, "overview_merged_gbl")).mkdir(parents=True, exist_ok=True)

    for metric_name in metric_names:
        if metric_name in df.columns:
            df[metric_name] = df[metric_name].astype(float, copy=True)

    # Make an aggregated overview table with medians with all relevant metrics in the same table
    for name, relevant_metrics in OVERVIEW_TABLE_merged_table_relevant_metric_groups:
        filtered_merged_metric_names = [metric_name for metric_name in relevant_metrics if
                                        metric_name in df.columns]
        if any([True for el in ["Attribution (Avg. Score)", "Main Ideas (Avg. Score)"] if
                el in filtered_merged_metric_names]):
            # remove the base values
            filtered_merged_metric_names.remove("Attribution")
            filtered_merged_metric_names.remove("Main Ideas")
        # agg_funcs_full = [np.median, bootstrap_CI(np.median, confidence_level=confidence_level)] * len(filtered_merged_metric_names)
        # agg_funcs_median = [np.median] * len(filtered_merged_metric_names)
        gblList_List = [groupbyList, [groupbyList[0]], [groupbyList[1]]]
        gblList_Extension = []
        for gbl in gblExtension:
            gblList_Extension.append([gbl[0]])
            gblList_Extension.append([gbl[1]])
            gblList_Extension.append(gbl)
        gblList_List = gblList_List + gblList_Extension
        if "Model" in groupbyList:
            # get other entry
            other_entry = [el for el in groupbyList if el != "Model"][0]
            gblList_List = gblList_List + [[other_entry], [other_entry, "Model Type"],
                                           [other_entry, "Model Type", "Model"], ["Model Type"]]
        # remove duplicate entries in gblList_List
        gblList_List = list(set([tuple(gblList) for gblList in gblList_List]))
        for gblList in gblList_List:
            merged_groupbyList = [gblElement for gblElement in groupbyList]

            col_new_full = [gblElement for gblElement in gblList]
            col_new_mean = [gblElement for gblElement in gblList]
            col_new_median = [gblElement for gblElement in gblList]
            orig_groupbyList = [gblElement for gblElement in gblList]
            for metric_name in filtered_merged_metric_names:
                if metric_name in OVERVIEW_TABLE_Rename_Map:  # rename if necessary
                    metric_name = OVERVIEW_TABLE_Rename_Map[metric_name]

                col_new_full.extend([f"{metric_name}", f"{metric_name} {ci_level_str}"])
                col_new_mean.extend([f"{metric_name}", f"{metric_name} St.Dev."])
                col_new_median.append(f"{metric_name}")
            # Calculate multiple statistics + their CI and save them in a table
            df_merged_full = df.groupby(orig_groupbyList).agg(
                {metric_name: [np.median, bootstrap_CI(np.median, confidence_level=confidence_level)] for metric_name in
                 filtered_merged_metric_names}).reset_index()
            df_merged_mean = df.groupby(orig_groupbyList).agg(
                {metric_name: [np.mean, np.std] for metric_name in filtered_merged_metric_names}).reset_index()
            df_merged_median = df.groupby(orig_groupbyList).agg(
                {metric_name: [np.median] for metric_name in filtered_merged_metric_names}).reset_index()
            df_merged_full.columns = col_new_full
            df_merged_mean.columns = col_new_mean
            df_merged_median.columns = col_new_median
            df_merged_full = df_merged_full.round(3)
            df_merged_mean = df_merged_mean.round(3)
            df_merged_median = df_merged_median.round(3)

            # make a nice table to copy the text directly from:
            df_merged_full_NICE = df_merged_full.copy()
            ci_columns = [col for col in df_merged_full_NICE.columns if ci_level_str in col]
            for metric_name in filtered_merged_metric_names:
                metric_col = OVERVIEW_TABLE_Rename_Map[
                    metric_name] if metric_name in OVERVIEW_TABLE_Rename_Map else metric_name
                metric_ci_col = f"{metric_col} {ci_level_str}"
                round_func = lambda x: round(x, 3) if not math.isnan(x) else "NaN"
                CI_niceify = lambda x: f"({ci_level_str}: {round_func(x[0])}-{round_func(x[1])})" if not math.isnan(
                    x[0]) else "NaN"
                df_merged_full_NICE[metric_col] = df_merged_full_NICE.apply(
                    lambda row: f"{row[metric_col]} {CI_niceify(row[metric_ci_col])}", axis=1)
            # remove the ci columns
            df_merged_full_NICE = df_merged_full_NICE.drop(columns=ci_columns)

            # save as csv and latex table
            gblString = "_".join(gblList)
            df_merged_full.to_csv(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_median_CI_{name}____{gblString}.csv"),
                index=False)
            df_merged_full.to_latex(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_median_CI_{name}____{gblString}.tex"),
                float_format="%.3f", index=False)
            df_merged_median.to_csv(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_median_only_{name}____{gblString}.csv"),
                index=False)
            df_merged_median.to_latex(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_median_only_{name}____{gblString}.tex"),
                float_format="%.3f", index=False)
            df_merged_mean.to_csv(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_mean_only_{name}____{gblString}.csv"),
                index=False)
            df_merged_mean.to_latex(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_mean_only_{name}____{gblString}.tex"),
                float_format="%.3f", index=False)
            df_merged_full_NICE.to_csv(
                os.path.join(experiment_path, "overview_merged",
                             f"merged_overview_table_median_CI_NICE_{name}____{gblString}.csv"),
                index=False)

    # Make similar tables for each metric, but for all passed groupbylists in gblExtension (only medians)
    for gbl in gblExtension:
        for name, relevant_metrics in OVERVIEW_TABLE_merged_table_relevant_metric_groups:
            filtered_merged_metric_names = [metric_name for metric_name in relevant_metrics if
                                            metric_name in df.columns]
            if any([True for el in ["Attribution (Avg. Score)", "Main Ideas (Avg. Score)"] if
                    el in filtered_merged_metric_names]):
                # remove the base values
                filtered_merged_metric_names.remove("Attribution")
                filtered_merged_metric_names.remove("Main Ideas")
            # agg_funcs_full = [np.median, bootstrap_CI(np.median, confidence_level=confidence_level)] * len(filtered_merged_metric_names)
            # agg_funcs_median = [np.median] * len(filtered_merged_metric_names)
            col_new_median = [gblElement for gblElement in gbl]
            col_new_mean = [gblElement for gblElement in gbl]
            orig_groupbyList = [gblElement for gblElement in gbl]
            orig_groupbyList_First = [gbl[0]]
            orig_groupbyList_First_median = [gbl[0]]
            for metric_name in filtered_merged_metric_names:
                if metric_name in OVERVIEW_TABLE_Rename_Map:  # rename if necessary
                    metric_name = OVERVIEW_TABLE_Rename_Map[metric_name]

                col_new_median.append(f"{metric_name}")
                col_new_mean.extend([f"{metric_name}", f"{metric_name} St.Dev."])
                orig_groupbyList_First_median.append(f"{metric_name}")
            # Calculate medians
            df_merged_median = df.groupby(orig_groupbyList).agg(
                {metric_name: [np.median] for metric_name in filtered_merged_metric_names}).reset_index()
            df_merged_mean = df.groupby(orig_groupbyList).agg(
                {metric_name: [np.mean, np.std] for metric_name in filtered_merged_metric_names}).reset_index()
            df_merged_firstGbl_median = df.groupby(orig_groupbyList_First).agg(
                {metric_name: [np.median] for metric_name in filtered_merged_metric_names}).reset_index()
            df_merged_median.columns = col_new_median
            df_merged_mean.columns = col_new_mean
            df_merged_firstGbl_median.columns = orig_groupbyList_First_median
            df_merged_median = df_merged_median.round(3)
            df_merged_mean = df_merged_mean.round(3)
            df_merged_firstGbl_median = df_merged_firstGbl_median.round(3)

            # save as csv and latex table
            df_merged_median.to_csv(
                os.path.join(experiment_path, "overview_merged_gbl",
                             f"merged_overview_table_median_only_{name}__{gbl[0]}_{gbl[1]}.csv"),
                index=False)
            df_merged_median.to_latex(
                os.path.join(experiment_path, "overview_merged_gbl",
                             f"merged_overview_table_median_only_{name}__{gbl[0]}_{gbl[1]}.tex"),
                float_format="%.3f", index=False)
            df_merged_mean.to_csv(
                os.path.join(experiment_path, "overview_merged_gbl",
                             f"merged_overview_table_mean_only_{name}__{gbl[0]}_{gbl[1]}.csv"),
                index=False)
            df_merged_mean.to_latex(
                os.path.join(experiment_path, "overview_merged_gbl",
                             f"merged_overview_table_mean_only_{name}__{gbl[0]}_{gbl[1]}.tex"),
                float_format="%.3f", index=False)
            df_merged_firstGbl_median.to_csv(
                os.path.join(experiment_path, "overview_merged_gbl",
                             f"merged_overview_table_median_only_{name}__{gbl[0]}.csv"),
                index=False)
            df_merged_firstGbl_median.to_latex(
                os.path.join(experiment_path, "overview_merged_gbl",
                             f"merged_overview_table_median_only_{name}__{gbl[0]}.tex"),
                float_format="%.3f", index=False)

    # make a table showing the median of each metric (one table per metric), grouped by groupbyList
    for metric_name in metric_names:
        if metric_name not in df.columns:  # for the SEAHORSE MDS columns
            continue

        # Calculate multiple statistics + their CI and save them in a table
        df[metric_name] = df[metric_name].astype(float, copy=True)
        df_metric_FULL = df.groupby(groupbyList).agg(
            {metric_name: [
                np.mean, bootstrap_CI(np.mean, confidence_level=confidence_level),
                np.std, bootstrap_CI(np.std, confidence_level=confidence_level),
                np.var, bootstrap_CI(np.var, confidence_level=confidence_level),
                percentile_agg(5), bootstrap_CI(percentile_agg(5), confidence_level=confidence_level),
                percentile_agg(25), bootstrap_CI(percentile_agg(25), confidence_level=confidence_level),
                np.median, bootstrap_CI(np.median, confidence_level=confidence_level),
                percentile_agg(75), bootstrap_CI(percentile_agg(75), confidence_level=confidence_level),
                percentile_agg(95), bootstrap_CI(percentile_agg(95), confidence_level=confidence_level),
            ]}).reset_index()
        col_new = groupbyList + [
            'Mean', f"Mean {ci_level_str}",
            'St.Dev.', f"St.Dev. {ci_level_str}",
            'Var.', f"Var. {ci_level_str}",
            '5th Perc.', f"5th Perc. {ci_level_str}",
            '25th Perc.', f"5th Perc. {ci_level_str}",
            'Median', f"Median {ci_level_str}",
            '75th Perc.', f"75th Perc. {ci_level_str}",
            '95th Perc.', f"95th Perc. {ci_level_str}",
        ]
        df_metric_FULL.columns = col_new
        df_metric_FULL = df_metric_FULL.round(ROUND_TO_DECIMALS)
        out_overview.append({
            "name": f"Full Statisics Table {metric_name}",
            "df": df_metric_FULL
        })
        # save as csv and latex table
        df_metric_FULL.to_csv(
            os.path.join(experiment_path, "overview_table", f"overview_table_full_{metric_name}.csv"),
            index=False)
        df_metric_FULL.to_latex(
            os.path.join(experiment_path, "overview_table", f"overview_table_full_{metric_name}.tex"),
            index=False)

        # Calculate the median of the metric + CI
        df[metric_name] = df[metric_name].astype(float, copy=True)
        df_metric = df.groupby(groupbyList).agg(
            {metric_name: ["median", bootstrap_CI(np.median, confidence_level=confidence_level)]}).reset_index()
        col_new = groupbyList + ['Median', ci_colname]
        df_metric.columns = col_new
        df_metric = df_metric.round(ROUND_TO_DECIMALS)
        out_overview.append({
            "name": f"median {metric_name}",
            "df": df_metric
        })
        # save as csv and latex table
        df_metric.to_csv(
            os.path.join(experiment_path, "overview_table", f"overview_table_median {metric_name}.csv"),
            index=False)
        df_metric.to_latex(
            os.path.join(experiment_path, "overview_table", f"overview_table_median {metric_name}.tex"),
            index=False)

        # prepare the data for the overview-table plot showing the median + confidence interval
        overview_x_1 = df_metric[groupbyList[0]]
        overview_x_2 = df_metric[groupbyList[1]]
        overview_x = [f"{x[0]}" + " / " + f"{x[1]}" for x in zip(overview_x_1, overview_x_2)]
        overview_y = df_metric["Median"]
        overview_yerr = df_metric[ci_colname]
        # reshape overview_yerr to (2,N) array
        overview_yerr = np.array(overview_yerr.tolist()).T
        # subtract y from both yerr values to get the lower and upper error bars, making sure that the lower error bar is always positive
        overview_yerr[0] = overview_y - overview_yerr[0]
        overview_yerr[1] = overview_yerr[1] - overview_y
        # make the plot
        plt.errorbar(overview_x, overview_y, yerr=overview_yerr, fmt='o', )
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.4)
        save_path = os.path.join(experiment_path, "overview_plots",
                                 f"overview_{groupbyList[0]}_{groupbyList[1]}_median_plot_{metric_name}.pdf")
        plt.savefig(save_path)
        plt.close()

        # make a second plot where seaborn calculates the errorbars
        for (a, b) in [(0, 1), (1, 0)]:
            hue_order = get_sorted_labels(df, groupbyList[b])
            x_order = get_sorted_labels(df, groupbyList[a])
            sns.pointplot(data=df, x=groupbyList[a], y=metric_name, hue=groupbyList[b], hue_order=hue_order,
                          order=x_order,
                          estimator=np.median, errorbar=('ci', 95), n_boot=1000, dodge=True,
                          linestyles='')  # , err_kws={"markersize": 10, "capsize": 0.1, "errwidth": 1.5, "palette": "colorblind"}
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
            plt.xticks(rotation=90)
            plt.tight_layout()
            save_path = os.path.join(experiment_path, "overview_plots",
                                     f"overview_seaborn_{groupbyList[a]}_{groupbyList[b]}_median_plot_{metric_name}.pdf")
            plt.savefig(save_path)
            plt.close()

        # TODO: Make a plot showing this overview table -> showing median performance + confidence interval for each model and promptVersion

    # loop over models (looking at them separately), and over the metrics (looking at them separately)
    # each table showing the median, 10th percentile, 90th percentile, and the stderr for each metric
    for model_name in df["Model"].unique():
        df_model = df[df["Model"] == model_name]
        for metric_name in metric_names:
            if metric_name not in df.columns:  # for the SEAHORSE MDS columns
                continue
            df_metric = df_model.groupby(groupbyList).agg({metric_name: agg_funcs}).reset_index()

            col_new = groupbyList + agg_names
            df_metric.columns = col_new
            df_metric = df_metric.round(ROUND_TO_DECIMALS)
            out_detail.append({
                "name": f"{model_name} {metric_name}",
                "df": df_metric
            })
    # and doing the same but looking first at the prompts individually, and then at the metrics within a given prompt
    for promptVersion in df["Prompt ID"].unique():
        df_prompt = df[df["Prompt ID"] == promptVersion]
        for metric_name in metric_names:
            if metric_name not in df.columns:  # for the SEAHORSE MDS columns
                continue
            df_metric = df_prompt.groupby(groupbyList).agg({metric_name: agg_funcs}).reset_index()

            col_new = groupbyList + agg_names
            df_metric.columns = col_new
            df_metric = df_metric.round(ROUND_TO_DECIMALS)
            out_detail.append({
                "name": f"{promptVersion} {metric_name}",
                "df": df_metric
            })
    # save the detail tables (as csv and tex table)
    for table in out_detail:
        table["df"].to_csv(
            os.path.join(experiment_path, "detail_table", f"detail_table_{table['name']}.csv"),
            index=False)
        # also export as latex table
        table["df"].to_latex(
            os.path.join(experiment_path, "detail_table", f"detail_table_{table['name']}.tex"),
            index=False)

    return out_overview, out_detail, agg_names


def statistical_tests(df, metric_names, comparison_columns=['Model', 'Prompt ID'], prompt_category_group="Model",
                      promptVersionCol="Prompt ID"):
    if df.shape[0] == 0:
        return None, None
    # Output a json file with all the statistical test outputs
    # Loop over all comparison columns, for each column, compare the individual values of each metric
    # additionally calculate columns based on the prompt version and use them as well

    df_stat = df.copy()

    """
    Prompt Categories:
    - German/English (Language of the prompt)
    - Short/Long (Length of the prompt)
    - TargetLang (whether it is mentioned in the prompt)
    - Length (Whether the length of the output is mentioned or not)
    - Domain (Whether the domain is mentioned or not)
    - Style (Whether the output style is mentioned or not)
    - Personification
    - Simplification
    """
    analyzeCategories = ["German", "English", "TargetLang", "Style", "Personification"]
    promptCategories = {
        "1": ["English", "Long", "TargetLang", "Length"],
        "2": ["German", "Long", "TargetLang", "Length"],
        "3": ["English", "Long", "TargetLang", "Length", "Domain", "Style"],
        "4": ["German", "Long", "TargetLang", "Length", "Domain", "Style"],
        "5": ["Short"],
        "6": ["English", "Short"],
        "7": ["German", "Short"],
        "8": ["English", "Short", "Domain"],
        "9": ["German", "Long", "Domain"],
        "10": ["English", "Long", "Length", "Simplification"],
        "11": ["German", "Long", "Length", "Simplification"],
        "12": ["English", "Long", "Length", "Domain", "Style", "Personification"],
        "13": ["German", "Long", "Length", "Domain", "Style", "Personification"],
        "14": ["English", "Long", "Length", "Domain", "Personification"],
        "15": ["German", "Long", "Length", "Domain", "Personification"],
        "16": ["English", "Long", "Domain", "Simplification"],
        "17": ["German", "Long", "Domain", "Simplification"],
        "18": ["English", "Long"],
        "19": ["German", "Long"],
        "20": ["English", "Long", "TargetLang"],
    }
    # Build the inverted index mapping the categories to a list of promptVersions
    promptCategoriesInv = {}
    for promptVersion in promptCategories:
        for category in promptCategories[promptVersion]:
            if category not in promptCategoriesInv:
                promptCategoriesInv[category] = []
            promptCategoriesInv[category].append(promptVersion)
    # Add the categories to the dataframe based on the promptVersion column
    for promptCat in analyzeCategories:
        if promptCat not in promptCategoriesInv:
            raise ValueError(f"Prompt Category {promptCat} not found in promptCategoriesInv")
        df_stat[promptCat] = df_stat["Prompt ID"].apply(lambda x: 1 if x in promptCategoriesInv[promptCat] else 0)

    results = {lbl: {} for lbl in comparison_columns + ['promptCategories']}

    # Calculate the tests for the label categories
    for group_label in df_stat[prompt_category_group].unique():
        df_group = df_stat[df_stat[prompt_category_group] == group_label]
        results['promptCategories'][group_label] = {}
        for metric_name in metric_names:
            results['promptCategories'][group_label][metric_name] = {}
            pValues = {}
            # make 1 vs all comparisons
            for promptCat in analyzeCategories:
                # get promptVersions for the current category (in-group), and the (out-group) promptVersions not in the current category
                promptVersions_in = promptCategoriesInv[promptCat]
                promptVersions_out = [promptVersion for promptVersion in df_group[promptVersionCol] if
                                      promptVersion not in promptVersions_in]

                in_group = df_group[df_group[promptVersionCol].isin(promptVersions_in)]
                out_group = df_group[df_group[promptVersionCol].isin(promptVersions_out)]

                # make dependent t-test and Wilcoxon matched-pairs test for each metric
                ttest_res = stats.ttest_rel(in_group[metric_name], out_group[metric_name])

                # TODO: Wilcoxon -> compare one-vs-all and take the mean of all paired samples?
                # TODO: Wilcoxon -> compare all pairs (1vs1)?
                # wilcoxon_res = stats.wilcoxon(in_group[metric_name], out_group[metric_name])

                # save the p-values
                pValues[promptCat] = {
                    "ttest": ttest_res.pvalue,
                    # "wilcoxon": wilcoxon_res.pvalue
                }
            # save the p-values (for each metric), both raw and adjusted (with bonferroni correction)
            results['promptCategories'][group_label][metric_name]["pValues"] = pValues
            pValuesList = [pValues[promptCat]["ttest"] for promptCat in pValues]
            pValuesAdj = stats.false_discovery_control(pValuesList)
            results['promptCategories'][group_label][metric_name]["pValuesAdjusted"] = {}
            for i, promptCat in enumerate(pValues):
                results['promptCategories'][group_label][metric_name]["pValuesAdjusted"][promptCat] = pValuesAdj[i]

    # loop over the comparison_columns

    # TODO: CONTINUE HERE

    # Kruskal Wallis Test -> test if the distributions of the metrics are the same across the groups
    # scipy.stats.kruskal
    # parametric -> one-way repeated measures ANOVA

    # pair-wise Wilcoxon Rank Sum Test -> test if the distributions of the metrics are the same between each pair of groups
    # pair-wise Wolcoxon matched-pairs test
    # scipy.stats.wilcoxon
    # parametric -> dependent t-test

    pass


sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
somajo_tokenizer = SoMaJo("de_CMC", split_camel_case=True, split_sentences=True)


def sentence_similarity_matcher(reference, target, count=2, ref_name="Reference", tgt_name="Target") -> str:
    """
    Returns a html fragment (interactive)
    """

    def _sub(match):
        return '\n\n ' + match.group(0) + " "

    # Help the tokenizer with some special patterns
    patterns = [
        (r"\*", "\n\n "),
        (r"•", "\n\n "),
        (r"•", "\n\n "),
        (r"\d\.", _sub),
        (r"\d\)", _sub),
    ]
    for pattern in patterns:
        if re.search(pattern[0], reference):
            reference = re.sub(pattern[0], pattern[1], reference)
        if re.search(pattern[0], target):
            target = re.sub(pattern[0], pattern[1], target)

    # Tokenize, embed, group

    embedding_model = lambda list: sbert_model.encode(list)

    ref_sentences = list(somajo_tokenizer.tokenize_text([reference]))
    ref_sentences_text = [" ".join([el.text for el in x]) for x in ref_sentences]
    tgt_sentences = list(somajo_tokenizer.tokenize_text([target]))
    tgt_sentences_text = [" ".join([el.text for el in x]) for x in tgt_sentences]

    ref_sentence_embeddings = embedding_model(ref_sentences_text)
    tgt_sentence_embeddings = embedding_model(tgt_sentences_text)

    similarity_matrix = util.cos_sim(tgt_sentence_embeddings, ref_sentence_embeddings)

    # get the indices of the most similar sentences
    indices = np.argsort(similarity_matrix, axis=1)
    indices = indices[:, -count:]

    # generate a list with the sentences and the indices in the other group (
    reference_list = []
    target_list = []
    for i, index in enumerate(tgt_sentences):
        target_list.append({
            "sentence": tgt_sentences_text[i],
            "similar": indices[i].tolist(),
            "index": i}
        )
    for i, index in enumerate(ref_sentences):
        similar_indices = []
        for j, index in enumerate(indices):
            if i in index:
                similar_indices.append(j)
        reference_list.append({
            "sentence": ref_sentences_text[i],
            "similar": similar_indices,
            "index": i,
        })

    # generate the html fragment
    ref_sentence_fragment = ""
    for el in reference_list:
        similar_classes = " ".join(["similar-" + str(x) for x in el["similar"]])
        similar_classes += " " + "match" if len(el["similar"]) > 0 else ""
        ref_sentence_fragment += f'<div class="sentence-similarity__reference__sentence {similar_classes}" data-index="{el["index"]}">{el["sentence"]}</div>\n'
    tgt_sentence_fragment = ""
    for el in target_list:
        similar_classes = " ".join(["similar-" + str(x) for x in el["similar"]])
        similar_classes += " " + "match" if len(el["similar"]) > 0 else ""
        tgt_sentence_fragment += f'<div class="sentence-similarity__target__sentence {similar_classes}" data-index="{el["index"]}">{el["sentence"]}</div>\n'

    similarity_fragment = f"""
    <div class="sentence-similarity">
        <div class="sentence-similarity__reference">
            <h3>{ref_name}</h3>
            <div class="sentence-similarity__reference__sentences">
                {ref_sentence_fragment}
            </div>
        </div>
        <div class="sentence-similarity__target">
            <h3>{tgt_name}</h3>
            <div class="sentence-similarity__target__sentences">
                {tgt_sentence_fragment}
            </div>
        </div>
    </div>
    """

    return similarity_fragment


def save_df_to_html_simple(df, html_path, groupbyList, newline_columns=['Prompt', 'Prediction', 'GT-Summary']):
    # prepare the html output
    df = df.transpose()
    # replace newlines with <br> in the specified columns
    df = df.apply(lambda row: df_row_replace(row, newline_columns, '\n', '<br>'))
    # but remove exxessive repetitions of newlines in the same specified columns
    df = df.apply(lambda row: df_row_apply_func(row, newline_columns, lambda x: re.sub(r'\n{4,}', '\n\n\n', x)))
    df = df.transpose()

    # sort the dataframe by the groupbyList
    df = df.sort_values(by=groupbyList)
    dft = df.transpose()
    # save
    html_data = dft.to_html(escape=False).replace("\\n", "").replace("\n", "")
    html_page = html_base.format(table=html_data)
    with open(html_path, "w") as f:
        f.write(html_page)


def save_inspect_examples_to_html(df, html_path, extendedGroupByList, applySimilarityMatcher=True,
                                  newline_columns=['Prompt', 'Prediction', 'GT-Summary'],
                                  similarity_matcher=[
                                      {'ref': "Prompt", 'tgt': "Prediction", 'col': "source-similarity"},
                                      {'ref': "Prediction", 'tgt': "GT-Summary", 'col': "truth-similarity"},
                                      {'ref': "Prompt", 'tgt': "GT-Summary", 'col': "ground-truth-similarity"}]):
    # add similarity matching columns
    if applySimilarityMatcher:
        for matcher in similarity_matcher:
            df.loc[:, matcher['col']] = df.apply(
                lambda row: sentence_similarity_matcher(row[matcher['ref']], row[matcher['tgt']],
                                                        ref_name=matcher['ref'], tgt_name=matcher['tgt']), axis=1)

    # prepare the html output
    df = df.transpose()
    df = df.apply(lambda row: df_row_replace(row, newline_columns, '\n', '<br>'))
    df = df.transpose()
    curr_doc_id = df['doc_id'].unique()[0]

    # sort the dataframe by the groupbyList
    groupbyList = extendedGroupByList[0]
    df = df.sort_values(by=groupbyList)
    dft = df.transpose()
    # save
    html_file_path = os.path.join(html_path, f"{curr_doc_id}.html")
    html_data = dft.to_html(escape=False).replace("\\n", "").replace("\n", "")
    html_page = html_base.format(table=html_data)
    with open(html_file_path, "w") as f:
        f.write(html_page)

    # filter by groupbyList[a] entry values and sort by other dimension
    html_example_subpath = os.path.join(html_path, f"{curr_doc_id}")
    pathlib.Path(html_example_subpath).mkdir(parents=True, exist_ok=True)
    for groupbyList in extendedGroupByList:
        assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"
        for (a, b) in [(0, 1), (1, 0)]:
            for group_label in df[groupbyList[a]].unique():
                df_group = df[df[groupbyList[a]] == group_label]
                df_group = df_group.sort_values(by=groupbyList[b])
                df_group = df_group.transpose()
                html_file_path = os.path.join(html_example_subpath,
                                              f"{curr_doc_id}_{groupbyList[a]}_{group_label}.html")

                html_data = df_group.to_html(escape=False).replace("\\n", "").replace("\n", "")
                html_page = html_base.format(table=html_data)
                with open(html_file_path, "w") as f:
                    f.write(html_page)


def df_row_replace(row, columns, replace, with_string):
    # Note: assumes columns are strings (values)
    for col in columns:
        # if element is float.nan -> skip
        if isinstance(row[col], float) and np.isnan(row[col]):
            continue
        row[col] = row[col].replace(replace, with_string)
    return row


def df_row_apply_func(row, columns, f):
    # Note: assumes columns are strings (values)
    for col in columns:
        # if element is float.nan -> skip
        if isinstance(row[col], float) and np.isnan(row[col]):
            continue
        row[col] = f(row[col])
    return row


html_base = """<!DOCTYPE html>
<meta http-equiv="content-type" content="text/html; charset=utf-8">
<html>
<head>
    <style>
        td {{
            text-align: left;
            vertical-align: top;
            min-width: 125rem;
        }}
        .sentence-similarity {{
            display: flex;
            flex-direction: row;
            justify-content: space-between;
        }}
        .sentence-similarity__reference,
        .sentence-similarity__target {{
            width: 50%;
        }}
        .sentence-similarity__reference__sentences,
        .sentence-similarity__target__sentences {{
            display: flex;
            flex-direction: column;
        }}
        .sentence-similarity__reference__sentence,
        .sentence-similarity__target__sentence {{
            padding: 0.15rem;
            margin: 0.2rem 0;
        }}
        .sentence-similarity__reference__sentence.highlight,
        .sentence-similarity__target__sentence.highlight {{
            background-color: #f7c6ff !important;
            border: 0.25px solid red;
        }}
        .sentence-similarity__reference__sentence.match,
        .sentence-similarity__target__sentence.match {{
            background-color: #bdfff6;
        }}
    </style>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
    <script>
        $(document).ready(function () {{
            $('.sentence-similarity__target__sentence').hover(function () {{
                // add highlight class to element hovering on and all elements with data-index equal to index in similar-{{idx}} classes of element hovering on
                $(this).addClass('highlight');
                var similarIndices = $(this).attr('class').match(/similar-\d+/g);
                var referenceSentence = $(this).closest('.sentence-similarity').find('.sentence-similarity__reference__sentence');
                referenceSentence = referenceSentence.filter(function () {{
                    var idx = $(this).data('index');
                    return similarIndices.includes('similar-' + idx);
                }});
                referenceSentence.addClass('highlight');
            }}, function () {{
                // remove highlight class from element hovering on and all elements with data-index equal to index in similar-{{idx}} classes of element hovering on
                $(this).removeClass('highlight');
                var similarIndices = $(this).attr('class').match(/similar-\d+/g);
                var referenceSentence = $(this).closest('.sentence-similarity').find('.sentence-similarity__reference__sentence');
                referenceSentence = referenceSentence.filter(function () {{
                    var idx = $(this).data('index');
                    return similarIndices.includes('similar-' + idx);
                }});
                referenceSentence.removeClass('highlight');
            }});

            $('.sentence-similarity__reference__sentence').hover(function () {{
                // add highlight class to element hovering on and all elements with data-index equal to index in similar-{{idx}} classes of element hovering on
                $(this).addClass('highlight');
                var similarIndices = $(this).attr('class').match(/similar-\d+/g);
                var targetSentence = $(this).closest('.sentence-similarity').find('.sentence-similarity__target__sentence');
                targetSentence = targetSentence.filter(function () {{
                    var idx = $(this).data('index');
                    return similarIndices.includes('similar-' + idx);
                }});
                targetSentence.addClass('highlight');
            }}, function () {{
                // remove highlight class from element hovering on and all elements with data-index equal to index in similar-{{idx}} classes of element hovering on
                $(this).removeClass('highlight');
                var similarIndices = $(this).attr('class').match(/similar-\d+/g);
                var targetSentence = $(this).closest('.sentence-similarity').find('.sentence-similarity__target__sentence');
                targetSentence = targetSentence.filter(function () {{
                    var idx = $(this).data('index');
                    return similarIndices.includes('similar-' + idx);
                }});
                targetSentence.removeClass('highlight');
            }});
        }});
    </script>
</head>
<body>

{table}

</body>
</html>"""


def save_inspect_examples_simple(df, experiment_path, inspect_name, extendedGroupByList=[], suffix=""):
    base_folder = f"inspect-examples{suffix}"

    # remove old html files
    html_path = os.path.join(experiment_path, base_folder, inspect_name)
    if os.path.exists(html_path):
        shutil.rmtree(html_path)
    pathlib.Path(html_path).mkdir(parents=True, exist_ok=True)

    # save examples
    doc_ids = df['doc_id'].unique()

    for doc_id in doc_ids:
        df_doc = df[df['doc_id'] == doc_id]
        save_inspect_examples_to_html(df_doc, html_path, extendedGroupByList=extendedGroupByList,
                                      newline_columns=['Prompt', 'Prediction', 'GT-Summary'],
                                      applySimilarityMatcher=False)


def find_inspect_examples(df, experiment_path, metric_names, groupbyList=["Model", "Prompt ID"], extendedGroupByList=[],
                          numExamples=2, suffix="", inspect_metrics=["BertScore F1"]):
    if df.shape[0] == 0:
        return {}
    print("Finding examples to inspect...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"
    # percentile ranges to sample from
    sample_ranges = {
        "worst": [0, 5],
        "median": [47.5, 52.5],
        "best": [95, 100],
        "all": [0, 100]
    }
    num_random_inspect_examples = 5
    max_num_interesting_doc_ids = 10

    # delete old folder if it exists
    base_folder = f"inspect-examples{suffix}"
    for metric_name in inspect_metrics:
        metric_subpath = os.path.join(experiment_path, base_folder, metric_name)
        if os.path.exists(metric_subpath):
            shutil.rmtree(metric_subpath)
        pathlib.Path(metric_subpath).mkdir(parents=True, exist_ok=True)

    # make subfolders to store specific examples
    # pathlib.Path(os.path.join(experiment_path, base_folder)).mkdir(parents=True, exist_ok=True)

    # Initialize
    def numExamples(category):
        if category == "all":
            return 5
        else:
            return 3

    out = {}
    for groupA in df[groupbyList[0]].unique():
        out[f"{groupA}"] = {}
        out[f"{groupA}"]["general"] = {
            f"{metric_name}": {
                "best": [],
                "worst": [],
                "median": [],
                "all": []
            } for metric_name in metric_names
        }
        for groupB in df[groupbyList[1]].unique():
            out[f"{groupA}"][f"{groupB}"] = {
                f"{metric_name}": {
                    "best": [],
                    "worst": [],
                    "median": [],
                    "all": []
                } for metric_name in metric_names
            }

    # fill the dict with examples
    for groupA in df[groupbyList[0]].unique():
        # get the df for the current model
        df_groupA = df[df[groupbyList[0]] == groupA]
        # if empty, skip
        if df_groupA.shape[0] == 0:
            continue
        # in general, compute the percentiles for each metric and sample the examples
        for metric_name in metric_names:
            for cat in out[f"{groupA}"]["general"][metric_name]:
                # calculate the percentiles
                percentile_range = sample_ranges[cat]
                percentile_values = np.percentile(df_groupA[metric_name], percentile_range)
                # sample the examples
                sample_population = df_groupA[
                    (df_groupA[metric_name] >= percentile_values[0]) & (df_groupA[metric_name] <= percentile_values[1])]
                df_sample = sample_population.sample(min(sample_population.shape[0], numExamples(cat)))
                # add the examples to the dict
                out[f"{groupA}"]["general"][metric_name][cat] = df_sample.to_dict(orient="records")
        # for each prompt, compute the percentiles for each metric and sample the examples
        for groupB in df[groupbyList[1]].unique():
            # get the df for the current prompt
            df_groupB = df_groupA[df_groupA[groupbyList[1]] == groupB]
            # if empty, skip
            if df_groupB.shape[0] == 0:
                continue
            for metric_name in metric_names:
                for cat in out[f"{groupA}"][f"{groupB}"][metric_name]:
                    # calculate the percentiles
                    percentile_range = sample_ranges[cat]
                    percentile_values = np.percentile(df_groupB[metric_name], percentile_range)
                    # sample the examples
                    sample_population = df_groupB[(df_groupB[metric_name] >= percentile_values[0]) & (
                            df_groupB[metric_name] <= percentile_values[1])]
                    df_sample = sample_population.sample(min(sample_population.shape[0], numExamples(cat)))
                    # add the examples to the dict
                    out[f"{groupA}"][f"{groupB}"][metric_name][cat] = df_sample.to_dict(orient="records")

    # Gather the examples for the inspect metrics, and make side-by-side comparisons to the other combinations in the data
    # ... also sample 10 random examples and make a side-by-side comparison to the other combinations in the data
    inspect_exclude_columns = ['Prompt', 'Prediction', "GT-Summary"]
    all_other_cols = [col for col in df.columns if col not in inspect_exclude_columns]
    for inspect_metric in inspect_metrics:
        interesting_docIds = {cat: [] for cat in sample_ranges if cat != "all"}
        for groupA in out:
            for groupB in out[groupA]:
                for cat in out[groupA][groupB][inspect_metric]:
                    if cat not in interesting_docIds:
                        continue
                    interesting_docIds[cat].extend([el["doc_id"] for el in out[groupA][groupB][inspect_metric][cat]])

        # exclude duplicate docIDs
        for cat in interesting_docIds:
            interesting_docIds[cat] = list(set(interesting_docIds[cat]))

        # make sure all the folders exist
        pathlib.Path(os.path.join(experiment_path, base_folder, inspect_metric)).mkdir(parents=True, exist_ok=True)
        for cat in interesting_docIds:
            pathlib.Path(os.path.join(experiment_path, base_folder, inspect_metric, cat)).mkdir(parents=True,
                                                                                                exist_ok=True)

        # for each document ID, get all the predictions for that document and save them in a html file
        for cat in interesting_docIds:
            # if more than max_num_interesting_doc_ids, sample a random subset
            if len(interesting_docIds[cat]) > max_num_interesting_doc_ids:
                interesting_docIds[cat] = list(
                    np.random.choice(interesting_docIds[cat], size=max_num_interesting_doc_ids, replace=False))
            for docID in interesting_docIds[cat]:
                doc_id_gt_summaries = df[df["doc_id"] == docID]["GT-Summary"].unique().tolist()
                # loop over all the gt-summaries, make 1 html file per gt-summary
                for summ_idx, gt_summary in enumerate(doc_id_gt_summaries):
                    df_docID = df[df['GT-Summary'].apply(lambda x: x.strip()) == gt_summary.strip()]

                    # save to html
                    html_path = os.path.join(experiment_path, base_folder, inspect_metric, cat)
                    save_inspect_examples_to_html(df_docID, html_path, extendedGroupByList=extendedGroupByList,
                                                  newline_columns=['Prompt', 'Prediction', 'GT-Summary'])

    def inspect_docIDs(docIDs, dst_folder, applySimilarityMatcher=True):
        pathlib.Path(os.path.join(experiment_path, base_folder, inspect_metric, dst_folder)).mkdir(parents=True,
                                                                                                   exist_ok=True)
        for docID in docIDs:
            doc_id_gt_summaries = df[df["doc_id"] == docID]["GT-Summary"].unique().tolist()
            # loop over all the gt-summaries, make 1 html file per gt-summary
            for summ_idx, gt_summary in enumerate(doc_id_gt_summaries):
                df_docID = df[df['GT-Summary'].apply(lambda x: x.strip()) == gt_summary.strip()]

                html_path = os.path.join(experiment_path, base_folder, inspect_metric, dst_folder)
                save_inspect_examples_to_html(df_docID, html_path, applySimilarityMatcher=applySimilarityMatcher,
                                              extendedGroupByList=extendedGroupByList,
                                              newline_columns=['Prompt', 'Prediction', 'GT-Summary'])

    def df_get_uniqueDocIDs(df, n):
        uniqueDocIDs = df["doc_id"].unique()
        random_docIDs = list(np.random.choice(uniqueDocIDs, size=min(len(uniqueDocIDs), n), replace=False))
        return random_docIDs

    # Sample some
    print("Sampling random, long, short examples ...")
    df_num_sample = 50
    # Long Inputs
    df_long_inputs = df.sample(n=df_num_sample, replace=False, weights=df["#Input Article Tokens"] ** 2)
    long_inputs_IDs = df_get_uniqueDocIDs(df_long_inputs, num_random_inspect_examples)
    inspect_docIDs(long_inputs_IDs, "long_inputs", applySimilarityMatcher=False)
    # Short Inputs
    df_short_inputs = df.sample(n=df_num_sample, replace=False, weights=(1 / df["#Input Article Tokens"]) ** 2)
    short_inputs_IDs = df_get_uniqueDocIDs(df_short_inputs, num_random_inspect_examples)
    inspect_docIDs(short_inputs_IDs, "short_inputs", applySimilarityMatcher=False)
    # Long GT-Summaries
    df_long_gt_summaries = df.sample(n=df_num_sample, replace=False, weights=df["#GT-Summary Tokens"] ** 2)
    long_gt_summaries_IDs = df_get_uniqueDocIDs(df_long_gt_summaries, num_random_inspect_examples)
    inspect_docIDs(long_gt_summaries_IDs, "long_gt_summaries", applySimilarityMatcher=False)
    # Short GT-Summaries
    df_short_gt_summaries = df.sample(n=df_num_sample, replace=False, weights=(1 / df["#GT-Summary Tokens"]) ** 2)
    short_gt_summaries_IDs = df_get_uniqueDocIDs(df_short_gt_summaries, num_random_inspect_examples)
    inspect_docIDs(short_gt_summaries_IDs, "short_gt_summaries", applySimilarityMatcher=False)
    # Output much longer than GT-Summary Length
    df_longer_output = df.sample(n=df_num_sample, replace=False,
                                 weights=(df["#Predicted Tokens"] / df["#GT-Summary Tokens"]) ** 2)
    longer_output_IDs = df_get_uniqueDocIDs(df_longer_output, num_random_inspect_examples)
    inspect_docIDs(longer_output_IDs, "longer_output_vs_gt", applySimilarityMatcher=False)
    # Output much shorter than GT-Summary Length
    df_shorter_output = df.sample(n=df_num_sample, replace=False,
                                  weights=(df["#GT-Summary Tokens"] / df["#Predicted Tokens"]) ** 2)
    shorter_output_IDs = df_get_uniqueDocIDs(df_shorter_output, num_random_inspect_examples)
    inspect_docIDs(shorter_output_IDs, "shorter_output_vs_gt", applySimilarityMatcher=False)
    # High Compression
    df_high_compression = df.sample(n=df_num_sample, replace=False, weights=df["Compression"] ** 2)
    high_compression_IDs = df_get_uniqueDocIDs(df_high_compression, num_random_inspect_examples)
    inspect_docIDs(high_compression_IDs, "high_compression", applySimilarityMatcher=False)
    # Low-Compression
    df_low_compression = df.sample(n=df_num_sample, replace=False, weights=(1 / df["Compression"]) ** 2)
    low_compression_IDs = df_get_uniqueDocIDs(df_low_compression, num_random_inspect_examples)
    inspect_docIDs(low_compression_IDs, "low_compression", applySimilarityMatcher=False)

    # also sample random document IDs and save them in a html file
    random_docIDs = df_get_uniqueDocIDs(df, num_random_inspect_examples) + [1, 85]
    inspect_docIDs(random_docIDs, "random", applySimilarityMatcher=False)

    return out


def custom_sort_specialcase(x):
    # If "NZZ, DE", "NZZ, EN", and "NZZ-S, DE" are in the list, sort them "NZZ, DE", "NZZ-S, DE", "NZZ, EN"
    if x == "NZZ, DE":
        return "NZZ-0 - NZZ, DE"
    elif x == "NZZ-S, DE":
        return "NZZ-1 - NZZ-S, DE"
    elif x == "NZZ, EN":
        return "NZZ-3 - NZZ, EN"
    else:
        return x


def label_sort_key_func(x, prioritize_string_order=False):
    # if there is a number in the label, sort by that number first, then the string order
    if isinstance(x, int):
        return (x, "")
    elif isinstance(x, float):
        return (x, "")
    elif isinstance(x, str) and prioritize_string_order:
        return (0, custom_sort_specialcase(x))

    match = re.search(r'\d+', x)
    if match:
        return (int(match.group()), x)
    else:
        return (0, x)


def make_special_catplots(df, save_base_path, metric_name, special_catplot_configs, savefig_suffix_kwargs={},
                          data_file_suffix="", fig_size_name="default", file_suffix="pdf"):
    sns.set(rc={'figure.figsize': (12, 8)})

    if len(special_catplot_configs) > 0:
        for catplot_config in special_catplot_configs:
            # check all the required keys are present
            required_keys = ["row", "col", "x", "hue", "is_baseline_row"]
            for required_key in required_keys:
                assert required_key in catplot_config, f"special_metric_catplots must contain the key '{required_key}'"

            # Note: "col" is allowed to be empty
            empty_col = False
            if catplot_config["col"] == "":
                empty_col = True

            # NOTE: the method assumes only one experiment belongs to the baseline method
            # ... unsure what happens if multiple baselines are being passed
            # make sure the baseline method is contained in each of the row and col categories
            df_special = df.copy()
            df_special_baseline_only = df_special[df_special.apply(catplot_config["is_baseline_row"], axis=1)]

            # Make the plot manually using subplots (to not blow up memory due to copying of baseline rows)
            fct_row_order_special = get_sorted_labels(df_special, catplot_config["row"])
            if not empty_col:
                fct_col_order_special = get_sorted_labels(df_special, catplot_config["col"])
            hue_labels_global = get_sorted_labels(df_special, catplot_config["hue"])
            x_order_global = get_sorted_labels(df_special, catplot_config["x"])

            # remove the baselines to only have the manually inserted entries
            df_special = df_special[~df_special.apply(catplot_config["is_baseline_row"], axis=1)]

            subfigure_width = 12
            if len(hue_labels_global) > 6:
                subfigure_width = 16
            n_rows = len(fct_row_order_special)
            if empty_col:
                n_cols = 1
            else:
                n_cols = len(fct_col_order_special)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subfigure_width, n_rows * 8), sharex=True,
                                     sharey=True)

            # make the actual plots
            for row_idx, row in enumerate(fct_row_order_special):
                if empty_col:
                    fct_col_order_special = [""]  # make sure the loop is executed once
                for col_idx, col in enumerate(fct_col_order_special):
                    if empty_col:
                        df_special_row_col = df_special[(df_special[catplot_config["row"]] == row)]
                    else:
                        df_special_row_col = df_special[
                            (df_special[catplot_config["row"]] == row) & (df_special[catplot_config["col"]] == col)]

                    # copy in the baseline if required
                    if empty_col:
                        df_special_baseline_only_row_col = df_special_baseline_only[
                            (df_special_baseline_only[catplot_config["row"]] == row)]
                    else:
                        df_special_baseline_only_row_col = df_special_baseline_only[
                            (df_special_baseline_only[catplot_config["row"]] == row) & (
                                    df_special_baseline_only[catplot_config["col"]] == col)]
                    # if baseline is NOT in the current group -> select valid subsample and insert it
                    if df_special_baseline_only_row_col.shape[0] == 0:
                        df_baselines_copied = df_special_baseline_only.copy()
                        # if either in the row or the column dimension -> non-empty -> take this non-empty subset
                        if df_baselines_copied[df_baselines_copied[catplot_config["row"]] == row].shape[0] > 0:
                            df_baselines_copied = df_baselines_copied[
                                df_baselines_copied[catplot_config["row"]] == row]
                        elif not empty_col and \
                                df_baselines_copied[df_baselines_copied[catplot_config["col"]] == col].shape[
                                    0] > 0:
                            df_baselines_copied = df_baselines_copied[
                                df_baselines_copied[catplot_config["col"]] == col]

                        df_baselines_copied[catplot_config["row"]] = row
                        if not empty_col:
                            df_baselines_copied[catplot_config["col"]] = col
                        df_special_row_col = pd.concat([df_special_row_col, df_baselines_copied], ignore_index=True)
                    else:
                        # otherwise, insert the baseline (without modifications)
                        df_special_row_col = pd.concat([df_special_row_col, df_special_baseline_only],
                                                       ignore_index=True)

                    # fct_x___order_special = get_sorted_labels(df_special_row_col, catplot_config["x"])
                    # fct_hue_order_special = get_sorted_labels(df_special_row_col, catplot_config["hue"])
                    # make the plot
                    if empty_col:
                        ax_in_plot = axes[row_idx]
                    else:
                        ax_in_plot = axes[row_idx, col_idx]

                    sns.boxplot(data=df_special_row_col, x=catplot_config["x"], hue=catplot_config["hue"],
                                y=metric_name,
                                flierprops={"marker": "x"}, notch=True, bootstrap=1000,
                                order=x_order_global, hue_order=hue_labels_global, ax=ax_in_plot)
                    if empty_col:
                        ax_in_plot.set_title(f"{catplot_config['row']} {row}", fontsize=24)
                    else:
                        ax_in_plot.set_title(f"{catplot_config['row']} {row}, {catplot_config['col']} {col}",
                                             fontsize=24)
                    # only set x-label for the bottom-most row, else remove it
                    if row_idx == n_rows - 1:
                        ax_in_plot.set_xlabel(catplot_config["x"], fontsize=24)
                    else:
                        ax_in_plot.set_xlabel("")
                    # only set y-label for the left-most column, else remove it
                    if col_idx == 0:
                        ax_in_plot.set_ylabel(metric_name, fontsize=24)
                    else:
                        ax_in_plot.set_ylabel("")
                    ax_in_plot.tick_params(labelsize=20)
                    # don't show legend for the individual subplots -> only one shared one
                    ax_in_plot.get_legend().remove()
                    # Set y-ticks to be 0.1 steps
                    if metric_name in metric_0_1_range:
                        ax_in_plot.set_yticks(np.arange(0, 1.1, 0.1))
                    if len(x_order_global) > sns_num_xticks_rotation_threshold:
                        ax_in_plot.tick_params(axis='x', rotation=90)
                        plt.tight_layout()
                    # plt.setp(axes[row_idx, col_idx].get_legend().get_texts(), fontsize='16')
                    # plt.setp(axes[row_idx, col_idx].get_legend().get_title(), fontsize='16')

            # make a common legend
            # handlesAx, labelsAx = axes[0, 0].get_legend_handles_labels()
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.5,
                       title=catplot_config["hue"], title_fontsize=24, fontsize=20)
            plt.tight_layout()

            # save
            special_catplot_path = os.path.join(save_base_path,
                                                f"SpecialCatPlot_{metric_name}__{catplot_config['row']}_{catplot_config['col']}__x_{catplot_config['x']}_hue_{catplot_config['hue']}_box_plot{data_file_suffix}__{fig_size_name}.{file_suffix}")
            plt.savefig(special_catplot_path, bbox_inches='tight', **savefig_suffix_kwargs)

            # Also make a zoomed-in version
            min_lim = df_special[metric_name].min()
            max_lim = df_special[metric_name].max()
            # set the limits for all the subplots
            for row_idx, row in enumerate(fct_row_order_special):
                for col_idx, col in enumerate(fct_col_order_special):
                    if empty_col:
                        ax_in_plot = axes[row_idx]
                    else:
                        ax_in_plot = axes[row_idx, col_idx]
                    ax_in_plot.set_ylim(min_lim, max_lim)

            plt.tight_layout()
            special_catplot_path = os.path.join(save_base_path,
                                                f"SpecialCatPlot_{metric_name}__{catplot_config['row']}_{catplot_config['col']}__x_{catplot_config['x']}_hue_{catplot_config['hue']}_box_plot{data_file_suffix}__{fig_size_name}__ZOOMED_ALL.{file_suffix}")
            plt.savefig(special_catplot_path, bbox_inches='tight', **savefig_suffix_kwargs)

            # Also make a zoomed-in version
            min_lim = df_special[metric_name].quantile(0.05)
            max_lim = df_special[metric_name].quantile(0.95)
            # set the limits for all the subplots
            for row_idx, row in enumerate(fct_row_order_special):
                for col_idx, col in enumerate(fct_col_order_special):
                    if empty_col:
                        ax_in_plot = axes[row_idx]
                    else:
                        ax_in_plot = axes[row_idx, col_idx]
                    ax_in_plot.set_ylim(min_lim, max_lim)

            plt.tight_layout()
            special_catplot_path = os.path.join(save_base_path,
                                                f"SpecialCatPlot_{metric_name}__{catplot_config['row']}_{catplot_config['col']}__x_{catplot_config['x']}_hue_{catplot_config['hue']}_box_plot{data_file_suffix}__{fig_size_name}__ZOOMED.{file_suffix}")
            plt.savefig(special_catplot_path, bbox_inches='tight', **savefig_suffix_kwargs)
            plt.close()


def get_sorted_labels(data_df, column, prioritize_string_order=False):
    sorted_axis_labels = data_df[column].unique().tolist()
    sorted_axis_labels.sort(key=lambda x: label_sort_key_func(x, prioritize_string_order=prioritize_string_order), )
    return sorted_axis_labels


def sorted_labels_from_list(labels, prioritize_string_order=False):
    sorted_axis_labels = labels
    sorted_axis_labels.sort(key=lambda x: label_sort_key_func(x, prioritize_string_order=prioritize_string_order))
    return sorted_axis_labels


def make_metric_distribution_figures(df, save_base_path, metric_names, groupbyList=["Model", "Prompt ID"],
                                     groupByListExtension=[], file_suffix="", facet_col_n_docs="N-Input Docs",
                                     facet_col_size="#Input Article Tokens", skip_catplots=False, skip_violins=False) -> \
        Tuple[
            List[List[str]], List[List[str]]]:
    skip_individual_model_plots = True
    skip_individual_prompt_plots = True

    if df.shape[0] == 0:
        return [], []
    print("Making metric distribution figures...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"

    prompt_plot_paths = []
    model_plot_paths = []

    # TODO: groupByListExtension
    fullGroupByList = [[groupbyList[0], groupbyList[1]], [groupbyList[1], groupbyList[0]]] + groupByListExtension

    # make all subfolders
    for metric_name in metric_names:
        pathlib.Path(os.path.join(save_base_path, f"metric-{metric_name}")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(save_base_path, f"metric-{metric_name}", "PNG")).mkdir(parents=True, exist_ok=True)
    for model_name in df["Model"].unique():
        pathlib.Path(os.path.join(save_base_path, model_name)).mkdir(parents=True, exist_ok=True)
    for promptVersion in df["Prompt ID"].unique():
        pathlib.Path(os.path.join(save_base_path, f"Prompt-{promptVersion}")).mkdir(parents=True, exist_ok=True)

    max_xlabels_threshold = 50

    # sorted prompt versions (ordered as numbers, not strings)
    # promptVersions = df["Prompt ID"].unique().tolist()
    # promptVersions.sort(key=lambda x: int(x))
    # ...
    residualGroupBy = [col for col in groupbyList if col not in ["Prompt ID", "Model"]]
    out_paths = []

    # Loop over the metrics -> make 1 figure per metric, comparing groupByList[0] with groupByList[1] (with hue) -> make plot in both directions
    for metric_name in metric_names:

        figsizes = [(14, 8)]  # , (14, 8), (19, 8)  # [None, (20, 10), (18, 10)]
        save_suffixes = [("pdf", {})]  # [("pdf", {}), ("png", {"dpi": 300}), ("png", {"dpi": 250})]
        for fig_size, (suffix, suffix_kwargs) in zip(figsizes, save_suffixes):
            metric_plt_base_path = os.path.join(save_base_path, f"metric-{metric_name}")
            if suffix == "png":
                metric_plt_base_path = os.path.join(save_base_path, f"metric-{metric_name}", "PNG")

            # Set temporary theme
            fig_size_name = "default"
            if fig_size is not None:
                metric_plt_theme = {
                    "style": "darkgrid", "rc": {"figure.figsize": fig_size, "font.size": 24, },
                }
                sns.set_theme(**metric_plt_theme)
                sns.set(rc={'figure.figsize': fig_size})
                fig_size_name = f"{fig_size[0]}x{fig_size[1]}"

            for gbl in fullGroupByList:
                sorted_x_axis_labels = get_sorted_labels(df, gbl[0], prioritize_string_order=True)
                sorted_hue_labels = get_sorted_labels(df, gbl[1], prioritize_string_order=True)

                if not skip_violins:
                    violin_plot = sns.violinplot(data=df, x=gbl[0], hue=gbl[1], y=metric_name, points=500, cut=0,
                                                 order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                    # sns.move_legend(violin_plot, "upper left", bbox_to_anchor=(1, 1))
                    if metric_name in metric_0_1_range:
                        violin_plot.set_ylim(0, 1)
                        violin_plot.set_yticks(np.arange(0, 1.1, 0.1))
                    if len(sorted_x_axis_labels) > sns_num_xticks_rotation_threshold:
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                    # save
                    violin_plot_path = os.path.join(metric_plt_base_path,
                                                    f"{metric_name}_{gbl[0]}_{gbl[1]}_violin_plot{file_suffix}__{fig_size_name}.{suffix}")
                    plt.savefig(violin_plot_path, bbox_inches='tight', **suffix_kwargs)
                    out_paths.append(violin_plot_path)
                    plt.close()

                # dist_plot = sns.displot(data=df.dropna(), x=groupbyList[a], hue=groupbyList[b], y=metric_name, kind="kde", fill=True)
                # if metric_name in metric_0_1_range:
                #     dist_plot.set_ylim(0, 1)
                # # save
                # dist_plot_path = os.path.join(save_base_path, f"{metric_name}_{groupbyList[a]}_{groupbyList[b]}_distribution_plot{file_suffix}.pdf")
                # plt.savefig(dist_plot_path)
                # out_paths.append(dist_plot_path)
                # plt.close()

                sns.set(rc={'figure.figsize': fig_size})
                box_plot = sns.boxplot(data=df, x=gbl[0], hue=gbl[1], y=metric_name, flierprops={"marker": "x"},
                                       notch=True, bootstrap=1000,
                                       order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5, fontsize=20,
                           title=gbl[1], title_fontsize=18)
                if metric_name in metric_0_1_range:
                    box_plot.set_ylim(0, 1)
                # box_plot.axes.set_title("Title", fontsize=50)
                box_plot.set_xlabel(gbl[0], fontsize=20)
                box_plot.set_ylabel(metric_name, fontsize=20, rotation=0, labelpad=20)
                box_plot.yaxis.set_label_coords(0.0, 1.0)
                box_plot.tick_params(labelsize=18)
                # Set y-ticks to be 0.1 steps
                if metric_name in metric_0_1_range:
                    box_plot.set_yticks(np.arange(0, 1.1, 0.1))
                plt.setp(box_plot.get_legend().get_texts(), fontsize='16')
                plt.setp(box_plot.get_legend().get_title(), fontsize='16')
                # save
                plt.tight_layout()
                box_plot_path = os.path.join(metric_plt_base_path,
                                             f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)
                out_paths.append(box_plot_path)

                if len(sorted_x_axis_labels) > sns_num_xticks_rotation_threshold:
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    box_plot_path = os.path.join(metric_plt_base_path,
                                                 f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}_ROTATED.{suffix}")
                    plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)
                    plt.xticks(rotation=0)

                # Also make a zoomed-in version
                min_lim = df[metric_name].min()
                max_lim = df[metric_name].max()
                box_plot.set_ylim(min_lim, max_lim)
                box_plot_path = os.path.join(metric_plt_base_path,
                                             f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}__ZOOMED_ALL.{suffix}")
                plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)

                # Also make a zoomed-in version
                min_lim = df[metric_name].quantile(0.1)
                max_lim = df[metric_name].quantile(0.9)
                box_plot.set_ylim(min_lim, max_lim)
                box_plot_path = os.path.join(metric_plt_base_path,
                                             f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}__ZOOMED.{suffix}")
                plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)

                # and make a version with the legend on top, and reset the range for the plots
                if metric_name in metric_0_1_range:
                    box_plot.set_ylim(0, 1)
                else:
                    min_lim = df[metric_name].min()
                    max_lim = df[metric_name].max()
                    box_plot.set_ylim(min_lim, max_lim)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', borderaxespad=0.5, ncols=4)
                plt.setp(box_plot.get_legend().get_texts(), fontsize='20')
                plt.setp(box_plot.get_legend().get_title(), fontsize='20')
                box_plot_path = os.path.join(metric_plt_base_path,
                                             f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}_TopLegend.{suffix}")
                plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)

                plt.close()

                df[facet_col_n_docs] = df[facet_col_n_docs].astype(int)

                # Make a facet-plot with the number of input documents as the facet column
                for fct_row, fct_hue in [(facet_col_n_docs, gbl[1]), (gbl[1], facet_col_n_docs)]:

                    fct_hue_order = get_sorted_labels(df, fct_hue)
                    fct_row_order = get_sorted_labels(df, fct_row)
                    fct_x___order = get_sorted_labels(df, gbl[0])
                    if not skip_catplots:
                        catplt = sns.catplot(df, row=fct_row, row_order=fct_row_order, height=10, aspect=1.5, x=gbl[0],
                                             order=fct_x___order, hue=fct_hue, hue_order=fct_hue_order, y=metric_name,
                                             kind='box',
                                             flierprops={"marker": "x"}, notch=True,
                                             bootstrap=1000)
                        sns.move_legend(catplt, "upper left", bbox_to_anchor=(1, 1), borderaxespad=0.5)
                        # Set y-ticks to be 0.1 steps
                        if metric_name in metric_0_1_range:
                            catplt.set(yticks=np.arange(0, 1.1, 0.1))
                        # always rotate them
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                        metrics_n_in_docs_plt_path = os.path.join(metric_plt_base_path,
                                                                  f"CatPlot_{metric_name}_Cat_{fct_row}_{gbl[0]}_{fct_hue}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                        plt.savefig(metrics_n_in_docs_plt_path)
                        plt.close()

                    # Make also a CatAgg plot where the row and x categories are concatenated -> all shown within 1 plot instead of multiple concatenated plots
                    if not skip_catplots:
                        df_catagg = df.copy()
                        catagg_colname = f"{fct_row} / {gbl[0]}"
                        df_catagg[catagg_colname] = df_catagg.apply(
                            lambda row: f"{row[fct_row]}\n{row[gbl[0]]}", axis=1)
                        catagg_x___order = get_sorted_labels(df_catagg, catagg_colname)
                        if len(catagg_x___order) > max_xlabels_threshold:
                            print("Skipping catagg plot where row and x categories are concatenated")
                        else:
                            catagg_hue___order = get_sorted_labels(df_catagg, fct_hue)
                            box_plot = sns.boxplot(data=df_catagg, x=catagg_colname, hue=fct_hue, y=metric_name,
                                                   flierprops={"marker": "x"},
                                                   notch=True, bootstrap=1000,
                                                   order=catagg_x___order, hue_order=catagg_hue___order)
                            plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                            if metric_name in metric_0_1_range:
                                box_plot.set_ylim(0, 1)
                            # box_plot.axes.set_title("Title", fontsize=50)
                            box_plot.set_xlabel(catagg_colname, fontsize=18)
                            box_plot.set_ylabel(metric_name, fontsize=18)
                            box_plot.tick_params(labelsize=16)
                            # Set y-ticks to be 0.1 steps
                            if metric_name in metric_0_1_range:
                                box_plot.set_yticks(np.arange(0, 1.1, 0.1))
                            if len(sorted_x_axis_labels) > sns_num_xticks_rotation_threshold:
                                plt.xticks(rotation=90)
                                plt.tight_layout()
                            plt.setp(box_plot.get_legend().get_texts(), fontsize='16')
                            plt.setp(box_plot.get_legend().get_title(), fontsize='16')
                            # save
                            metrics_n_in_docs_plt_path = os.path.join(metric_plt_base_path,
                                                                      f"CatAggPlot_{metric_name}_Cat_{fct_row}_{gbl[0]}__{fct_hue}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                            plt.savefig(metrics_n_in_docs_plt_path, bbox_inches='tight', **suffix_kwargs)
                            plt.close()

                # Make a facet-plot with the number of input article tokens as the facet column (bucket-size 500)
                if not skip_catplots:
                    bckt_size = 1000
                    bckt_colname = f"{facet_col_size} Bucket"
                    df[bckt_colname] = df[facet_col_size].apply(
                        lambda x: f"{int(math.floor(x / bckt_size)) * bckt_size}".rjust(
                            5) + "-" + f"{int(math.ceil(x / bckt_size)) * bckt_size}".rjust(5))
                    bckt_vals = df[bckt_colname].unique().tolist()
                    bckt_vals.sort()
                    for fct_row, fct_hue in [(bckt_colname, gbl[1]), (gbl[1], bckt_colname)]:
                        fct_hue_order = get_sorted_labels(df, fct_hue)
                        fct_row_order = get_sorted_labels(df, fct_row)
                        fct_x___order = get_sorted_labels(df, gbl[0])
                        if not skip_catplots:
                            catplt = sns.catplot(df, row=fct_row, row_order=fct_row_order, height=10, aspect=1.5,
                                                 x=gbl[0],
                                                 order=fct_x___order, hue=fct_hue, hue_order=fct_hue_order,
                                                 y=metric_name,
                                                 kind='box',
                                                 flierprops={"marker": "x"}, notch=True,
                                                 bootstrap=1000)
                            sns.move_legend(catplt, "upper left", bbox_to_anchor=(1, 1), borderaxespad=0.5)
                            # Set y-ticks to be 0.1 steps
                            if metric_name in metric_0_1_range:
                                catplt.set(yticks=np.arange(0, 1.1, 0.1))
                                # always rotate them
                                plt.xticks(rotation=90)
                                plt.tight_layout()
                            metrics_in_size_plt_path = os.path.join(metric_plt_base_path,
                                                                    f"CatPlot_{metric_name}_Cat_{fct_row}_{gbl[0]}_{fct_hue}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                            plt.savefig(metrics_in_size_plt_path, **suffix_kwargs)
                            plt.close()

                    # Make also a CatAgg plot where the row and x categories are concatenated -> all shown within 1 plot instead of multiple concatenated plots
                    df_catagg = df.copy()
                    catagg_colname = f"{fct_row} / {gbl[0]}"
                    df_catagg[catagg_colname] = df_catagg.apply(
                        lambda row: f"{row[fct_row]}\n{row[gbl[0]]}", axis=1)
                    catagg_x___order = get_sorted_labels(df_catagg, catagg_colname)
                    catagg_hue___order = get_sorted_labels(df_catagg, fct_hue)
                    if len(catagg_x___order) > max_xlabels_threshold:
                        print("Skipping catagg plot where row and x categories are concatenated")
                    else:
                        box_plot = sns.boxplot(data=df_catagg, x=catagg_colname, hue=fct_hue, y=metric_name,
                                               flierprops={"marker": "x"},
                                               notch=True, bootstrap=1000,
                                               order=catagg_x___order, hue_order=catagg_hue___order)
                        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                        if metric_name in metric_0_1_range:
                            box_plot.set_ylim(0, 1)
                        # box_plot.axes.set_title("Title", fontsize=50)
                        box_plot.set_xlabel(catagg_colname, fontsize=18)
                        box_plot.set_ylabel(metric_name, fontsize=18)
                        box_plot.tick_params(labelsize=16)
                        # Set y-ticks to be 0.1 steps
                        if metric_name in metric_0_1_range:
                            box_plot.set_yticks(np.arange(0, 1.1, 0.1))
                        if len(sorted_x_axis_labels) > sns_num_xticks_rotation_threshold:
                            plt.xticks(rotation=90)
                            plt.tight_layout()
                        plt.setp(box_plot.get_legend().get_texts(), fontsize='16')
                        plt.setp(box_plot.get_legend().get_title(), fontsize='16')
                        # save
                        metrics_n_in_docs_plt_path = os.path.join(metric_plt_base_path,
                                                                  f"CatAggPlot_{metric_name}_Cat_{fct_row}_{gbl[0]}__{fct_hue}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                        plt.savefig(metrics_n_in_docs_plt_path, bbox_inches='tight', **suffix_kwargs)
                        plt.close()

                # Reset the theme
                sns.set_theme(**DEFAULT_THEME)

            # Make special metric catplots if specified
            special_catplot_configs = experiment_config[experiment_name].get("special_metric_catplots", [])
            make_special_catplots(df, metric_plt_base_path, metric_name, special_catplot_configs,
                                  savefig_suffix_kwargs=suffix_kwargs, data_file_suffix=file_suffix,
                                  fig_size_name=fig_size_name, file_suffix=suffix)

        # Prompt Variant Plot
        prompt_desc_col = "Prompt Description"
        prompt_variant_col = "Prompt Variant"
        prompt_sep_col = "Has Prompt-Summary Separator"
        for plt_kind in ['box', 'violin']:
            catplt = sns.catplot(df, row=prompt_desc_col, height=10, aspect=1.5, x=prompt_variant_col,
                                 hue=prompt_sep_col,
                                 hue_order=[False, True], y=metric_name, kind=plt_kind, flierprops={"marker": "x"},
                                 notch=True,
                                 bootstrap=1000)
            # Set y-ticks to be 0.1 steps
            if metric_name in metric_0_1_range:
                catplt.set(yticks=np.arange(0, 1.1, 0.1))
            prompt_variant_plot_path = os.path.join(save_base_path, f"metric-{metric_name}",
                                                    f"PromptVariant_{metric_name}__{prompt_desc_col}_{prompt_variant_col}_{prompt_sep_col}_{plt_kind}_plot{file_suffix}.pdf")
            plt.savefig(prompt_variant_plot_path)
            plt.close()

    # Loop over the models -> making 1 figure per metric, comparing the prompts (on the same model)
    if not skip_individual_model_plots:
        for model_name in df["Model"].unique():
            out_paths = []
            df_model = df[df["Model"] == model_name]

            for metric_name in metric_names:
                # make a violin plot showing the distribution of the metric values for each prompt
                if len(residualGroupBy) > 0:
                    sorted_x_axis_labels = get_sorted_labels(df, "Prompt ID")
                    sorted_hue_labels = get_sorted_labels(df, residualGroupBy[0])
                    if not skip_violins:
                        violin_plot = sns.violinplot(data=df_model, x="Prompt ID", hue=residualGroupBy[0],
                                                     y=metric_name,
                                                     order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                        if metric_name in metric_0_1_range:
                            violin_plot.set_ylim(0, 1)
                        # save
                        violin_plot_path = os.path.join(save_base_path, model_name,
                                                        f"{model_name}_{metric_name}_violin_plot{file_suffix}.pdf")
                        plt.savefig(violin_plot_path)
                        out_paths.append(violin_plot_path)
                        plt.close()

                    box_plot = sns.boxplot(data=df_model, x="Prompt ID", hue=residualGroupBy[0], y=metric_name,
                                           order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    if metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # save
                    box_plot_path = os.path.join(save_base_path, model_name,
                                                 f"{model_name}_{metric_name}_box_plot{file_suffix}.pdf")
                    plt.savefig(box_plot_path)
                    out_paths.append(box_plot_path)
                    plt.close()

                    if not skip_violins:
                        violin_plot = sns.violinplot(data=df_model, x=residualGroupBy[0], hue='Prompt ID',
                                                     y=metric_name,
                                                     order=sorted_hue_labels, hue_order=sorted_x_axis_labels)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                        if metric_name in metric_0_1_range:
                            violin_plot.set_ylim(0, 1)
                        # save
                        violin_plot_path = os.path.join(save_base_path, model_name,
                                                        f"{model_name}_{metric_name}_R_violin_plot{file_suffix}.pdf")
                        plt.savefig(violin_plot_path)
                        out_paths.append(violin_plot_path)
                        plt.close()

                    box_plot = sns.boxplot(data=df_model, x=residualGroupBy[0], hue='Prompt ID', y=metric_name,
                                           order=sorted_hue_labels, hue_order=sorted_x_axis_labels)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    if metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # save
                    box_plot_path = os.path.join(save_base_path, model_name,
                                                 f"{model_name}_{metric_name}_R_box_plot{file_suffix}.pdf")
                    plt.savefig(box_plot_path)
                    out_paths.append(box_plot_path)
                    plt.close()
                else:
                    if not skip_violins:
                        sorted_x_axis_labels = get_sorted_labels(df, "Prompt ID")
                        violin_plot = sns.violinplot(data=df_model, x="Prompt ID", y=metric_name,
                                                     order=sorted_x_axis_labels)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                        # sns.move_legend(box_plot, "upper left", bbox_to_anchor=(1, 1))
                        if metric_name in metric_0_1_range:
                            violin_plot.set_ylim(0, 1)
                        # save
                        violin_plot_path = os.path.join(save_base_path, model_name,
                                                        f"{model_name}_{metric_name}_violin_plot{file_suffix}.pdf")
                        plt.savefig(violin_plot_path)
                        out_paths.append(violin_plot_path)
                        plt.close()

                    box_plot = sns.boxplot(data=df_model, x="Prompt ID", y=metric_name, order=sorted_x_axis_labels)
                    # sns.move_legend(box_plot, "upper left", bbox_to_anchor=(1, 1))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    if metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # save
                    box_plot_path = os.path.join(save_base_path, model_name,
                                                 f"{model_name}_{metric_name}_box_plot{file_suffix}.pdf")
                    plt.savefig(box_plot_path)
                    out_paths.append(box_plot_path)
                    plt.close()
            model_plot_paths.append(out_paths)

    # Loop over the prompts -> making 1 figure per metric, comparing the models (on the same prompt)
    if not skip_individual_prompt_plots:
        for promptVersion in df["Prompt ID"].unique():
            out_paths = []
            df_prompt = df[df["Prompt ID"] == promptVersion]
            for metric_name in metric_names:
                # make a violin plot showing the distribution of the metric values for each model
                if len(residualGroupBy) > 0:
                    sorted_x_axis_labels = get_sorted_labels(df, "Model")
                    sorted_hue_labels = get_sorted_labels(df, residualGroupBy[0])
                    if not skip_violins:
                        violin_plot = sns.violinplot(data=df_prompt, x="Model", hue=residualGroupBy[0], y=metric_name,
                                                     order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                        if metric_name in metric_0_1_range:
                            violin_plot.set_ylim(0, 1)
                        # save
                        violin_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}",
                                                        f"Prompt_{promptVersion}_{metric_name}_violin_plot{file_suffix}.pdf")
                        plt.savefig(violin_plot_path)
                        out_paths.append(violin_plot_path)
                        plt.close()

                    box_plot = sns.boxplot(data=df_prompt, x="Model", hue=residualGroupBy[0], y=metric_name,
                                           order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    if metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # save
                    box_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}",
                                                 f"Prompt_{promptVersion}_{metric_name}_box_plot{file_suffix}.pdf")
                    plt.savefig(box_plot_path)
                    out_paths.append(box_plot_path)
                    plt.close()

                    if not skip_violins:
                        violin_plot = sns.violinplot(data=df_prompt, x=residualGroupBy[0], hue="Model", y=metric_name,
                                                     order=sorted_hue_labels, hue_order=sorted_x_axis_labels)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                        if metric_name in metric_0_1_range:
                            violin_plot.set_ylim(0, 1)
                        # save
                        violin_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}",
                                                        f"Prompt_{promptVersion}_{metric_name}_R_violin_plot{file_suffix}.pdf")
                        plt.savefig(violin_plot_path)
                        out_paths.append(violin_plot_path)
                        plt.close()

                    box_plot = sns.boxplot(data=df_prompt, x=residualGroupBy[0], hue="Model", y=metric_name,
                                           order=sorted_hue_labels, hue_order=sorted_x_axis_labels)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    if metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # save
                    box_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}",
                                                 f"Prompt_{promptVersion}_{metric_name}_R_box_plot{file_suffix}.pdf")
                    plt.savefig(box_plot_path)
                    out_paths.append(box_plot_path)
                    plt.close()
                else:
                    if not skip_violins:
                        sorted_x_axis_labels = get_sorted_labels(df, "Model")
                        violin_plot = sns.violinplot(data=df_prompt, x="Model", y=metric_name,
                                                     order=sorted_x_axis_labels)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                        if metric_name in metric_0_1_range:
                            violin_plot.set_ylim(0, 1)
                        # save
                        violin_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}",
                                                        f"Prompt_{promptVersion}_{metric_name}_violin_plot{file_suffix}.pdf")
                        plt.savefig(violin_plot_path)
                        out_paths.append(violin_plot_path)
                        plt.close()

                    box_plot = sns.boxplot(data=df_prompt, x="Model", y=metric_name, order=sorted_x_axis_labels)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    if metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # save
                    box_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}",
                                                 f"Prompt_{promptVersion}_{metric_name}_box_plot{file_suffix}.pdf")
                    plt.savefig(box_plot_path)
                    out_paths.append(box_plot_path)
                    plt.close()
            prompt_plot_paths.append(out_paths)

    return prompt_plot_paths, model_plot_paths


def make_seahorse_distribution_figures(df, save_base_path, standard_evaluation_metric_names,
                                       groupbyList=["Model", "Prompt ID"], groupByListExtension=[],
                                       file_suffix="", facet_col_n_docs="N-Input Docs",
                                       facet_col_size="#Input Article Tokens", skip_catplots=False):
    print("Making SEAHORSE distribution figures...")

    save_base_path = os.path.join(save_base_path, "SEAHORSE")

    fullGroupByList = [[groupbyList[0], groupbyList[1]], [groupbyList[1], groupbyList[0]]] + groupByListExtension

    metric_names = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in SEAHORSE_Plot_Metric_Names if
                    extension not in SEAHORSE_Binary_Names]
    binary_names = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in SEAHORSE_Plot_Metric_Names if
                    extension in SEAHORSE_Binary_Names]
    deeper_seahorse_inspection_metric_names = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in
                                               SEAHORSE_deeper_inspection_extensions if
                                               extension not in SEAHORSE_Binary_Names]

    # make all subfolders
    for metric_name in metric_names:
        if metric_name in df.columns:
            pathlib.Path(os.path.join(save_base_path, f"seahorse-{metric_name}")).mkdir(parents=True, exist_ok=True)
    for binary_name in binary_names:
        if binary_name in df.columns:
            pathlib.Path(os.path.join(save_base_path, f"seahorse-{binary_name}")).mkdir(parents=True, exist_ok=True)

    # TODO: make scatter plots showing the basic seahorse scores against each other, and also against basic metrics
    #  ... to visually inspect whether there is a correlation between the values

    figsizes = [(14, 8)]  # [None, (20, 10), (18, 10)]
    save_suffixes = [("pdf", {})]  # [("pdf", {}), ("png", {"dpi": 300}), ("png", {"dpi": 250})]
    for fig_size, (suffix, suffix_kwargs) in zip(figsizes, save_suffixes):
        for gbl in fullGroupByList:
            sorted_x_axis_labels = get_sorted_labels(df, gbl[0], prioritize_string_order=True)
            sorted_hue_labels = get_sorted_labels(df, gbl[1], prioritize_string_order=True)

            # make all the plots for the metric names
            for metric_name in metric_names:
                # if the metric name is not a column -> skip
                if metric_name not in df.columns:
                    print(f"Skipping {metric_name} because it is not a column in the dataframe")
                    continue

                # Make the actual plots
                metric_plt_base_path = os.path.join(save_base_path, f"seahorse-{metric_name}")
                if suffix == "png":
                    metric_plt_base_path = os.path.join(save_base_path, f"seahorse-{metric_name}", "PNG")

                # Set temporary theme
                fig_size_name = "default"
                if fig_size is not None:
                    metric_plt_theme = {
                        "style": "darkgrid", "rc": {"figure.figsize": fig_size, "font.size": 24, },
                    }
                    sns.set_theme(**metric_plt_theme)
                    fig_size_name = f"{fig_size[0]}x{fig_size[1]}"

                # Show distribution via a boxplot
                box_plot = sns.boxplot(data=df, x=gbl[0], hue=gbl[1], y=metric_name, flierprops={"marker": "x"},
                                       notch=True, bootstrap=1000,
                                       order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                if metric_name in metric_0_1_range:
                    box_plot.set_ylim(0, 1)
                # box_plot.axes.set_title("Title", fontsize=50)
                box_plot.set_xlabel(gbl[0], fontsize=20)
                box_plot.set_ylabel(metric_name, fontsize=20)
                box_plot.tick_params(labelsize=18)
                # Set y-ticks to be 0.1 steps
                if metric_name in metric_0_1_range:
                    box_plot.set_yticks(np.arange(0, 1.1, 0.1))
                if len(sorted_x_axis_labels) > sns_num_xticks_rotation_threshold:
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                plt.setp(box_plot.get_legend().get_texts(), fontsize='20')
                plt.setp(box_plot.get_legend().get_title(), fontsize='20')
                # save
                box_plot_path = os.path.join(metric_plt_base_path,
                                             f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)

                # make an additional plot with the legend on the bottom in the center
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", borderaxespad=0.5,
                           ncols=4)
                plt.setp(box_plot.get_legend().get_texts(), fontsize='20')
                plt.setp(box_plot.get_legend().get_title(), fontsize='20')
                box_plot_path = os.path.join(metric_plt_base_path,
                                             f"{metric_name}_{gbl[0]}_{gbl[1]}_box_plot{file_suffix}__{fig_size_name}_TopLegend.{suffix}")
                plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)
                plt.close()

                df[facet_col_n_docs] = df[facet_col_n_docs].astype(int)

                # Make a facet-plot with the number of input documents as the facet column
                if not skip_catplots:
                    for fct_row, fct_hue in [(facet_col_n_docs, gbl[1]), (gbl[1], facet_col_n_docs)]:
                        fct_hue_order = get_sorted_labels(df, fct_hue)
                        fct_row_order = get_sorted_labels(df, fct_row)
                        fct_x___order = get_sorted_labels(df, gbl[0])
                        catplt = sns.catplot(df, row=fct_row, row_order=fct_row_order, height=10, aspect=1.5, x=gbl[0],
                                             order=fct_x___order, hue=fct_hue, hue_order=fct_hue_order, y=metric_name,
                                             kind='box',
                                             flierprops={"marker": "x"}, notch=True,
                                             bootstrap=1000)
                        # Set y-ticks to be 0.1 steps
                        if metric_name in metric_0_1_range:
                            catplt.set(yticks=np.arange(0, 1.1, 0.1))
                        # always rotate them
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                        metrics_n_in_docs_plt_path = os.path.join(metric_plt_base_path,
                                                                  f"CatPlot_{metric_name}_Cat_{fct_row}_{gbl[0]}_{fct_hue}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                        plt.savefig(metrics_n_in_docs_plt_path)
                        plt.close()

            # ... and also for the binary columns
            for binary_name in binary_names:
                if binary_name not in df.columns:
                    print(f"Skipping {binary_name} because it is not a column in the dataframe")
                    continue

                binary_plt_base_path = os.path.join(save_base_path, f"seahorse-{binary_name}")
                if suffix == "png":
                    binary_plt_base_path = os.path.join(save_base_path, f"seahorse-{binary_name}", "PNG")

                # Set temporary theme
                fig_size_name = "default"
                if fig_size is not None:
                    binary_plt_theme = {
                        "style": "darkgrid", "rc": {"figure.figsize": fig_size, "font.size": 24, },
                    }
                    sns.set_theme(**binary_plt_theme)
                    fig_size_name = f"{fig_size[0]}x{fig_size[1]}"

                # make a stacked 100% bar plot (with both group-by-dimensions)
                df_binary_grouped = df.groupby(gbl)[binary_name].value_counts(normalize=True).unstack(binary_name)
                df_binary_grouped = df_binary_grouped.reset_index()
                if False not in df_binary_grouped.columns:
                    df_binary_grouped[False] = 0.0
                if True not in df_binary_grouped.columns:
                    df_binary_grouped[True] = 0.0
                df_binary_grouped["False real"] = df_binary_grouped[False]
                df_binary_grouped[False] = 1.0
                # sns.barplot(data=df_binary_grouped, x=gbl[0], y=False, hue=gbl[1], order=sorted_x_axis_labels,
                #             color="tab:blue")
                sorted_x_axis_labels_str = [str(x) for x in sorted_x_axis_labels]
                sorted_hue_labels_str = [str(x) for x in sorted_hue_labels]
                df_binary_grouped[gbl[0]] = df_binary_grouped[gbl[0]].astype(str)
                df_binary_grouped[gbl[1]] = df_binary_grouped[gbl[1]].astype(str)
                sns.barplot(data=df_binary_grouped, x=gbl[0], y=True, hue=gbl[1], order=sorted_x_axis_labels_str,
                            hue_order=sorted_hue_labels_str)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                plt.tight_layout()
                binary_plt_path = os.path.join(binary_plt_base_path,
                                               f"{binary_name}__{gbl[0]}_{gbl[1]}_stacked_plot{file_suffix}__{fig_size_name}.{suffix}")
                plt.savefig(binary_plt_path, bbox_inches='tight', **suffix_kwargs)
                plt.close()

                # save as tex table
                df_binary_grouped_to_tex = df_binary_grouped.copy()
                df_binary_grouped_to_tex[False] = df_binary_grouped_to_tex["False real"]
                df_binary_grouped_to_tex = df_binary_grouped_to_tex.drop(columns=["False real"])
                df_binary_grouped_to_tex = df_binary_grouped_to_tex.rename(columns={True: "True", False: "False"})
                binary_plt_path = os.path.join(binary_plt_base_path,
                                               f"{binary_name}__{gbl[0]}_{gbl[1]}_stacked_plot{file_suffix}__{fig_size_name}.tex")
                df_binary_grouped_to_tex.to_latex(binary_plt_path, index=False, escape=False, float_format="%.3f")

                # only with 1 group-by-dimension
                for gbl_subval in [gbl[0], gbl[1]]:
                    # make a stacked 100% bar plot (with both group-by-dimensions)
                    df_binary_grouped = df.groupby([gbl_subval])[binary_name].value_counts(normalize=True).unstack(
                        binary_name)
                    df_binary_grouped = df_binary_grouped.reset_index()
                    if False not in df_binary_grouped.columns:
                        df_binary_grouped[False] = 0.0
                    if True not in df_binary_grouped.columns:
                        df_binary_grouped[True] = 0.0
                    df_binary_grouped["False real"] = df_binary_grouped[False]
                    df_binary_grouped[False] = 1.0
                    # sns.barplot(data=df_binary_grouped, x=gbl[0], y=False, hue=gbl[1], order=sorted_x_axis_labels,
                    #             color="tab:blue")
                    sorted_gbl_subval_labels = get_sorted_labels(df, gbl_subval, prioritize_string_order=True)
                    sns.barplot(data=df_binary_grouped, x=gbl_subval, y=True, order=sorted_gbl_subval_labels, )
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                    plt.tight_layout()
                    binary_plt_path = os.path.join(binary_plt_base_path,
                                                   f"{binary_name}__{gbl[0]}__stacked_plot{file_suffix}__{fig_size_name}.{suffix}")
                    plt.savefig(binary_plt_path, bbox_inches='tight', **suffix_kwargs)
                    plt.close()

                    # save as tex table
                    df_binary_grouped_to_tex = df_binary_grouped.copy()
                    df_binary_grouped_to_tex[False] = df_binary_grouped_to_tex["False real"]
                    df_binary_grouped_to_tex = df_binary_grouped_to_tex.drop(columns=["False real"])
                    df_binary_grouped_to_tex = df_binary_grouped_to_tex.rename(columns={True: "True", False: "False"})
                    binary_plt_path = os.path.join(binary_plt_base_path,
                                                   f"{binary_name}__{gbl[0]}_{gbl[1]}_stacked_plot{file_suffix}__{fig_size_name}.tex")
                    df_binary_grouped_to_tex.to_latex(binary_plt_path, index=False, escape=False, float_format="%.3f")

                # e.g. sns.catplot(x='car',hue='TARGET_happiness',data=df,kind="count")
                # e.g. pd.crosstab(df['car'],df['TARGET_happiness']).plot.bar(stacked=True) -> stacked, make sure to make at least 1 stacked bar plot

        for metric_name in deeper_seahorse_inspection_metric_names:
            metric_plt_base_path = os.path.join(save_base_path, f"seahorse-{metric_name}")
            if suffix == "png":
                metric_plt_base_path = os.path.join(save_base_path, f"seahorse-{metric_name}", "PNG")

            # Set temporary theme
            fig_size_name = "default"
            if fig_size is not None:
                metric_plt_theme = {
                    "style": "darkgrid", "rc": {"figure.figsize": fig_size, "font.size": 24, },
                }
                sns.set_theme(**metric_plt_theme)
                fig_size_name = f"{fig_size[0]}x{fig_size[1]}"

            # Make special metric catplots if specified
            special_catplot_configs = experiment_config[experiment_name].get("special_metric_catplots", [])
            make_special_catplots(df, metric_plt_base_path, metric_name, special_catplot_configs,
                                  savefig_suffix_kwargs=suffix_kwargs, data_file_suffix=file_suffix,
                                  fig_size_name=fig_size_name, file_suffix=suffix)

        for binary_name in binary_names:

            binary_plt_base_path = os.path.join(save_base_path, f"seahorse-{binary_name}")
            if suffix == "png":
                binary_plt_base_path = os.path.join(save_base_path, f"seahorse-{binary_name}", "PNG")

            # create metric plots with the binary column as the hue -> BUT ONLY FOR MAIN GROUP BY
            for std_eval_metric_name in standard_evaluation_metric_names:
                for gbl in groupbyList:
                    sorted_x_axis_labels = get_sorted_labels(df, gbl, prioritize_string_order=True)
                    subgroup_hue_labels = get_sorted_labels(df, binary_name, prioritize_string_order=True)

                    box_plot = sns.boxplot(data=df, x=gbl, hue=binary_name, y=std_eval_metric_name,
                                           flierprops={"marker": "x"},
                                           notch=True, bootstrap=1000,
                                           order=sorted_x_axis_labels, hue_order=subgroup_hue_labels)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5)
                    if std_eval_metric_name in metric_0_1_range:
                        box_plot.set_ylim(0, 1)
                    # box_plot.axes.set_title("Title", fontsize=50)
                    box_plot.set_xlabel(gbl[0], fontsize=18)
                    box_plot.set_ylabel(std_eval_metric_name, fontsize=18)
                    box_plot.tick_params(labelsize=16)
                    # Set y-ticks to be 0.1 steps
                    if std_eval_metric_name in metric_0_1_range:
                        box_plot.set_yticks(np.arange(0, 1.1, 0.1))
                    if len(sorted_x_axis_labels) > sns_num_xticks_rotation_threshold:
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                    plt.setp(box_plot.get_legend().get_texts(), fontsize='16')
                    plt.setp(box_plot.get_legend().get_title(), fontsize='16')
                    # save
                    box_plot_path = os.path.join(binary_plt_base_path,
                                                 f"{std_eval_metric_name}__{gbl}__{binary_name}_box_plot{file_suffix}__{fig_size_name}.{suffix}")
                    plt.savefig(box_plot_path, bbox_inches='tight', **suffix_kwargs)
                    plt.close()

    # # make a pairplot for all metrics to look at the correlations
    # pairplot_seahorse_metric_names = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in
    #                                   SEAHORSE_Pairplot_Metric_Names]
    # pairplot_seahorse_metric_names = [x for x in pairplot_seahorse_metric_names if x in df.columns]
    # pairplot_metrics = StandardEval_Pairplot_Metric_Names + pairplot_seahorse_metric_names  # + binary_names
    # for gbl in [groupbyList]:  # fullGroupByList:
    #     sorted_groups = get_sorted_labels(df, gbl[0], prioritize_string_order=True)
    #     # sorted_hue_labels = get_sorted_labels(df, gbl[1], prioritize_string_order=True)
    #
    #     if len(sorted_groups) > 1:  # only initial sample, do more if it is interesting
    #         sorted_groups = sorted_groups[:1]
    #
    #     for group in sorted_groups:
    #         df_group = df[df[gbl[0]] == group]
    #
    #         # prevent empty axes
    #         skip = False
    #         hue_vals = df_group[gbl[1]].unique()
    #         for hue_val in hue_vals:
    #             if len(df_group[df_group[gbl[1]] == hue_val]) < 50:
    #                 skip = True
    #                 break
    #         if len(df_group) < 20 or skip:
    #             continue
    #
    #         sns.pairplot(df_group, hue=gbl[1], vars=pairplot_metrics, kind="kde", diag_kind="kde")
    #         plt.tight_layout()
    #         pairplot_path = os.path.join(save_base_path, f"Pairplot_metrics_{gbl[0]}_{group}_by_{gbl[1]}.pdf")
    #         plt.savefig(pairplot_path)
    #         plt.close()

    return


def get_prompts_by_experiment_name(experiment_name):
    # Get the prompts from the prompts_bag.json file for the given experiment
    prompts_bag_path = f"prompts_bag.json"
    with open(prompts_bag_path, "r") as f:
        prompts_bag = json.load(f)
        if experiment_name not in prompts_bag:
            experiment_prompts_bag_alias = experiment_config[experiment_name].get("prompts_bag_alias", experiment_name)
            if experiment_prompts_bag_alias in prompts_bag:
                prompts = prompts_bag[experiment_prompts_bag_alias]
            else:
                raise ValueError(f"No prompts found for experiment {experiment_name}")
        else:
            prompts = prompts_bag[experiment_name]
    return prompts


def get_metrics_info(df) -> Tuple[List[str], Dict[str, bool]]:
    """
    Returns a list of metric names and a dicti onary mapping each metric name to a list of percentiles to be calculated.
    :param df: DataFrame containing the results
    :return: metric_names, metric_ordering
        metric_names: List of metric names
        metric_ordering: Dictionary mapping each metric name a boolean indicating whether a larger metric value is better or not
    """
    exclude = [column_rename_map[x] for x in [
        'prompt_0', 'logit_0', 'truth', 'dataset', 'promptVersion', 'model', 'model-fullname', 'lang', 'lang_score',
        'temperature', 'precision', 'task_name', 'dataset-annotation', 'n-shot', 'preprocessing-method',
        'preprocessing-parameters', 'preprocessing-order',
        "prompt-description", "prompt-variant", "prompt-separator", "prompt-desc-id"
    ]]
    exclude += ['doc_id', 'N-Input Docs', '#Input Article Tokens', '#Input Article Full Tokens',
                '#Predicted Tokens', '#Predicted Full Tokens', '#GT-Summary Tokens', '#GT-Summary Full Tokens',
                'Preprocessing + N-Shot', '#Prompt Tokens', 'Prompt Name', 'Prompt Language',
                'Prompt Lang. (Separator)', "Model Type",
                'Is Bulletpoint Summary', "Dataset Annotation (Is Bullet Point Summary)",
                "Large Article", "Dataset Annotation (incl. article size)", "Size subset",
                "Dataset Annotation (Prompt Name)", "Dataset ID", "Preprocessing (N-Shot)",
                "Preprocessing (N-Shot) (#Tokens)", "Method (N-Shot) (#Tokens)", "Method (N-Shot)",
                "Example Source + N-Shot",
                "Method (N-Shot) (Source)", "Method (#Tokens) (Source)", "(N-Shot) (#Tokens) (Source)", "Method",
                ]
    exclude_metrics = ["Coverage (Prompt)", "Density (Prompt)", "Compression (Prompt)", "Compression (Full)"]
    exclude_seahorse = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in
                        SEAHORSE_Plot_Metric_Names]
    exclude_seahorse_ext = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in SEAHORSE_MDS_Colnames]
    exclude.extend(exclude_metrics)
    exclude.extend(exclude_seahorse_ext)
    exclude.extend(exclude_seahorse)

    # include some seahorse metrics
    include_seahorse = ["Attribution", "Main Ideas", "Attribution (Avg. Score)", "Main Ideas (Avg. Score)"]
    # filter out the seahorse metrics that are not in the dataframe
    include_seahorse = [x for x in include_seahorse if x in df.columns]
    # remove them from the exclude list
    exclude = [x for x in exclude if x not in include_seahorse]

    metric_names = [col for col in list(df.columns) if col not in exclude]
    metric_ordering_all = {
        "R-1": True,
        "R-2": True,
        "R-L": True,
        "BertScore Precision": True,
        "BertScore Recall": True,
        "BertScore F1": True,
        "Coverage": False,
        "Coverage (Prompt)": False,
        "Density": False,
        "Density (Prompt)": False,
        "Compression (Article)": False,
        "Compression": False,
        "Compression (Prompt)": False,
        "Compression (Full)": False,
        "Attribution": True,
        "Main Ideas": True,
        "Attribution (Avg. Score)": True,
        "Main Ideas (Avg. Score)": True,
    }

    # sort metric-names ascending
    metric_names.sort()

    return metric_names, {metric_name: metric_ordering_all[metric_name] for metric_name in metric_names}


"""
Generic Experiment Setups
"""
# "Preprocessing (N-Shot) (#Tokens)",
# "Method (N-Shot) (#Tokens)",
# "Method (N-Shot) (Source)",
# "Method (#Tokens) (Source)",
# "(N-Shot) (#Tokens) (Source)"
fewshot_experiment__experimental_setup = {
    "groupByList": ["Preprocessing (N-Shot) (#Tokens)", "Size subset"],
    "groupByListVariants": [
        ["Method (N-Shot)", "Preprocessing Parameters"],
        ["Method", "N-Shot"],
        ["Example Source + N-Shot", "Method"],
        ["Method (N-Shot) (#Tokens)", "Size subset"],
        ["Method (N-Shot) (Source)", "Size subset"],
        ["Method (#Tokens) (Source)", "Size subset"],
        ["(N-Shot) (#Tokens) (Source)", "Preprocessing Method"],
        ["Preprocessing + N-Shot", "Preprocessing Parameters"],
        ["Preprocessing + N-Shot", "Dataset Annotation"],
    ],
    "special_metric_catplots": [
        {
            "row": "Size subset",
            "col": "Preprocessing Parameters",
            "x": "Method",
            "hue": "N-Shot",
            "is_baseline_row": lambda row: row["Method"] == "Baseline (truncated)",
        },
        {
            "row": "Size subset",
            "col": "N-Shot",
            "x": "Method",
            "hue": "Preprocessing Parameters",
            "is_baseline_row": lambda row: row["Method"] == "Baseline (truncated)",
        },
        {
            "row": "Size subset",
            "col": "Preprocessing Parameters",
            "x": "N-Shot",
            "hue": "Method",
            "is_baseline_row": lambda row: row["Method"] == "Baseline (truncated)",
        },
        {
            "row": "Size subset",
            "col": "N-Shot",
            "x": "Preprocessing Parameters",
            "hue": "Method",
            "is_baseline_row": lambda row: row["Method"] == "Baseline (truncated)",
        },
        {
            "row": "Size subset",
            "col": "Preprocessing Parameters",
            "x": "Method (N-Shot)",
            "hue": "Dataset Annotation",
            "is_baseline_row": lambda row: row["Method"] == "Baseline (truncated)",
        },
    ],
    "models": ["meta-llama/Llama-2-70b-chat-hf"],
    "datasets": ["Wikinews"],
    "prompts_bag_alias": "few-shot-experiment-full",
}

fewshot_experiment__experimental_setup_MULTINEWS = {
    "groupByList": ["Preprocessing + N-Shot", "Preprocessing Parameters"],
    "groupByListVariants": [
        ["Preprocessing Parameters", "Preprocessing + N-Shot"],
        ["Preprocessing + N-Shot", "Dataset Annotation"],
        ["Dataset Annotation", "Preprocessing + N-Shot"],
    ],
    "models": ["meta-llama/Llama-2-70b-chat-hf"],
    "datasets": ["Multinews"],
    "prompts_bag_alias": "few-shot-experiment-full",
}

summchain__experimental_setup = {
    "groupByList": ["Method", "Size subset"],
    "models": ["meta-llama/Llama-2-70b-chat-hf"],
    "groupByListVariants": [
        ["Method (N-Shot)", "Size subset"],
        ["Method", "N-Shot"],
    ],
    "special_metric_catplots": [
        {
            "row": "Size subset",
            "col": "",
            "x": "Method",
            "hue": "N-Shot",
            "is_baseline_row": lambda row: row["Method"] in ["Baseline (truncated)", "Baseline (trunc.)", "Baseline"],
        },
        {
            "row": "Size subset",
            "col": "",
            "x": "N-Shot",
            "hue": "Method",
            "is_baseline_row": lambda row: row["Method"] in ["Baseline (truncated)", "Baseline (trunc.)", "Baseline"],
        },
        {
            "row": "N-Shot",
            "col": "",
            "x": "Size subset",
            "hue": "Method",
            "is_baseline_row": lambda row: row["Method"] in ["Baseline (truncated)", "Baseline (trunc.)", "Baseline"],
        },
    ],
    "datasets": ["Wikinews"],
    "prompts_bag_alias": "mds-baseline",
}

base_experiment_groupByList_variants = [
    ["Model", "Prompt Name"],
    ["Prompt Name", "Has Prompt-Summary Separator"],
    ["Model", "Has Prompt-Summary Separator"],
    ["Model", "Prompt Language"],
    ["Model", "Prompt Lang. (Separator)"],
]

"""
    SELECT THE EXPERIMENT TO BUILD THE REPORT ON HERE
"""
# TODO
experiment_name = "mds-summarization-chain-comparison-presentation"
# few-shot-experiment-20Min-vs-Wiki-Examples
# few-shot-experiment-clustering
# few-shot-experiment-clustering-BEST
# few-shot-experiment-distMMR
# few-shot-experiment-distMMR-BEST
# few-shot-experiment-baselines
# few-shot-experiment-main

# mds-summarization-chain
# mds-summarization-chain-comparison

# mds-2stage-experiment
# mds-baseline-german
# mds-2stage-verus-intermediate
# mds-summarization-chain
# mds-prefix-experiment
# mds-ordered-chunks-initial-overview

# prompt-experiment-large-llama2-vs-leolm
# mds-cluster-chunks-vs-2stage-experiment
# mds-cluster-chunks-experiment

# DONE
# base-experiment
# base-experiment-separator-and-german-only
# base-experiment-temperature
# least-to-most-prompting-stage2
# prompt-experiment-large
# prompt-experiment-large-all-llama-comparison
# prompt-experiment-large-all-llama-comparison-good-prompts


"""
    ADD NEW EXPERIMENTS HERE
"""
# read the json file into a dictionary
experiment_config = {
    "all-data-full-preprocessing": {
        "groupByList": ["Model", "Prompt ID"],
        "models": [
            # "meta-llama/Llama-2-7b-chat-hf",
            # "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            "bigscience/bloomz-7b1-mt",
            "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            "garage-bAInd/Platypus2-70B-instruct",
            # "gpt-4",
            # "palm2",
            "LeoLM/leo-hessianai-7b-chat",
            "LeoLM/leo-hessianai-13b-chat",
        ],
        "groupByListVariants": [],
        "datasets": ["20Minuten", "Wikinews"]
    },
    "few-shot-initial": {
        "groupByList": ["N-Shot", "Prompt ID"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Name", "N-Shot"],
            ["N-Shot", "Prompt Name"],
        ],
        "datasets": ["20Minuten"]
    },
    "few-shot-experiment-20Min-vs-Wiki-Examples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-baselines": fewshot_experiment__experimental_setup,
    "few-shot-experiment-full": fewshot_experiment__experimental_setup,
    "few-shot-experiment-full-20MinutesExamples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-full-WikinewsExamples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-full-1024": fewshot_experiment__experimental_setup,
    "few-shot-experiment-full-1536": fewshot_experiment__experimental_setup,
    "few-shot-experiment-full-2048": fewshot_experiment__experimental_setup,
    "few-shot-experiment-clustDist": fewshot_experiment__experimental_setup,
    "few-shot-experiment-clustering": fewshot_experiment__experimental_setup,
    "few-shot-experiment-clustering-1536": fewshot_experiment__experimental_setup,
    "few-shot-experiment-clustering-BEST": fewshot_experiment__experimental_setup,
    "few-shot-experiment-clustering-20MinutesExamples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-clustering-WikinewsExamples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-distMMR": fewshot_experiment__experimental_setup,
    "few-shot-experiment-distMMR-BEST": fewshot_experiment__experimental_setup,
    "few-shot-experiment-distMMR-20MinutesExamples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-distMMR-WikinewsExamples": fewshot_experiment__experimental_setup,
    "few-shot-experiment-COMPARISONS": fewshot_experiment__experimental_setup,
    "few-shot-experiment-SELECTION": {},  # TODO
    "few-shot-experiment-main-sizeComparison": fewshot_experiment__experimental_setup,
    "few-shot-experiment-main-1024": fewshot_experiment__experimental_setup,
    "few-shot-experiment-main-1536": fewshot_experiment__experimental_setup,
    "few-shot-experiment-main": fewshot_experiment__experimental_setup,
    "few-shot-experiment-multinews": fewshot_experiment__experimental_setup_MULTINEWS,
    "mds-baseline": {
        "groupByList": ["Prompt Name", "Dataset Annotation (incl. article size)"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Name", "Dataset Annotation"],
            ["Prompt Desc. [ID]", "Dataset Annotation"],
        ],
        "datasets": ["Wikinews"]
    },
    "mds-baseline-german": {
        "groupByList": ["Prompt Name", "Dataset Annotation (incl. article size)"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Name", "Dataset Annotation"],
            ["Prompt Desc. [ID]", "Dataset Annotation"],
        ],
        "datasets": ["Wikinews"],
        "prompts_bag_alias": "mds-baseline",
    },
    "mds-summarization-chain": summchain__experimental_setup,
    "mds-summarization-chain-comparison": summchain__experimental_setup,
    "mds-summarization-chain-comparison-presentation": summchain__experimental_setup,
    "mds-shuffling-and-annotation-experiment": {
        "groupByList": ["Prompt Name", "Dataset Annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Preprocessing Method", "Preprocessing Order"],
            ["Dataset Annotation", "Size subset"],
            ["Prompt Name", "Dataset Annotation (incl. article size)"],
            ["Prompt Desc. [ID]", "Dataset Annotation"],
        ],
        "datasets": ["Wikinews"]
    },
    "mds-prefix-experiment": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-2stage-experiment": {
        "groupByList": ["Preprocessing Parameters", "Size subset"],  # preprocessing_parameters
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Dataset Annotation", "Size subset"],
            ["Dataset Annotation", "Large Article"],
        ],
        "datasets": ["Wikinews"],
        "additional_prompts": [2, 41]
    },
    "mds-2stage-verus-intermediate": {
        "groupByList": ["Dataset Annotation (Prompt Name)", "Large Article"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [],
        "datasets": ["Wikinews"],
        "additional_prompts": [2, 41],
        "prompts_bag_alias": "mds-2stage-experiment",
    },
    "mds-cluster-chunks-experiment": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "groupByListVariants": [
            ["Preprocessing Method", "Preprocessing Parameters"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-cluster-chunks-experiment-short": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "groupByListVariants": [
            ["Preprocessing Method", "Preprocessing Parameters"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-cluster-chunks-vs-sentence-chunks-experiment": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "groupByListVariants": [
            ["Preprocessing Method", "Preprocessing Parameters"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-cluster-chunks-vs-2stage-experiment": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "groupByListVariants": [
            ["Prompt Description", "Dataset Annotation"],
            ["Preprocessing Method", "Preprocessing Parameters"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-sentence-chunks-experiment": {
        "groupByList": ["Preprocessing Method", "Preprocessing Parameters"],
        "groupByListVariants": [
            ["Prompt Name", "Dataset Annotation"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-ordered-chunks-initial-overview": {
        "groupByList": ["Preprocessing Method", "Preprocessing Parameters"],
        "groupByListVariants": [
            ["Prompt Name", "Dataset Annotation"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-ordered-chunks-initial-1sentence": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "groupByListVariants": [
            ["Dataset Annotation", "Prompt Description"],  # VERY GOOD PLOT DIMENSIONS
            ["Prompt Description", "Dataset Annotation"],
            ["Preprocessing Method", "Preprocessing Parameters"],
            ["Preprocessing Method", "Preprocessing Order"],
        ],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "base-experiment": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            "bigscience/bloomz-7b1-mt",
        ],
        "groupByListVariants": base_experiment_groupByList_variants,
        "datasets": ["20Minuten"]
    },
    "base-experiment-shard1": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            "bigscience/bloomz-7b1-mt",
        ],
        "groupByListVariants": base_experiment_groupByList_variants,
        "datasets": ["20Minuten"]
    },
    "base-experiment-shard2": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            "bigscience/bloomz-7b1-mt",
        ],
        "groupByListVariants": base_experiment_groupByList_variants,
        "datasets": ["20Minuten"]
    },
    "base-experiment-separator-and-german-only": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            "bigscience/bloomz-7b1-mt",
        ],
        "groupByListVariants": base_experiment_groupByList_variants,
        "datasets": ["20Minuten"],
        "prompts_bag_alias": "base-experiment",
    },
    "base-experiment-temperature": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            # "tiiuae/falcon-40b-instruct",
            "bigscience/bloomz-7b1-mt"
        ],
        "groupByListVariants": [
            ["Prompt Name", "Model"],
            ["Prompt Name", "Temperature"],
            ["Prompt Desc. [ID]", "Model"],
            ["Prompt Desc. [ID]", "Temperature"],
            ["Model", "Temperature"],
        ],
        "datasets": ["20Minuten"],
        "prompts_bag_alias": "base-experiment",
    },
    "base-experiment-temperature-basic-1-S": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            # "tiiuae/falcon-40b-instruct",
            "bigscience/bloomz-7b1-mt"
        ],
        "groupByListVariants": [
            ["Prompt Name", "Model"],
            ["Prompt Name", "Temperature"],
            ["Prompt Desc. [ID]", "Model"],
            ["Prompt Desc. [ID]", "Temperature"],
            ["Model", "Temperature"],
        ],
        "datasets": ["20Minuten"],
        "prompts_bag_alias": "base-experiment",
    },
    "experiment-sizes": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            "garage-bAInd/Platypus2-70B-instruct",
        ],
        "datasets": ["20Minuten"]
    },
    "least-to-most-prompting-stage1": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"]
    },
    "least-to-most-prompting-stage1+2": {
        "groupByList": ["Prompt ID", "Dataset Annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"],
        "additional_prompts": [21, 22, 30, 31, 32, 33]
    },
    "least-to-most-prompting-stage2": {
        "groupByList": ["Prompt Name", "Dataset Annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Description", "Dataset Annotation"],
            ["Prompt Description", "Dataset Annotation (Is Bullet Point Summary)"],
            ["Dataset Annotation (Prompt Name)", "Prompt Description"],
        ],
        "datasets": ["20Minuten"],
        "additional_prompts": [21, 22, 30, 31, 32, 33]
    },
    "versions-experiment": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "gpt-4",
            "palm2",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            "garage-bAInd/Platypus2-70B-instruct",
        ],
        "groupByListVariants": [],
        "datasets": ["20Minuten"]
    },
    "versions-experiment-all": {
        "groupByList": ["Prompt ID", "Model"],
        "models": [
            "gpt-4",
            "palm2",
            "palm2-oct",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            "garage-bAInd/Platypus2-70B-instruct",
        ],
        "groupByListVariants": [
            ["Prompt Name", "Model"],
        ],
        "datasets": ["20Minuten"]
    },
    "versions-experiment-llama2-gpt4-palm2": {
        "groupByList": ["Prompt ID", "Model"],
        "models": [
            "gpt-4",
            "palm2",
            # "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            # "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            # "garage-bAInd/Platypus2-70B-instruct",
        ],
        "datasets": ["20Minuten"]
    },
    "versions-experiment-llama2-gpt4-palm2-prompts-2-4": {
        "groupByList": ["Prompt ID", "Model"],
        "models": [
            "gpt-4",
            "palm2",
            # "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            # "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            # "garage-bAInd/Platypus2-70B-instruct",
        ],
        "groupByListVariants": [
            ["Prompt Description", "Model"],
            ["Prompt Desc. [ID]", "Model"],
        ],
        "datasets": ["20Minuten"]
    },
    "versions-experiment-gpt4-only": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["gpt-4"],
        "datasets": ["20Minuten"]
    },
    "empty-experiment": {
        "groupByList": ["Task Name", "Prompt ID"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
        ],
        "datasets": ["20minSmol"]
    },
    "prompt-experiment-large": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Desc. [ID]", "Model"],
        ],
        "datasets": ["20Minuten"]
    },
    "prompt-experiment-large-all-llama-comparison": {
        "groupByList": ["Prompt ID", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ],
        "groupByListVariants": [
            ["Prompt Name", "Model"],
            ["Prompt Desc. [ID]", "Model"],
        ],
        "datasets": ["20Minuten"],
        "prompts_bag_alias": "prompt-experiment-large",
    },
    "prompt-experiment-large-all-llama-comparison-good-prompts": {
        "groupByList": ["Prompt ID", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ],
        "groupByListVariants": [
            ["Prompt Name", "Model"],
            ["Model", "Prompt Name"],
            ["Prompt Desc. [ID]", "Model"],
        ],
        "datasets": ["20Minuten"],
        "prompts_bag_alias": "prompt-experiment-large",
    },
    "prompt-experiment-large-basic": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"]
    },
    "prompt-experiment-large-llama2-vs-leolm": {
        "groupByList": ["Prompt Name", "Model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "LeoLM/leo-hessianai-7b-chat",
            "LeoLM/leo-hessianai-13b-chat",
        ],
        "groupByListVariants": [
            ["Model", "Prompt Name"],
        ],
        "datasets": ["20Minuten"],
        "prompts_bag_alias": "prompt-experiment-large",
    },
    "prompt-experiment-large-NZZ": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Description", "Has Prompt-Summary Separator"],
            ["Prompt Desc. [ID]", "Has Prompt-Summary Separator"],
        ],
        "datasets": ["20Minuten"]
    },
    "prompt-experiment-large-output-size": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "groupByListVariants": [
            ["Prompt Variant", "Model"],
        ],
        "datasets": ["20Minuten"]
    },
    "prompt-experiment-large-variants-only": {
        "groupByList": ["Prompt ID", "Model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"]
    },
    "prompt-experiment-large-vs-llama2-gpt4-palm2": {
        "groupByList": ["Prompt ID", "Model"],
        "models": [
            "gpt-4",
            "palm2",
            # "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            # "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            # "garage-bAInd/Platypus2-70B-instruct",
        ],
        "groupByListVariants": [
            ["Prompt Desc. [ID]", "Model"],
            ["Prompt Description", "Model"],
        ],
        "datasets": ["20Minuten"]
    },
    "tmp": {}
}

RESULTS_BASE_PATH = 'results_bag'

groupByList = experiment_config[experiment_name]["groupByList"]
models = experiment_config[experiment_name]["models"]
datasets = experiment_config[experiment_name]["datasets"]
additional_prompts = experiment_config[experiment_name].get("additional_prompts", [])
shortNames = {
    "gpt-4": "GPT 4",
    "palm2": "PaLM 2",
    "palm2-oct": "PaLM 2 (Oct.)",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2  7b",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-2 13b",
    "meta-llama/Llama-2-70b-chat-hf": "Llama-2 70b",
    "tiiuae/falcon-7b-instruct": "Falcon  7b",
    "tiiuae/falcon-40b-instruct": "Falcon 40b",
    "bigscience/bloomz-7b1-mt": "BloomZ  7b",
    "fangloveskari/ORCA_LLaMA_70B_QLoRA": "OrcaLlama2 70B",
    "garage-bAInd/Platypus2-70B-instruct": "Platypus2 70B",
    "LeoLM/leo-hessianai-7b": "LeoLM  7B (Pretrained)",
    "LeoLM/leo-hessianai-13b": "LeoLM 13B (Pretrained)",
    "LeoLM/leo-hessianai-7b-chat": "LeoLM  7B",
    "LeoLM/leo-hessianai-13b-chat": "LeoLM 13B",
}
model_type_by_shortname = {
    "GPT 4": "GPT",
    "PaLM 2": "PaLM2",
    "PaLM 2 (Oct.)": "PaLM2",
    "Llama-2  7b": "Llama2",
    "Llama-2 13b": "Llama2",
    "Llama-2 70b": "Llama2",
    "Falcon  7b": "Falcon",
    "Falcon 40b": "Falcon",
    "BloomZ  7b": "BloomZ",
    "OrcaLlama2 70B": "OrcaLlama2",
    "Platypus2 70B": "Platypus2",
    "LeoLM  7B (Pretrained)": "LeoLM",
    "LeoLM 13B (Pretrained)": "LeoLM",
    "LeoLM  7B": "LeoLM",
    "LeoLM 13B": "LeoLM",
}
datasetLanguage = {
    "Wikinews": "de",
    "20Minuten": "de",
    "Multinews": "en",
}
shortLangToLanguage = {
    "de": "german",
    "en": "english",
}
datasetNameMap = {
    "MultinewsTrunc3584": "Multinews",
    "MultiCD040SSimDyn1024": "Multinews",
    "MultiCD040SSimDyn1536": "Multinews",
    "Wikinews": "Wikinews",
    "WikinewsTrunc3584": "Wikinews",
    "WikinewsTrunc3584SUBS": "Wikinews",
    "WikinewsSummarizationChain": "Wikinews",
    "WikinewsSummarizationChainAblation": "Wikinews",
    "WikinewsCDS2i0": "Wikinews",
    "WikinewsCDS4i10": "Wikinews",
    "WikinewsCDS4i11": "Wikinews",
    "WikinewsCDS4i1": "Wikinews",
    "WikinewsCDS4i2": "Wikinews",
    "WikinewsCDS4i3": "Wikinews",
    "WikinewsCDS4i4": "Wikinews",
    "WikinewsCDS4i5": "Wikinews",
    "WikinewsCDS4i6": "Wikinews",
    "WikinewsCDS4i7": "Wikinews",
    "WikinewsCDS4i8": "Wikinews",
    "WikinewsCDS4i9": "Wikinews",
    "WikinewsClustDistS2": "Wikinews",
    "WikinewsS2": "Wikinews",
    "20Minuten": "20Minuten",
    "20minSmol": "20Minuten",
    "20minTS250": "20Minuten",
    "20min0": "20Minuten",
    "20min1": "20Minuten",
    "20min2": "20Minuten",
    "20min3": "20Minuten",
    "20minLtm2p22S": "20Minuten",
    "20minLtm2p22E": "20Minuten",
    "20minLtm2p31S": "20Minuten",
    "20minLtm2p31E": "20Minuten",
    "20minLtm2p33S": "20Minuten",
    "20minLtm2p33E": "20Minuten",
    "WikinewsClean": "Wikinews",
    "WikinewsSLR": "Wikinews",
    "WikinewsSL": "Wikinews",
    "WikinewsSimple": "Wikinews",
    "WikinewsSimpleS": "Wikinews",
    "WikinewsSimpleA": "Wikinews",
    "WikinewsSimpleAS": "Wikinews",
    "WikinewsSC32": "Wikinews",
    "WikinewsSC64": "Wikinews",
    "WikinewsSC128": "Wikinews",
    "WikinewsSC256": "Wikinews",
    "WikinewsSC512": "Wikinews",
    "WikinewsSCS2": "Wikinews",
    "WikinewsSCS4": "Wikinews",
    "WikinewsSCS8": "Wikinews",
    "WikinewsSCS16": "Wikinews",
    "WikinewsSCS32": "Wikinews",
    "WikinewsSplit": "Wikinews",
    "WikinewsSplitS2O": "Wikinews",
    "WikinewsSplitS2S": "Wikinews",
    "WikinewsSplitS2OP41": "Wikinews",
    "WikinewsSplitS2SP41": "Wikinews",
    "WikinewsClust1C": "Wikinews",
    "WikinewsClust1O": "Wikinews",
    "WikinewsClust1R": "Wikinews",
    "WikinewsClust5C": "Wikinews",
    "WikinewsClust5O": "Wikinews",
    "WikinewsClust5R": "Wikinews",
    "WikinewsClust10C": "Wikinews",
    "WikinewsClust10O": "Wikinews",
    "WikinewsClust10R": "Wikinews",
    "WikinewsSent1L00": "Wikinews",
    "WikinewsSent1L05": "Wikinews",
    "WikinewsSent3L00": "Wikinews",
    "WikinewsSent3L05": "Wikinews",
    "WikinewsSent5L00": "Wikinews",
    "WikinewsSent5L05": "Wikinews",
    "WikinewsSent10L00": "Wikinews",
    "WikinewsSent10L05": "Wikinews",
    "WikiCh1024": "Wikinews",
    "WikiCh1536": "Wikinews",
    "WikiCl0N1024": "Wikinews",
    "WikiCl0N1536": "Wikinews",
    "WikiCl0N2048": "Wikinews",
    "WikiCl1N21024": "Wikinews",
    "WikiCl1N21536": "Wikinews",
    "WikiCl1N22048": "Wikinews",
    "WikiCl2S21024": "Wikinews",
    "WikiCl2S21536": "Wikinews",
    "WikiCl2S22048": "Wikinews",
    "WikiCl1SW1024": "Wikinews",
    "WikiCl1SW1536": "Wikinews",
    "WikiCl2SW1024": "Wikinews",
    "WikiCl2SW1536": "Wikinews",
    "WikiDi0S1024": "Wikinews",
    "WikiDi0S1536": "Wikinews",
    "WikiDi1S21024": "Wikinews",
    "WikiDi1S21536": "Wikinews",
    "WikiDi1SW1024": "Wikinews",
    "WikiDi1SW1536": "Wikinews",
    "WikiDi2S21024": "Wikinews",
    "WikiDi2S21536": "Wikinews",
    "WikiDi2SW1024": "Wikinews",
    "WikiLe1024": "Wikinews",
    "WikiLe1536": "Wikinews",
    "WikiLe1S21024": "Wikinews",
    "WikiLe1S21536": "Wikinews",
    "WikiLe1SW1536": "Wikinews",
    "WikiDi2SW1536": "Wikinews",
    "WikiRa1024": "Wikinews",
    "WikiRa1536": "Wikinews",
    "WikiRa1S21024": "Wikinews",
    "WikiRa1S21536": "Wikinews",
    "WikiRa1SW1024": "Wikinews",
    "WikiRa1SW1536": "Wikinews",
    "WikiCD040SSimDyn1024": "Wikinews",
    "WikiCD040SSimDyn1536": "Wikinews",
    "WikiCD041SSimDyn1024": "Wikinews",
    "WikiCD042SSimDyn1024": "Wikinews",
    "WikiCl0SSimDyn1536": "Wikinews",
    "WikiCD041SSimDyn1536": "Wikinews",
    "WikiCD042SSimDyn1536": "Wikinews",
    "WikiCD043SSimDyn1536": "Wikinews",
    "WikiCD050SSimDyn1536": "Wikinews",
    "WikiCD051SSimDyn1536": "Wikinews",
    "WikiCD052SSimDyn1536": "Wikinews",
    "WikiCD053SSimDyn1536": "Wikinews",
    "WikiCD060SSimDyn1536": "Wikinews",
    "WikiCD061SSimDyn1536": "Wikinews",
    "WikiCD062SSimDyn1536": "Wikinews",
    "WikiCD063SSimDyn1536": "Wikinews",
    "WikiCl1SSimDyn1536": "Wikinews",
    "WikiCl2SSimDyn1536": "Wikinews",
    "WikiChN1024": "Wikinews",
    "WikiChN1536": "Wikinews",
    "WikiChN1S1024": "Wikinews",
    "WikiChN1S1536": "Wikinews",
    "WikiChN2S1024": "Wikinews",
    "WikiChN2S1536": "Wikinews",
    "WikiCl1SSimDynW2048": "Wikinews",
    "WikiCl2SSimDynW2048": "Wikinews",
    "WikiDi090S1024": "Wikinews",
    "WikiDi090S1536": "Wikinews",
    "WikiDi091SW1024": "Wikinews",
    "WikiDi091SW1536": "Wikinews",
    "WikiDi092SW1024": "Wikinews",
    "WikiDi092SW1536": "Wikinews",
}
datasetAnnotationMap = {
    "20Minuten": "20Minuten, 250 samples",
    "20minTS250": "20Minuten, 250 samples",
    "20min0": "20Minuten, shard 1",
    "20min1": "20Minuten, shard 2",
    "20min2": "20Minuten, shard 3",
    "20min3": "20Minuten, shard 4",
    "20minLtm2p22S": "20Min LtM, Bullet",
    "20minLtm2p22E": "20Min LtM, Bullet",
    "20minLtm2p31S": "20Min LtM, Instr.-Q",
    "20minLtm2p31E": "20Min LtM, Instr.-Q",
    "20minLtm2p33S": "20Min LtM, Q-Instr.",
    "20minLtm2p33E": "20Min LtM, Q-Instr.",
    "MultinewsTrunc3584": "basic,\ntruncated to 3584 tokens",
    "MultiCD040SSimDyn1024": "Ex.Src: Multinews",
    "MultiCD040SSimDyn1536": "Ex.Src: Multinews",
    "Wikinews": "full articles,\noriginal order",
    "WikinewsClean": "manual cleaning,\nfull artices,\noriginal order",
    "WikinewsTrunc3584": "manual cleaning,\ntruncated to 3584 tokens,\noriginal order",
    "WikinewsSummarizationChain": "Summarization Chain",
    "WikinewsSummarizationChainAblation": "Summarization Chain (Ablation)",
    "WikinewsSimple": "no annotation,\noriginal order",
    "WikinewsSimpleS": "no annotation,\nrandom order",
    "WikinewsSimpleA": "article idx ann.,\noriginal order",
    "WikinewsSimpleAS": "article idx ann.,\nrandom order",
    "WikinewsSC32": " 32 word prefix",
    "WikinewsSC64": " 64 word prefix",
    "WikinewsSC128": "128 word prefix",
    "WikinewsSC256": "256 word prefix",
    "WikinewsSC512": "512 word prefix",
    "WikinewsSCS2": " 2 sentence prefix",
    "WikinewsSCS4": " 4 sentence prefix",
    "WikinewsSCS8": " 8 sentence prefix",
    "WikinewsSCS16": "16 sentence prefix",
    "WikinewsSCS32": "32 sentence prefix",
    "WikinewsSplit": "2-stage, stage 1",
    "WikinewsSplitS2O": "2-stage, SDS-2,\noriginal order",
    "WikinewsSplitS2S": "2-stage, SDS-2,\nrandom order",
    "WikinewsSplitS2OP41": "2-stage, SDS-4,\noriginal order",
    "WikinewsSplitS2SP41": "2-stage, SDS-4,\nrandom order",
    "WikinewsClust1C": "Cluster,\n 1 Sentence,\ncluster size order",
    "WikinewsClust1O": "Cluster,\n 1 Sentence,\noriginal order",
    "WikinewsClust1R": "Cluster,\n 1 Sentence,\nrandom order",
    "WikinewsClust5C": "Cluster,\n 5 Sentences,\ncluster size order",
    "WikinewsClust5O": "Cluster,\n 5 Sentences,\noriginal order",
    "WikinewsClust5R": "Cluster,\n 5 Sentences,\nrandom order",
    "WikinewsClust10C": "Cluster,\n10 Sentences,\ncluster size order",
    "WikinewsClust10O": "Cluster,\n10 Sentences,\noriginal order",
    "WikinewsClust10R": "Cluster,\n10 Sentences,\nrandom order",
    "WikinewsSent1L00": "Embedding Similarity,\n 1 Sentence,\nMMR, lambda 0.0",
    "WikinewsSent1L05": "Embedding Similarity,\n 1 Sentence,\nMMR, lambda 0.5",
    "WikinewsSent3L00": "Embedding Similarity,\n 3 Sentences,\nMMR, lambda 0.0",
    "WikinewsSent3L05": "Embedding Similarity,\n 3 Sentences,\nMMR, lambda 0.5",
    "WikinewsSent5L00": "Embedding Similarity,\n 5 Sentences,\nMMR, lambda 0.0",
    "WikinewsSent5L05": "Embedding Similarity,\n 5 Sentences,\nMMR, lambda 0.5",
    "WikinewsSent10L00": "Embedding Similarity,\n10 Sentences,\nMMR, lambda 0.0",
    "WikinewsSent10L05": "Embedding Similarity,\n10 Sentences,\nMMR, lambda 0.5",
    "WikiCh1024": "-",
    "WikiCh1536": "-",
    "WikiLe1024": "-",
    "WikiLe1536": "-",
    "WikiRa1024": "-",
    "WikiRa1536": "-",
    "WikiCl0N1024": "-",
    "WikiCl0N1536": "-",
    "WikiCl0N2048": "-",
    "WikiCl1N21024": "Ex.Src: 20Minuten",
    "WikiCl1N21536": "Ex.Src: 20Minuten",
    "WikiCl1N22048": "Ex.Src: 20Minuten",
    "WikiCl2S21024": "Ex.Src: 20Minuten",
    "WikiCl2S21536": "Ex.Src: 20Minuten",
    "WikiCl2S22048": "Ex.Src: 20Minuten",
    "WikiCl1SW1024": "Ex.Src: Wikinews",
    "WikiCl1SW1536": "Ex.Src: Wikinews",
    "WikiCl2SW1024": "Ex.Src: Wikinews",
    "WikiCl2SW1536": "Ex.Src: Wikinews",
    "WikiDi0S1024": "-",
    "WikiDi0S1536": "-",
    "WikiDi1S21024": "Ex.Src: 20Minuten",
    "WikiDi1S21536": "Ex.Src: 20Minuten",
    "WikiDi2S21024": "Ex.Src: 20Minuten",
    "WikiDi2S21536": "Ex.Src: 20Minuten",
    "WikiDi1SW1024": "Ex.Src: Wikinews",
    "WikiDi1SW1536": "Ex.Src: Wikinews",
    "WikiDi2SW1024": "Ex.Src: Wikinews",
    "WikiLe1S21024": "Ex.Src: 20Minuten",
    "WikiLe1S21536": "Ex.Src: 20Minuten",
    "WikiLe1SW1536": "Ex.Src: Wikinews",
    "WikiDi2SW1536": "Ex.Src: Wikinews",
    "WikiRa1S21024": "Ex.Src: 20Minuten",
    "WikiRa1S21536": "Ex.Src: 20Minuten",
    "WikiRa1SW1024": "Ex.Src: Wikinews",
    "WikiRa1SW1536": "Ex.Src: Wikinews",
    "WikiCD040SSimDyn1024": "-",
    "WikiCD040SSimDyn1536": "-",
    "WikiCD041SSimDyn1024": "Ex.Src: Wikinews",
    "WikiCD042SSimDyn1024": "Ex.Src: Wikinews",
    "WikiCl0SSimDyn1536": "-",
    "WikiCD041SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD042SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD043SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD050SSimDyn1536": "-",
    "WikiCD051SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD052SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD053SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD060SSimDyn1536": "-",
    "WikiCD061SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD062SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCD063SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCl1SSimDyn1536": "Ex.Src: Wikinews",
    "WikiCl2SSimDyn1536": "Ex.Src: Wikinews",
    "WikiChN1024": "-",
    "WikiChN1536": "-",
    "WikiChN1S1024": "Ex.Src: Wikinews",
    "WikiChN1S1536": "Ex.Src: Wikinews",
    "WikiChN2S1024": "Ex.Src: Wikinews",
    "WikiChN2S1536": "Ex.Src: Wikinews",
    "WikiCl1SSimDynW2048": "Ex.Src: Wikinews",
    "WikiCl2SSimDynW2048": "Ex.Src: Wikinews",
    "WikiDi090S1024": "-",
    "WikiDi090S1536": "-",
    "WikiDi091SW1024": "Ex.Src: Wikinews",
    "WikiDi091SW1536": "Ex.Src: Wikinews",
    "WikiDi092SW1024": "Ex.Src: Wikinews",
    "WikiDi092SW1536": "Ex.Src: Wikinews",
}


def example_source_abbreviation(example_source) -> str:
    abbreviations = {
        "-": "-",
        "Ex.Src: Wikinews": "Wiki",
        "Ex.Src: 20Minuten": "20Min",
    }
    if example_source in abbreviations:
        return abbreviations[example_source]
    return example_source


preprocessing_method = {
    "20Minuten": "-",
    "MultinewsTrunc3584": "-",
    "MultiCD040SSimDyn1024": "-",
    "MultiCD040SSimDyn1536": "-",
    "Wikinews": "-",
    "WikinewsTrunc3584": "manual cleaning",
    "WikinewsClean": "manual cleaning",
    "WikinewsSummarizationChain": "Summarization Chain",
    "WikinewsSummarizationChainAblation": "Summarization Chain (Ablation)",
    "WikinewsSimple": "-",
    "WikinewsSimpleS": "-",
    "WikinewsSimpleA": "article idx ann.",
    "WikinewsSimpleAS": "article idx ann.",
    "WikinewsSC32": "token prefix",
    "WikinewsSC64": "token prefix",
    "WikinewsSC128": "token prefix",
    "WikinewsSC256": "token prefix",
    "WikinewsSC512": "token prefix",
    "WikinewsSCS2": "sentence prefix",
    "WikinewsSCS4": "sentence prefix",
    "WikinewsSCS8": "sentence prefix",
    "WikinewsSCS16": "sentence prefix",
    "WikinewsSCS32": "sentence prefix",
    "WikinewsSplit": "2-stage, initial",
    "WikinewsSplitS2O": "2-stage summary",
    "WikinewsSplitS2S": "2-stage summary",
    "WikinewsSplitS2OP41": "2-stage summary",
    "WikinewsSplitS2SP41": "2-stage summary",
    "WikinewsClust1C": "Cluster",
    "WikinewsClust1O": "Cluster",
    "WikinewsClust1R": "Cluster",
    "WikinewsClust5C": "Cluster",
    "WikinewsClust5O": "Cluster",
    "WikinewsClust5R": "Cluster",
    "WikinewsClust10C": "Cluster",
    "WikinewsClust10O": "Cluster",
    "WikinewsClust10R": "Cluster",
    "WikinewsSent1L00": "Embedding Similarity",
    "WikinewsSent1L05": "Embedding Similarity",
    "WikinewsSent3L00": "Embedding Similarity",
    "WikinewsSent3L05": "Embedding Similarity",
    "WikinewsSent5L00": "Embedding Similarity",
    "WikinewsSent5L05": "Embedding Similarity",
    "WikinewsSent10L00": "Embedding Similarity",
    "WikinewsSent10L05": "Embedding Similarity",
    "WikiCh1024": "Oracle",
    "WikiCh1536": "Oracle",
    "WikiLe1024": "Lead",
    "WikiLe1536": "Lead",
    "WikiRa1024": "Random",
    "WikiRa1536": "Random",
    "WikiLe1SW1536": "Lead",
    "WikiDi2SW1536": "Distance-MMR (0.75)",
    "WikiCl0N1024": "Clustering",
    "WikiCl0N1536": "Clustering",
    "WikiCl0N2048": "Clustering",
    "WikiCl1N21024": "Clustering",
    "WikiCl1N21536": "Clustering",
    "WikiCl1N22048": "Clustering",
    "WikiCl2S21024": "Clustering",
    "WikiCl2S21536": "Clustering",
    "WikiCl2S22048": "Clustering",
    "WikiCl1SW1024": "Clustering",
    "WikiCl1SW1536": "Clustering",
    "WikiCl2SW1024": "Clustering",
    "WikiCl2SW1536": "Clustering",
    "WikiDi0S1024": "Distance-MMR (0.75)",
    "WikiDi0S1536": "Distance-MMR (0.75)",
    "WikiDi1S21024": "Distance-MMR (0.75)",
    "WikiDi1S21536": "Distance-MMR (0.75)",
    "WikiDi2S21024": "Distance-MMR (0.75)",
    "WikiDi2S21536": "Distance-MMR (0.75)",
    "WikiDi1SW1024": "Distance-MMR (0.75)",
    "WikiDi1SW1536": "Distance-MMR (0.75)",
    "WikiDi2SW1024": "Distance-MMR (0.75)",
    "WikiLe1S21024": "Lead",
    "WikiLe1S21536": "Lead",
    "WikiRa1S21024": "Random",
    "WikiRa1S21536": "Random",
    "WikiRa1SW1024": "Random",
    "WikiRa1SW1536": "Random",
    "WikiCD040SSimDyn1024": "ClustDist 0.4 (SimDyn)",
    "WikiCD040SSimDyn1536": "ClustDist 0.4 (SimDyn)",
    "WikiCD041SSimDyn1024": "ClustDist 0.4 (SimDyn)",
    "WikiCD042SSimDyn1024": "ClustDist 0.4 (SimDyn)",
    "WikiCl0SSimDyn1536": "Clustering (SimDyn)",
    "WikiCD041SSimDyn1536": "ClustDist 0.4 (SimDyn)",
    "WikiCD042SSimDyn1536": "ClustDist 0.4 (SimDyn)",
    "WikiCD043SSimDyn1536": "ClustDist 0.4 (SimDyn)",
    "WikiCD050SSimDyn1536": "ClustDist 0.5 (SimDyn)",
    "WikiCD051SSimDyn1536": "ClustDist 0.5 (SimDyn)",
    "WikiCD052SSimDyn1536": "ClustDist 0.5 (SimDyn)",
    "WikiCD053SSimDyn1536": "ClustDist 0.5 (SimDyn)",
    "WikiCD060SSimDyn1536": "ClustDist 0.6 (SimDyn)",
    "WikiCD061SSimDyn1536": "ClustDist 0.6 (SimDyn)",
    "WikiCD062SSimDyn1536": "ClustDist 0.6 (SimDyn)",
    "WikiCD063SSimDyn1536": "ClustDist 0.6 (SimDyn)",
    "WikiCl1SSimDyn1536": "Clustering (SimDyn)",
    "WikiCl2SSimDyn1536": "Clustering (SimDyn)",
    "WikiChN1024": "Oracle",
    "WikiChN1536": "Oracle",
    "WikiChN1S1024": "Oracle",
    "WikiChN1S1536": "Oracle",
    "WikiChN2S1024": "Oracle",
    "WikiChN2S1536": "Oracle",
    "WikiCl1SSimDynW2048": "Clustering (SimDyn)",
    "WikiCl2SSimDynW2048": "Clustering (SimDyn)",
    "WikiDi090S1024": "Distance-MMR (0.9)",
    "WikiDi090S1536": "Distance-MMR (0.9)",
    "WikiDi091SW1024": "Distance-MMR (0.9)",
    "WikiDi091SW1536": "Distance-MMR (0.9)",
    "WikiDi092SW1024": "Distance-MMR (0.9)",
    "WikiDi092SW1536": "Distance-MMR (0.9)",
}


def preprocessing_abbr(preprocessing_method) -> str:
    abbreviations = {
        "Cheat": "Oracle",
        "Oracle": "Oracle",
        "Lead": "Lead",
        "Random": "Rand.",
        "Clustering": "Clust.",
        "Distance-MMR": "MMR",
        "Distance-MMR (0.75)": "MMR 0.75",
        "Distance-MMR (0.9)": "MMR 0.9",
        "Clustering (SimDyn)": "Clust. SimDyn",
        "ClustDist 0.4 (SimDyn)": "CD 0.4 SimDym",
        "ClustDist 0.5 (SimDyn)": "CD 0.5 SimDyn",
        "ClustDist 0.6 (SimDyn)": "CD 0.6 SimDyn",
    }
    if preprocessing_method in abbreviations:
        return abbreviations[preprocessing_method]
    return preprocessing_method


preprocessing_order = {
    "20Minuten": "-",
    "MultinewsTrunc3584": "original",
    "MultiCD040SSimDyn1024": "original",
    "MultiCD040SSimDyn1536": "original",
    "Wikinews": "original",
    "WikinewsSummarizationChain": "original",
    "WikinewsSummarizationChainAblation": "original",
    "WikinewsTrunc3584": "original",
    "WikinewsClean": "original",
    "WikinewsSimple": "original",
    "WikinewsSimpleS": "random",
    "WikinewsSimpleA": "original",
    "WikinewsSimpleAS": "random",
    "WikinewsSC32": "original",
    "WikinewsSC64": "original",
    "WikinewsSC128": "original",
    "WikinewsSC256": "original",
    "WikinewsSC512": "original",
    "WikinewsSCS2": "original",
    "WikinewsSCS4": "original",
    "WikinewsSCS8": "original",
    "WikinewsSCS16": "original",
    "WikinewsSCS32": "original",
    "WikinewsSplit": "-",
    "WikinewsSplitS2O": "original",
    "WikinewsSplitS2S": "random",
    "WikinewsSplitS2OP41": "original",
    "WikinewsSplitS2SP41": "random",
    "WikinewsClust1C": "cluster size",
    "WikinewsClust1O": "original",
    "WikinewsClust1R": "random",
    "WikinewsClust5C": "cluster size",
    "WikinewsClust5O": "original",
    "WikinewsClust5R": "random",
    "WikinewsClust10C": "cluster size",
    "WikinewsClust10O": "original",
    "WikinewsClust10R": "random",
    "WikinewsSent1L00": "MMR, lambda 0.0",
    "WikinewsSent1L05": "MMR, lambda 0.5",
    "WikinewsSent3L00": "MMR, lambda 0.0",
    "WikinewsSent3L05": "MMR, lambda 0.5",
    "WikinewsSent5L00": "MMR, lambda 0.0",
    "WikinewsSent5L05": "MMR, lambda 0.5",
    "WikinewsSent10L00": "MMR, lambda 0.0",
    "WikinewsSent10L05": "MMR, lambda 0.5",
    "WikiCh1024": "original",
    "WikiCh1536": "original",
    "WikiLe1024": "original",
    "WikiLe1536": "original",
    "WikiRa1024": "original",
    "WikiRa1536": "original",
    "WikiLe1SW1536": "original",
    "WikiDi2SW1536": "original",
    "WikiCl0N1024": "original",
    "WikiCl0N1536": "original",
    "WikiCl0N2048": "original",
    "WikiCl1N21024": "original",
    "WikiCl1N21536": "original",
    "WikiCl1N22048": "original",
    "WikiCl2S21024": "original",
    "WikiCl2S21536": "original",
    "WikiCl2S22048": "original",
    "WikiCl1SW1024": "original",
    "WikiCl1SW1536": "original",
    "WikiCl2SW1024": "original",
    "WikiCl2SW1536": "original",
    "WikiDi0S1024": "original",
    "WikiDi0S1536": "original",
    "WikiDi1S21024": "original",
    "WikiDi1S21536": "original",
    "WikiDi2S21024": "original",
    "WikiDi2S21536": "original",
    "WikiDi1SW1024": "original",
    "WikiDi1SW1536": "original",
    "WikiDi2SW1024": "original",
    "WikiLe1S21024": "original",
    "WikiLe1S21536": "original",
    "WikiRa1S21024": "original",
    "WikiRa1S21536": "original",
    "WikiRa1SW1024": "original",
    "WikiRa1SW1536": "original",
    "WikiCD040SSimDyn1024": "original",
    "WikiCD040SSimDyn1536": "original",
    "WikiCD041SSimDyn1024": "original",
    "WikiCD042SSimDyn1024": "original",
    "WikiCl0SSimDyn1536": "original",
    "WikiCD041SSimDyn1536": "original",
    "WikiCD042SSimDyn1536": "original",
    "WikiCD043SSimDyn1536": "original",
    "WikiCD050SSimDyn1536": "original",
    "WikiCD051SSimDyn1536": "original",
    "WikiCD052SSimDyn1536": "original",
    "WikiCD053SSimDyn1536": "original",
    "WikiCD060SSimDyn1536": "original",
    "WikiCD061SSimDyn1536": "original",
    "WikiCD062SSimDyn1536": "original",
    "WikiCD063SSimDyn1536": "original",
    "WikiCl1SSimDyn1536": "original",
    "WikiCl2SSimDyn1536": "original",
    "WikiChN1024": "original",
    "WikiChN1536": "original",
    "WikiChN1S1024": "original",
    "WikiChN1S1536": "original",
    "WikiChN2S1024": "original",
    "WikiChN2S1536": "original",
    "WikiCl1SSimDynW2048": "original",
    "WikiCl2SSimDynW2048": "original",
    "WikiDi090S1024": "original",
    "WikiDi090S1536": "original",
    "WikiDi091SW1024": "original",
    "WikiDi091SW1536": "original",
    "WikiDi092SW1024": "original",
    "WikiDi092SW1536": "original",
}
dataset_n_fewshot_annotation_map = {
    "MultinewsTrunc3584": "0",
    "WikinewsSummarizationChain": "1",
    "WikinewsSummarizationChainAblation": "0",
    "MultiCD040SSimDyn1024": "0",
    "MultiCD040SSimDyn1536": "0",
    "WikiCh1024": "0",
    "WikiCh1536": "0",
    "WikiLe1024": "0",
    "WikiLe1536": "0",
    "WikiRa1024": "0",
    "WikiRa1536": "0",
    "WikiLe1SW1536": "1",
    "WikiDi2SW1536": "2",
    "WikiCl0N1024": "0",
    "WikiCl0N1536": "0",
    "WikiCl0N2048": "0",
    "WikiCl1N21024": "1",
    "WikiCl1N21536": "1",
    "WikiCl1N22048": "1",
    "WikiCl2S21024": "2",
    "WikiCl2S21536": "2",
    "WikiCl2S22048": "2",
    "WikiCl1SW1024": "1",
    "WikiCl1SW1536": "1",
    "WikiCl2SW1024": "2",
    "WikiCl2SW1536": "2",
    "WikiDi0S1024": "0",
    "WikiDi0S1536": "0",
    "WikiDi1S21024": "1",
    "WikiDi1S21536": "1",
    "WikiDi2S21024": "2",
    "WikiDi2S21536": "2",
    "WikiDi1SW1024": "1",
    "WikiDi1SW1536": "1",
    "WikiDi2SW1024": "2",
    "WikiLe1S21024": "1",
    "WikiLe1S21536": "1",
    "WikiRa1S21024": "1",
    "WikiRa1S21536": "1",
    "WikiRa1SW1024": "1",
    "WikiRa1SW1536": "1",
    "WikiCD040SSimDyn1024": "0",
    "WikiCD040SSimDyn1536": "0",
    "WikiCD041SSimDyn1024": "1",
    "WikiCD042SSimDyn1024": "2",
    "WikiCl0SSimDyn1536": "0",
    "WikiCD041SSimDyn1536": "1",
    "WikiCD042SSimDyn1536": "2",
    "WikiCD043SSimDyn1536": "3",
    "WikiCD050SSimDyn1536": "0",
    "WikiCD051SSimDyn1536": "1",
    "WikiCD052SSimDyn1536": "2",
    "WikiCD053SSimDyn1536": "3",
    "WikiCD060SSimDyn1536": "0",
    "WikiCD061SSimDyn1536": "1",
    "WikiCD062SSimDyn1536": "2",
    "WikiCD063SSimDyn1536": "3",
    "WikiCl1SSimDyn1536": "1",
    "WikiCl2SSimDyn1536": "2",
    "WikiChN1024": "0",
    "WikiChN1536": "0",
    "WikiChN1S1024": "1",
    "WikiChN1S1536": "1",
    "WikiChN2S1024": "2",
    "WikiChN2S1536": "2",
    "WikiCl1SSimDynW2048": "1",
    "WikiCl2SSimDynW2048": "2",
    "WikiDi090S1024": "0",
    "WikiDi090S1536": "0",
    "WikiDi091SW1024": "1",
    "WikiDi091SW1536": "1",
    "WikiDi092SW1024": "2",
    "WikiDi092SW1536": "2",
}
preprocessing_parameters = {
    "20Minuten": "-",
    "MultinewsTrunc3584": "-",
    "MultiCD040SSimDyn1024": "#Input Tokens: 1024",
    "MultiCD040SSimDyn1536": "#Input Tokens: 1536",
    "Wikinews": "-",
    "WikinewsTrunc3584": "-",
    "WikinewsSummarizationChain": "-",
    "WikinewsSummarizationChainAblation": "-",
    "WikinewsClean": "-",
    "WikinewsSimple": "-",
    "WikinewsSimpleS": "-",
    "WikinewsSimpleA": "-",
    "WikinewsSimpleAS": "-",
    "WikinewsSC32": " 32 tokens",
    "WikinewsSC64": " 64 tokens",
    "WikinewsSC128": "128 tokens",
    "WikinewsSC256": "256 tokens",
    "WikinewsSC512": "512 tokens",
    "WikinewsSCS2": " 2 sentences",
    "WikinewsSCS4": " 4 sentences",
    "WikinewsSCS8": " 8 sentences",
    "WikinewsSCS16": "16 sentences",
    "WikinewsSCS32": "32 sentences",
    "WikinewsSplit": "-",
    "WikinewsSplitS2O": "SDS-2, original",
    "WikinewsSplitS2S": "SDS-2, random",
    "WikinewsSplitS2OP41": "SDS-4, original",
    "WikinewsSplitS2SP41": "SDS-4, random",
    "WikinewsClust1C": " 1 sentence, cluster size",
    "WikinewsClust1O": " 1 sentence, original",
    "WikinewsClust1R": " 1 sentence, random",
    "WikinewsClust5C": " 5 sentences, cluster size",
    "WikinewsClust5O": " 5 sentences, original",
    "WikinewsClust5R": " 5 sentences, random",
    "WikinewsClust10C": "10 sentences, cluster size",
    "WikinewsClust10O": "10 sentences, original",
    "WikinewsClust10R": "10 sentences, random",
    "WikinewsSent1L00": " 1 sentence, MMR-lambda 0.0",
    "WikinewsSent1L05": " 1 sentence, MMR-lambda 0.5",
    "WikinewsSent3L00": " 3 sentences, MMR-lambda 0.0",
    "WikinewsSent3L05": " 3 sentences, MMR-lambda 0.5",
    "WikinewsSent5L00": " 5 sentences, MMR-lambda 0.0",
    "WikinewsSent5L05": " 5 sentences, MMR-lambda 0.5",
    "WikinewsSent10L00": "10 sentences, MMR-lambda 0.0",
    "WikinewsSent10L05": "10 sentences, MMR-lambda 0.5",
    "WikiCh1024": "1024 Tokens",
    "WikiCh1536": "1536 Tokens",
    "WikiLe1024": "1024 Tokens",
    "WikiLe1536": "1536 Tokens",
    "WikiRa1024": "1024 Tokens",
    "WikiRa1536": "1536 Tokens",
    "WikiLe1SW1536": "1536 Tokens",
    "WikiDi2SW1536": "1536 Tokens",
    "WikiCl0N1024": "1024 Tokens",
    "WikiCl0N1536": "1536 Tokens",
    "WikiCl0N2048": "2048 Tokens",
    "WikiCl1N21024": "1024 Tokens",
    "WikiCl1N21536": "1536 Tokens",
    "WikiCl1N22048": "2048 Tokens",
    "WikiCl2S21024": "1024 Tokens",
    "WikiCl2S21536": "1536 Tokens",
    "WikiCl2S22048": "2048 Tokens",
    "WikiCl1SW1024": "1024 Tokens",
    "WikiCl1SW1536": "1536 Tokens",
    "WikiCl2SW1024": "1024 Tokens",
    "WikiCl2SW1536": "1536 Tokens",
    "WikiDi0S1024": "1024 Tokens",
    "WikiDi0S1536": "1536 Tokens",
    "WikiDi1S21024": "1024 Tokens",
    "WikiDi1S21536": "1536 Tokens",
    "WikiDi2S21024": "1024 Tokens",
    "WikiDi2S21536": "1536 Tokens",
    "WikiDi1SW1024": "1024 Tokens",
    "WikiDi1SW1536": "1536 Tokens",
    "WikiDi2SW1024": "1024 Tokens",
    "WikiLe1S21024": "1024 Tokens",
    "WikiLe1S21536": "1536 Tokens",
    "WikiRa1S21024": "1024 Tokens",
    "WikiRa1S21536": "1536 Tokens",
    "WikiRa1SW1024": "1024 Tokens",
    "WikiRa1SW1536": "1536 Tokens",
    "WikiCD040SSimDyn1024": "1024 Tokens",
    "WikiCD040SSimDyn1536": "1536 Tokens",
    "WikiCD041SSimDyn1024": "1024 Tokens",
    "WikiCD042SSimDyn1024": "1024 Tokens",
    "WikiCl0SSimDyn1536": "1536 Tokens",
    "WikiCD041SSimDyn1536": "1536 Tokens",
    "WikiCD042SSimDyn1536": "1536 Tokens",
    "WikiCD043SSimDyn1536": "1536 Tokens",
    "WikiCD050SSimDyn1536": "1536 Tokens",
    "WikiCD051SSimDyn1536": "1536 Tokens",
    "WikiCD052SSimDyn1536": "1536 Tokens",
    "WikiCD053SSimDyn1536": "1536 Tokens",
    "WikiCD060SSimDyn1536": "1536 Tokens",
    "WikiCD061SSimDyn1536": "1536 Tokens",
    "WikiCD062SSimDyn1536": "1536 Tokens",
    "WikiCD063SSimDyn1536": "1536 Tokens",
    "WikiCl1SSimDyn1536": "1536 Tokens",
    "WikiCl2SSimDyn1536": "1536 Tokens",
    "WikiChN1024": "1024 Tokens",
    "WikiChN1536": "1536 Tokens",
    "WikiChN1S1024": "1024 Tokens",
    "WikiChN1S1536": "1536 Tokens",
    "WikiChN2S1024": "1024 Tokens",
    "WikiChN2S1536": "1536 Tokens",
    "WikiCl1SSimDynW2048": "2048 Tokens",
    "WikiCl2SSimDynW2048": "2048 Tokens",
    "WikiDi090S1024": "1024 Tokens",
    "WikiDi090S1536": "1536 Tokens",
    "WikiDi091SW1024": "1024 Tokens",
    "WikiDi091SW1536": "1536 Tokens",
    "WikiDi092SW1024": "1024 Tokens",
    "WikiDi092SW1536": "1536 Tokens",
}
prompt_description = {  # short description of the prompt to ID it instead of the prompt ID
    "1": "Basic",
    "2": "Basic",
    "40": "Basic",
    "41": "Basic",
    "42": "Basic",
    "3": "NZZ Prompt",
    "4": "NZZ Prompt",
    "23": "NZZ Prompt",
    "5": "TL;DR",
    "25": "Separator",
    "6": "Basic",
    "7": "Basic",
    "43": "Basic",
    "8": "Basic",
    "9": "Basic",
    "44": "Basic",
    "10": "Simplification",
    "20": "Simplification",
    "11": "Simplification",
    "45": "Simplification",
    "12": "Personification",
    "13": "Personification",
    "46": "Personification",
    "14": "Personification",
    "15": "Personification",
    "47": "Personification",
    "16": "Non-expert-audience",
    "17": "Non-expert-audience",
    "48": "Non-expert-audience",
    "18": "Orig. Tone",
    "19": "Orig. Tone",
    "49": "Orig. Tone",
    "21": "Bullet-point",
    "22": "Bullet-point",
    "30": "Q&A",
    "31": "Q&A",
    "32": "Q&A",
    "33": "Q&A",
    "34": "LtM - Summ all",
    "35": "LtM - Summ all",
    "36": "LtM - Summ all",
    "37": "LtM - Summ all",
    "50": "MDS",
    "51": "MDS",
    "52": "MDS",
    "100": "Basic",
}
prompt_name_map = {  # short description of the prompt to ID it instead of the prompt ID
    "1": "Basic-1-S, EN",
    "2": "Basic-1-S, DE",
    "40": "Basic-2-S, DE",
    "41": "Basic-4-S, DE",
    "42": "Basic-3-S, DE",
    "3": "NZZ, EN",  # NZZ Prompt, EN
    "4": "NZZ, DE",  # NZZ Prompt, DE
    "23": "NZZ-S, DE",  # NZZ Prompt-S, DE
    "5": "TL;DR",
    "25": "Sep.-Only-S",  # Separator-Only-S
    "6": "Basic-5, EN",
    "7": "Basic-5, DE",
    "43": "Basic-5-S, DE",
    "8": "Basic-6, EN",
    "9": "Basic-6, DE",
    "44": "Basic-6-S, DE",
    "10": "Simplif.-1, EN",  # Simplification-1, EN
    "20": "Simplif.-2, EN",  # Simplification-2, EN
    "11": "Simplif.-1, DE",  # Simplification-1, DE
    "45": "Simplif.-1-S, DE",  # Simplification-1-S, DE
    "12": "Person.-1, EN",  # Personification-1, EN
    "13": "Person.-1, DE",  # Personification-1, DE
    "46": "Person.-1-S, DE",  # Personification-1-S, DE
    "14": "Person.-2, EN",  # Personification-2, EN
    "15": "Person.-2, DE",  # Personification-2, DE
    "47": "Person.-2-S, DE",  # Personification-2-S, DE
    "16": "NonExpert, EN",  # Non-expert-audience, EN
    "17": "NonExpert, DE",  # Non-expert-audience, DE
    "48": "NonExpert-S, DE",  # Non-expert-audience-S, DE
    "18": "Orig. Tone, EN",
    "19": "Orig. Tone, DE",
    "49": "Orig. Tone-S, DE",
    "21": "Bullet-S, EN",  # Bullet-point-S, EN
    "22": "Bullet-S, DE",  # Bullet-point-S, DE
    "30": "Q&A-Instr. First, EN",
    "31": "Q&A-Instr. First, DE",
    "32": "Q&A-Quest. First, EN",
    "33": "Q&A-Quest. First, DE",
    "34": "LtM Final-Last, EN",
    "35": "LtM Final-Last, DE",
    "36": "LtM Final-First, EN",
    "37": "LtM Final-First, DE",
    "50": "MDS-1-S, EN",
    "51": "MDS-1-S, DE",
    "52": "MDS-2-S, DE",
    "100": "Custom",
}
prompt_name_MDS_override = {  # map from task-name to prompt-id to prompt-name
    "MDSSumm": {},  # no override rules
    "MDSFCO": {
        "100": "SDS-3",
    },
    "MDSChain": {
        "100": "Chain-Update",  # summarization-chain-intermediate
    },
    "MDS2S": {
        "2": "SDS-2",
        "41": "SDS-4",
    },
    "MDS": {
        "100": "Chain-Update",  # summarization-chain
        "42": "SDS-4",
    },
}

if experiment_name.startswith("mds"):
    prompt_description["42"] = "MDS"
prompt_variant = {  # general variant of the prompt (not the langage)
    "1": "2-3 Sentences",
    "2": "<=  3 Sentences",
    "40": "<=  8 Sentences",
    "41": "<= 10 Sentences",
    "42": "<=  6 Sentences",
    "3": "-",
    "4": "-",
    "23": "-",
    "5": "-",
    "25": "-",
    "6": "-",
    "7": "-",
    "43": "-",
    "8": "Domain",
    "9": "Domain",
    "44": "Domain",
    "10": "-",
    "20": "Tgt Language",
    "11": "-",
    "45": "-",
    "12": "Journalist, Format",
    "13": "Journalist, Format",
    "46": "Journalist, Format",
    "14": "Author",
    "15": "Author",
    "47": "Author",
    "16": "-",
    "17": "-",
    "48": "-",
    "18": "-",
    "19": "-",
    "49": "-",
    "21": "-",
    "22": "-",
    "30": "Instr. / Quest.",
    "31": "Instr. / Quest.",
    "32": "Quest. / Instr.",
    "33": "Quest. / Instr.",
    "34": "Input / Instr.",
    "35": "Input / Instr.",
    "36": "Instr. / Input",
    "37": "Instr. / Input",
    "50": "",
    "51": "",
    "52": "<= 6 Sentences",
    "100": "-",
}
prompt_annotation = {
    # Whether the prompt contains a separation annotation, separating prompt from summary, e.g. "Summary:", "Zusammenfassung:", "TL;DR:"
    "1": True,
    "2": True,
    "40": True,
    "41": True,
    "42": True,
    "3": False,
    "4": False,
    "23": True,
    "5": True,
    "25": True,
    "6": False,
    "7": False,
    "43": True,
    "8": False,
    "9": False,
    "44": True,
    "10": False,
    "20": False,
    "11": False,
    "45": True,
    "12": False,
    "13": False,
    "46": True,
    "14": False,
    "15": False,
    "47": True,
    "16": False,
    "17": False,
    "48": True,
    "18": False,
    "19": False,
    "49": True,
    "21": True,
    "22": True,
    "30": True,
    "31": True,
    "32": True,
    "33": True,
    "34": True,
    "35": True,
    "36": True,
    "37": True,
    "50": True,
    "51": True,
    "52": True,
    "100": True,
}

column_rename_map = {
    "rouge1": "R-1",
    "rouge2": "R-2",
    "rougeL": "R-L",
    "bertscore_precision": "BertScore Precision",
    "bertscore_recall": "BertScore Recall",
    "bertscore_f1": "BertScore F1",
    "coverage": "Coverage (Prompt)",
    "density": "Density (Prompt)",
    "compression": "Compression (Prompt)",
    "promptVersion": "Prompt ID",
    "model": "Model",
    "model-fullname": "Model-Identifier",
    "dataset": "Dataset",
    "dataset-annotation": "Dataset Annotation",
    "n-shot": "N-Shot",
    "preprocessing-method": "Preprocessing Method",
    "preprocessing-parameters": "Preprocessing Parameters",
    "preprocessing-order": "Preprocessing Order",
    "prompt-description": "Prompt Description",
    "prompt-desc-id": "Prompt Desc. [ID]",
    "prompt-variant": "Prompt Variant",
    "prompt-separator": "Has Prompt-Summary Separator",
    "temperature": "Temperature",
    "precision": "Model Precision",
    "prompt_0": "Prompt",
    "logit_0": "Prediction",
    "truth": "GT-Summary",
    "task_name": "Task Name",
    "lang": "Language",
    "lang_score": "Language Score",
    "attribution_score": "Attribution",
    "attribution_truth_value": "Attribution (Truth Value)",
    "attribution_mds_scores": "Attribution (MDS Scores)",
    "attribution_mds_truth_values": "Attribution (Truth Values)",
    "attribution_mds_avg_score": "Attribution (Avg. Score)",
    "attribution_mds_avg_truth_value": "Attribution (Agreement)",
    "conciceness_score": "Conciseness",
    "conciceness_truth_value": "Conciseness (Truth Value)",
    "conciceness_mds_scores": "Conciseness (MDS Scores)",
    "conciceness_mds_truth_values": "Conciseness (Truth Values)",
    "conciceness_mds_avg_score": "Conciseness (Avg. Score)",
    "conciceness_mds_avg_truth_value": "Conciseness (Agreement)",
    "main-ideas_score": "Main Ideas",
    "main-ideas_truth_value": "Main Ideas (Truth Value)",
    "main-ideas_mds_scores": "Main Ideas (MDS Scores)",
    "main-ideas_mds_truth_values": "Main Ideas (Truth Values)",
    "main-ideas_mds_avg_score": "Main Ideas (Avg. Score)",
    "main-ideas_mds_avg_truth_value": "Main Ideas (Agreement)",
}
SEAHORSE_Base_Names = ["Attribution", "Conciseness", "Main Ideas"]
SEAHORSE_Binary_Names = [" (Truth Value)"]
SEAHORSE_Plot_Metric_Names = [
    "", " (Truth Value)", " (Avg. Score)", " (Agreement)",
    " (St. Dev.)", " (Mean)", " (Median)", " (Min)", " (Max)",
    " (90th Percentile)", " (75th Percentile)", " (25th Percentile)", " (10th Percentile)",
    " (First)", " (Last)", " (Middle)",
]
SEAHORSE_MDS_Colnames = [" (Truth Values)", " (MDS Scores)"]
SEAHORSE_Pairplot_Metric_Names = ["", " (Avg. Score)", " (Agreement)", " (Mean)", " (Median)"]
SEAHORSE_overview_plot_additional_extensions = [
    " (Avg. Score)", " (Agreement)", " (St. Dev.)", " (First)", " (Last)", " (Middle)", " (Median)",
    " (90th Percentile)", " (75th Percentile)", " (25th Percentile)", " (10th Percentile)",
]
SEAHORSE_deeper_inspection_extensions = [
    "", " (Avg. Score)", " (Agreement)", " (Median)",
    " (90th Percentile)", " (75th Percentile)", " (25th Percentile)", " (10th Percentile)",
    " (First)", " (Last)", " (Middle)",
]
SEAHORSE_overview_plot_additional_names = [f"{base}{extension}" for base in SEAHORSE_Base_Names for extension in
                                           SEAHORSE_overview_plot_additional_extensions]
OVERVIEW_TABLE_merged_table_relevant_metric_groups = [
    ("metrics", ["R-1", "R-2", "R-L", "BertScore F1", "BertScore Precision", "BertScore Recall", ]),
    ("extracts", ["Coverage", "Density", "Compression", ]),
    ("seahorse", ["Attribution", "Attribution (Avg. Score)", "Main Ideas", "Main Ideas (Avg. Score)"]),
]
OVERVIEW_TABLE_Rename_Map = {
    "BertScore F1": "Bert-F1",
    "BertScore Precision": "Bert-Prec.",
    "BertScore Recall": "Bert-Rec.",
    "Attribution (Avg. Score)": "Attribution",
    "Main Ideas (Avg. Score)": "Main Ideas",
}

StandardEval_Pairplot_Metric_Names = [
    "R-1",  # "R-2", "R-L",
    "BertScore F1",  # "BertScore Precision", "BertScore Recall",
    "Coverage",  # "Density", "Compression",
]

metric_0_1_range = [column_rename_map[x] for x in
                    ["rouge1", "rouge2", "rougeL", "bertscore_precision", "bertscore_recall", "bertscore_f1",
                     "coverage"]]


# Main function
def main(reload_data=True, reload_preprocessed_dataset=False, no_cache_reload_all=False):
    sep_str = '=' * 100
    print(f"{sep_str}\n\tExperiment: {experiment_name}\n{sep_str}\n")

    RESULTS_PATH = os.path.join(RESULTS_BASE_PATH, experiment_name)

    if reload_data:
        # Aggregate results and create DataFrame
        df = load_all_results(RESULTS_PATH, models, shortNames, reload_preprocessed_dataset=reload_preprocessed_dataset,
                              no_cache_reload_all=no_cache_reload_all)
        # Save DataFrame as CSV
        save_dataframe(df, experiment_name)
    else:
        # Load DataFrame from CSV
        df = load_dataframe(experiment_name)

    metric_names, _ = get_metrics_info(df)

    prompts = get_prompts_by_experiment_name(experiment_name)

    # Create plots and overview report
    datasets = df["Dataset"].unique()
    for dataset in datasets:
        df_dataset = df[df["Dataset"] == dataset]

        report_name = f"{experiment_name}-{dataset}"
        report_path = os.path.join("reports", report_name)
        pathlib.Path(report_path).mkdir(parents=True, exist_ok=True)

        create_preprocessed_report(df_dataset, report_name, metric_names, prompts, skip_lang=False)


def make_report_plots(skip_cost_estimate=False, skip_metric_plots=False, skip_seahorse_plots=False,
                      skip_length_statistics=False, skip_inspect_examples=False, prioritize_inspect_examples=False,
                      skip_language_statistics=False, skip_failure_statistics=False,
                      skip_statistics_overview_tables=False, skip_catplots=False, skip_violins=False):
    prompts = get_prompts_by_experiment_name(experiment_name)

    print("Creating report plots ...")

    for dataset in datasets:
        experiment_path = os.path.join("reports", f"{experiment_name}-{dataset}")

        # Load all prepared data
        df = pd.read_csv(os.path.join(experiment_path, "df_filtered.csv"))
        df_all = pd.read_csv(os.path.join(experiment_path, "df_all.csv"))
        df_non_german = pd.read_csv(os.path.join(experiment_path, "df_non_german.csv"))
        df_empty = pd.read_csv(os.path.join(experiment_path, "df_empty.csv"))

        # # SPECIAL CASE -> DELETE AGAIN
        # TODO: EXPLORE THIS
        large_prompt_threshold = 3584  # 3259
        df_large_prompt = df[df["#Prompt Tokens"] >= large_prompt_threshold]
        df_large_prompt['Preprocessing + N-Shot'] = df_large_prompt.apply(
            lambda row: f"{row['Preprocessing + N-Shot']} (LARGE)", axis=1)
        df_small_prompt = df[df["#Prompt Tokens"] < large_prompt_threshold]
        df_small_prompt['Preprocessing + N-Shot'] = df_small_prompt.apply(
            lambda row: f"{row['Preprocessing + N-Shot']}", axis=1)
        df = df_small_prompt  # pd.concat([df_large_prompt, df_small_prompt])

        # from the small prompt examples, find all examples that start the prediction with a lowercase character
        df_lowercase = df[df["Prediction"].str[0].str.islower()]
        # order by prompt tokens descending
        df_lowercase = df_lowercase.sort_values(by="#Prompt Tokens", ascending=False)
        lowercase_html_path = os.path.join(experiment_path, "inspect_lowercase_prediction.html")
        save_df_to_html_simple(df_lowercase, lowercase_html_path, ['doc_id', 'Prompt', 'GT-Summary'])

        df_inspect_cutoff = df_small_prompt[df_small_prompt["#Prompt Tokens"] >= 3500]
        near_cutoff_html_path = os.path.join(experiment_path, "inspect_near_cutoff_for_large_prompt.html")
        save_df_to_html_simple(df_inspect_cutoff, near_cutoff_html_path, ['doc_id', 'Prompt', 'GT-Summary'])

        large_prompts_html_path = os.path.join(experiment_path, "inspect_large_prompts.html")
        save_df_to_html_simple(df_large_prompt, large_prompts_html_path, ['doc_id', 'Prompt', 'GT-Summary'])

        # # only keep Prompt and GT-Summary columns
        # df_large_prompt_save = df_large_prompt.copy()
        # df_large_prompt_save = df_large_prompt_save[['Prompt', 'GT-Summary']]
        # # rename columns
        # df_large_prompt_save.rename(columns={'Prompt': 'article', 'GT-Summary': 'summary'}, inplace=True)
        # # save as json
        # df_large_prompt_save.to_json(os.path.join(experiment_path, "df_large_prompts.json"), orient='records')

        """
        MANUAL INSPECTION OF EXAMPLES
        """
        with open('./llama2-tokenizer.json', 'r') as file:
            a_vocab = json.load(file)
        a_vocab_rev = {k: v for v, k in a_vocab['model']['vocab'].items()}
        enc_to_text = lambda x: "".join([a_vocab_rev[enc_i] for enc_i in x])
        find_row = lambda df, article_snipped: df[df['Prompt'].str.contains(article_snipped, regex=False)]

        # make a prompt-html-file showing all the used prompts
        df_prompts = pd.DataFrame(
            pd.DataFrame({"Prompt ID": df_all['Prompt ID'].unique().tolist() + additional_prompts}).drop_duplicates())
        # df_prompts = pd.DataFrame(df_all['Prompt ID'].drop_duplicates())
        df_prompts['Prompt'] = df_prompts.apply(lambda x: prompts[f"{x['Prompt ID']}"], axis=1)
        df_prompts.sort_values(by='Prompt ID', inplace=True)
        df_prompts.reset_index(inplace=True, drop=True)
        df_prompts["Prompt Length"] = df_prompts["Prompt"].apply(lambda x: len(text_to_llama_bpe(x)))

        # df_prompts = df_prompts.transpose()
        df_prompts = df_prompts.apply(lambda row: df_row_replace(row, ['Prompt'], '\n', '<br>'), axis=1)
        html_data = df_prompts.to_html(escape=False).replace("\\n", "").replace("\n", "")
        html_path = os.path.join(experiment_path, f"prompts_overview.html")
        html_page = html_base.format(table=html_data)
        with open(html_path, "w") as f:
            f.write(html_page)

        # also save as a latex table
        df_prompts["Prompt ID"] = df_prompts["Prompt ID"].apply(
            lambda x: prompt_name_map[f"{x}"] if f"{x}" in prompt_name_map else x)
        df_prompts["Prompt"] = df_prompts["Prompt"].apply(lambda x: x.replace("<br>", " \\newline "))
        df_prompts.to_latex(os.path.join(experiment_path, "prompts_overview.tex"), index=False,
                            column_format='p|p{0.8\\textwidth}')

        """
            EXPERIMENT COST ESTIMATE (IN TOKENS)
        """
        if not skip_cost_estimate:
            # Calculate the number of tokens per experiment
            df_dataset = df_all[df_all["Dataset"] == dataset]
            df_dataset_model = df_dataset[df_dataset["Model"] == shortNames[models[0]]]
            # concatenate the 'prompt_0' columns to 1 string
            prompt_list = df_dataset_model["Prompt"].tolist()
            total_prompt = " ".join(prompt_list)
            # split into tokens (using approximation)
            tokens = approxTokenize(total_prompt)
            numTokens = len(tokens)
            # calculate the average summary size (for all entries that did work out)
            df_dataset = df[df["Dataset"] == dataset]
            df_dataset_model = df_dataset[df_dataset["Model"] == shortNames[models[0]]]
            summary_list = df_dataset_model["GT-Summary"].tolist()
            summary_tokens = [approxTokenize(summary) for summary in summary_list]
            summary_tokens = [len(tokens) for tokens in summary_tokens if tokens]
            if len(summary_tokens) == 0:
                avgSummarySize = 0
            else:
                avgSummarySize = sum(summary_tokens) / len(summary_tokens)
            # Print out the cost
            numRows = df_dataset_model.shape[0]
            numPrompts = len(df['Prompt ID'].unique().tolist())
            print(
                f"Cost for {dataset} ({numPrompts} prompts): input {numTokens}, output {(numRows * avgSummarySize)}\n\tTokens: {numTokens}\n\tAvg. Summary Size: {avgSummarySize}\n\tNum. Rows: {numRows}\n")

        df_all_langs = pd.concat([df, df_non_german])

        metric_names, _ = get_metrics_info(df)

        def inspect_examples_report():
            num_inspect_examples_specific = 25
            inspectGBL = [groupByList] + experiment_config[experiment_name].get("groupByListVariants", [])

            """
            construct specific subsets of the data that may be interesting to look at
            """
            df_inspect_all = df.copy()

            # ... all entries that have low language scores
            df_inspect = df_inspect_all.sort_values(by="Language Score", ascending=True).head(
                num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "low-lang-score", extendedGroupByList=inspectGBL)

            # ... all entries that use up a lot of tokens
            df_inspect_all["Used Tokens"] = df_inspect_all["#Prompt Tokens"] + df_inspect_all["#Predicted Tokens"]
            # use max number of tokens
            df_inspect = df_inspect_all[df_inspect_all["Used Tokens"] >= 4096]
            save_inspect_examples_simple(df_inspect, experiment_path, "max-used-tokens", extendedGroupByList=inspectGBL)
            # use lot's of tokens
            df_inspect = df_inspect_all[df_inspect_all["Used Tokens"] < 4096]
            df_inspect = df_inspect.sort_values(by="Used Tokens", ascending=False).head(num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "high-used-tokens",
                                         extendedGroupByList=inspectGBL)
            # long prompts
            long_prompt_threshold = 3259
            df_inspect = df_inspect_all[df_inspect_all["#Prompt Tokens"] >= long_prompt_threshold]
            save_inspect_examples_simple(df_inspect, experiment_path, "long-prompt", extendedGroupByList=inspectGBL)

            # ... all entries that have a low metric score
            df_inspect = df_inspect_all.sort_values(by="BertScore F1", ascending=True).head(
                num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "low-bertscore-f1",
                                         extendedGroupByList=inspectGBL)

            df_inspect = df_inspect_all.sort_values(by="R-1", ascending=True).head(num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "low-rouge1", extendedGroupByList=inspectGBL)

            df_inspect = df_inspect_all.sort_values(by="Coverage", ascending=True).head(num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "low-coverage", extendedGroupByList=inspectGBL)

            # ... all entries that have a much shorter summary than the gt-summary
            df_inspect_all["Summary Length Ratio"] = df_inspect_all.apply(
                lambda row: row['#Predicted Tokens'] / row['#Input Article Tokens'] if row[
                                                                                           '#Input Article Tokens'] > 0 else -1,
                axis=1)
            df_inspect = df_inspect_all.sort_values(by="Summary Length Ratio", ascending=True).head(
                num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "short-summary", extendedGroupByList=inspectGBL)

            # ... all entries that have a much longer summary than the gt-summary
            df_inspect = df_inspect_all.sort_values(by="Summary Length Ratio", ascending=False).head(
                num_inspect_examples_specific)
            save_inspect_examples_simple(df_inspect, experiment_path, "long-summary", extendedGroupByList=inspectGBL)

            """
            Standard inspect examples (with deeper analysis)
            """
            inspect_examples = find_inspect_examples(df, experiment_path, metric_names, groupbyList=groupByList,
                                                     extendedGroupByList=inspectGBL, suffix="")
            with open(os.path.join(experiment_path, f"inspect_examples_{groupByList[0]}_{groupByList[1]}.json"),
                      "w") as f:
                json.dump(inspect_examples, f, indent=4)

        if not skip_inspect_examples:
            if prioritize_inspect_examples:
                inspect_examples_report()

        gblExtension = experiment_config[experiment_name].get("groupByListVariants", [])

        if not skip_failure_statistics:
            # Make plots showing the failure rate
            bucket_size = 1000
            bucket_colname = f"#Input Article Tokens Bucket"
            df_all[bucket_colname] = df_all["#Input Article Tokens"].apply(
                lambda x: f"{int(math.floor(x / bucket_size)) * bucket_size}".rjust(
                    5) + "-" + f"{int(math.ceil(x / bucket_size)) * bucket_size}".rjust(5))
            x_group = "Temperature"
            if x_group in groupByList:
                x_group = "Model"

            failure_statistics_plot(df_all, experiment_path, groupbyList=groupByList, groupByListExtension=gblExtension,
                                    x_group=x_group,
                                    groupByIterator=["Prompt Desc. [ID]", "Model", "Temperature", bucket_colname,
                                                     "N-Input Docs"])

        if not skip_metric_plots:
            # make violin (distribution) plot showing distribution of metric values per model and prompt
            # ... group by model (comparing prompts)
            # ... group by prompt (comparing models)
            _ = make_metric_distribution_figures(df, experiment_path, metric_names, groupbyList=groupByList,
                                                 groupByListExtension=gblExtension, file_suffix="",
                                                 skip_catplots=skip_catplots, skip_violins=skip_violins)
            gc.collect()

        if not skip_statistics_overview_tables:
            # calculate a statistics overview table (per model and prompt) -> calculate df, re-arrange for different views
            # ... showing median, 10th percentile, 90th percentile, and the stderr for each metric
            # ... showing 1 table for (model, prompt)
            # ... showing 1 table for (model) -> comparing prompts
            # ... showing 1 table for (prompt) -> comparing models
            pathlib.Path(os.path.join(experiment_path, "overview_table")).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.join(experiment_path, "detail_table")).mkdir(parents=True, exist_ok=True)
            if "Prompt ID" in groupByList:
                # copy the groupByList, and replace "Prompt ID" with "Prompt Desc. [ID]" in-place
                gbl = groupByList.copy()
                gbl[gbl.index("Prompt ID")] = "Prompt Desc. [ID]"
                gblList = [groupByList, gbl]
            else:
                gblList = [groupByList]
            for gbl in gblList:
                statistics_overview_metric_names = metric_names + SEAHORSE_Base_Names + SEAHORSE_overview_plot_additional_names
                tables_overview, tables_detail, agg_names = statistics_overview(experiment_path, df,
                                                                                statistics_overview_metric_names,
                                                                                groupbyList=gbl,
                                                                                gblExtension=gblExtension)
            gc.collect()

        if not skip_seahorse_plots:
            _ = make_seahorse_distribution_figures(df, experiment_path, metric_names, groupbyList=groupByList,
                                                   groupByListExtension=gblExtension, file_suffix="",
                                                   skip_catplots=skip_catplots)
            gc.collect()
        # _ = make_metric_distribution_figures(df_all, experiment_path, metric_names, groupbyList=groupByList, file_suffix="_all")
        # _ = make_metric_distribution_figures(df_non_german, experiment_path, metric_names, groupbyList=groupByList, file_suffix="_non_german")

        if not skip_language_statistics:
            # re-do language statistics plots
            # _ = language_statistics(df_all_langs, experiment_path, prompts)
            df_lang_stat, df_prompt_lang_effect = language_statistics(df_all_langs, experiment_path, prompts,
                                                                      groupbyList=groupByList)
            df_lang_stat.to_csv(os.path.join(experiment_path, "df_lang_stat.csv"), index=False)
            df_prompt_lang_effect.to_csv(os.path.join(experiment_path, "df_prompt_lang_effect.csv"), index=False)

        # per metric -> sample 2 documents with the worst performance and 2 documents with the best performance
        # ... and 2 documents with the median performance
        # inspect_examples = find_inspect_examples(df, experiment_path, metric_names, groupbyList=groupByList, suffix="")
        # inspect_examples_all = find_inspect_examples(df_all, experiment_path, metric_names, groupbyList=groupByList, suffix="_all")
        # inspect_examples_mp = find_inspect_examples(df, experiment_path, metric_names, groupbyList=["Model", "Prompt ID"])

        if not skip_length_statistics:
            # create the statistics for the token lengths and number of sentences
            gblExtension = [groupByList] + experiment_config[experiment_name].get("groupByListVariants", [])
            df_prompt_length_impact = length_statistics(df, experiment_path, groupbyList=groupByList,
                                                        gblExtension=gblExtension, approximation=True)
            # _ = length_statistics(df_all, experiment_path, groupbyList=groupByList, gblExtension=gblExtension,
            #                       approximation=True, file_suffix="_all")
            if df_prompt_length_impact is not None:
                df_prompt_length_impact.to_csv(os.path.join(experiment_path, "df_prompt_length_impact_mean.csv"),
                                               index=False)

        if not skip_inspect_examples:
            # save inspect examples in JSON
            if not prioritize_inspect_examples:
                inspect_examples_report()

        # with open(os.path.join(experiment_path, f"inspect_examples_{groupByList[0]}_{groupByList[1]}_all.json"), "w") as f:
        #     json.dump(inspect_examples_all, f, indent=4)
        # with open(os.path.join(experiment_path, f"inspect_examples_model_promptVersion.json"), "w") as f:
        #     json.dump(inspect_examples_mp, f, indent=4)


if __name__ == "__main__":

    # if --name is passed, read the experiment name from the command line and overwrite the default
    if "--name" in sys.argv:
        experiment_name_arg = sys.argv[sys.argv.index("--name") + 1]
        if experiment_name_arg in experiment_config:
            experiment_name = experiment_name_arg
            print(f"Overwriting experiment name with {experiment_name}")

            groupByList = experiment_config[experiment_name]["groupByList"]
            models = experiment_config[experiment_name]["models"]
            datasets = experiment_config[experiment_name]["datasets"]
            additional_prompts = experiment_config[experiment_name].get("additional_prompts", [])
            if experiment_name.startswith("mds"):
                prompt_description["42"] = "MDS"
        else:
            print(f"Experiment name {experiment_name_arg} not found in experiment_config")
            exit(-1)

    # if passing argument --full, run the main function
    if "--full" in sys.argv:
        reload_data = "--reload" in sys.argv
        reload_preprocessed_dataset = "--base_dataset_reload" in sys.argv
        no_cache_reload_all = "--no_cache_reload_all" in sys.argv
        main(reload_data=reload_data, reload_preprocessed_dataset=reload_preprocessed_dataset,
             no_cache_reload_all=no_cache_reload_all)

    skip_cost_estimate = "--skip_cost_estimate" in sys.argv
    skip_metric_plots = "--skip_metric_plots" in sys.argv
    skip_seahorse_plots = "--skip_seahorse_plots" in sys.argv
    skip_length_statistics = "--skip_length_statistics" in sys.argv
    skip_inspect_examples = "--skip_inspect_examples" in sys.argv
    skip_language_statistics = "--skip_language_statistics" in sys.argv
    skip_failure_statistics = "--skip_failure_statistics" in sys.argv
    skip_statistics_overview_tables = "--skip_statistics_overview_tables" in sys.argv
    prioritize_inspect_examples = "--inspect-examples" in sys.argv
    skip_catplots = "--skip_catplots" in sys.argv
    skip_violins = "--skip_violins" in sys.argv

    make_report_plots(
        skip_cost_estimate=skip_cost_estimate,
        skip_metric_plots=skip_metric_plots,
        skip_seahorse_plots=skip_seahorse_plots,
        skip_length_statistics=skip_length_statistics,
        skip_inspect_examples=skip_inspect_examples,
        prioritize_inspect_examples=prioritize_inspect_examples,
        skip_language_statistics=skip_language_statistics,
        skip_failure_statistics=skip_failure_statistics,
        skip_statistics_overview_tables=skip_statistics_overview_tables,
        skip_catplots=skip_catplots,
        skip_violins=skip_violins,
    )
