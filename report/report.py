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
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import jinja2
from typing import List, Dict, Union, Callable, Tuple
import fasttext
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

pd.set_option("display.precision", 4)
sns.set_theme(style="darkgrid", rc={'figure.figsize': (17, 7)})
sns.despine(bottom=True, left=True, offset=5)
# plt.tight_layout()

ROUND_TO_DECIMALS = 4


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
def load_results(model_name):
    model_name = rename_hf_model(model_name)
    results_path = f"../{RESULTS_PATH}/{model_name}"
    result_files = [f for f in os.listdir(results_path) if f.endswith(".json")]

    # Load the results from all JSON files and extend their fields as needed
    all_results = []
    for file in result_files:
        with open(os.path.join(results_path, file), "r") as f:
            result = json.load(f)

            task_name, nshot, dataset, promptVersion, temperature, precision = extract_dataset_and_task_name(file)
            for entry in result:
                entry["task_name"] = task_name
                entry["n-shot"] = nshot
                entry["dataset"] = dataset
                entry["promptVersion"] = promptVersion
                entry["model"] = model_name
                entry['temperature'] = temperature
                entry['precision'] = precision

            all_results.extend(result)

    out = pd.DataFrame(all_results)
    return out


# function receiving a list of model names calling load_results for each model and concatenating the results
def load_all_results(model_names, shortNames):
    dfs = []
    for model_name in model_names:
        df = load_results(model_name)
        dfs.append(df)

    # prepare shotNames map -> replace / with -
    shortNames = {rename_hf_model(model_name): shortNames[model_name] for model_name in shortNames}

    df = pd.concat(dfs)
    df.rename(columns={"model": "model-fullname"}, inplace=True)
    df["model"] = df["model-fullname"].map(shortNames)
    df["dataset-annotation"] = df["dataset"].apply(lambda x: datasetAnnotationMap[x] if x in datasetAnnotationMap else "")
    df["dataset"] = df["dataset"].apply(lambda x: x if x not in datasetNameMap else datasetNameMap[x])
    return df


# Function to save DataFrame as CSV under the experiment name
def save_dataframe(df, experiment_name):
    report_folder = "data"
    pathlib.Path(report_folder).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(report_folder, f"{experiment_name}.csv")
    df.to_csv(csv_path, index=False)


# Function to make plots and create an overview report
def create_preprocessed_report(df, experiment_name, metric_names, prompts, skip_lang=True):
    # groupByList = ["model", "promptVersion"]

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
        df = lang_detect(df, "logit_0")
    # before filtering out any rows -> save
    df.to_csv(os.path.join(experiment_path, "df_all.csv"), index=False)

    # create the statistics for the empty predictions
    # ... calculate the success rate per prompt
    # ... calculate the top 5 documents with the worst success rate
    df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions = empty_statistics(df, groupbyList=groupByList)
    # Filter out the empty predictions
    df_empty = df[df["logit_0"] == ""]
    df = df[df["logit_0"] != ""]

    # Calculate the language of the predicted text using spacy language detection
    # ... create a new column in the dataframe containing the predicted language
    # ... make a plot showing the distribution of the predicted languages per model and prompt
    if not skip_lang:
        df_lang_stat, df_prompt_lang_effect = language_statistics(df, experiment_path, prompts, groupbyList=groupByList)
        # Filter out the non-german predictions
        df_non_german = df[df["lang"] != "de"]
        df = df[df["lang"] == "de"]

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
        df_lang_stat.to_csv(os.path.join(experiment_path, "df_lang_stat.csv"), index=False)
        df_prompt_lang_effect.to_csv(os.path.join(experiment_path, "df_prompt_lang_effect.csv"), index=False)
        df_non_german.to_csv(os.path.join(experiment_path, "df_non_german.csv"), index=False)


def empty_statistics(df, groupbyList=["model", "promptVersion"], text_col="logit_0", topN=5, docCol="doc_id", promptCol="promptVersion") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.shape[0] == 0:
        return None, None, None, None, None
    print("Calculating empty-prediction statistics...")
    # Get the count of empty predictions
    df_empty_stat = df[df[text_col] == ""].groupby(groupbyList).agg(
        {"logit_0": "count"}).reset_index()
    df_empty_stat = df_empty_stat.rename(columns={"logit_0": "empty_count"})

    # Calculate the success rate per document
    df_num_empty_docs = df[df[text_col] == ""].groupby(docCol).agg(
        {"logit_0": "count"}).reset_index()
    df_num_empty_docs = df_num_empty_docs.rename(columns={"logit_0": "empty_count"})

    # Calculate the success rate per prompt
    df_num_empty_prompts = df[df[text_col] == ""].groupby(promptCol).agg(
        {"logit_0": "count"}).reset_index()
    df_num_empty_prompts = df_num_empty_prompts.rename(columns={"logit_0": "empty_count"})

    # Get the worst documents and prompts (with the highest number of empty predictions)
    worst_empty_docs_IDs = df_num_empty_docs.sort_values(by="empty_count", ascending=False).head(topN)[docCol].tolist()
    worst_empty_promptVersions_IDs = df_num_empty_prompts.sort_values(by="empty_count", ascending=False).head(topN)[promptCol].tolist()
    worst_empty_docs = df[df[docCol].isin(worst_empty_docs_IDs)]
    worst_empty_promptVersions = df[df[promptCol].isin(worst_empty_promptVersions_IDs)]

    # df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions
    return df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions


def failure_statistics_plot(df_all, experiment_path, groupbyList=["model", "promptVersion"], x_group="temperature", text_col="logit_0", langCol="lang", docCol="doc_id", promptCol="promptVersion"):
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

    # Calculate the failures
    df_failures = df_all.copy()
    df_failures["failure"] = df_failures.apply(lambda x: get_failure(x), axis=1)
    df_failures['failure'] = df_failures['failure'].astype('category')

    # Aggregate the success rates per model and prompt (by category)
    df_failure_stat = df_failures.groupby(groupbyList + ["failure", x_group]).agg({"logit_0": "count"}).reset_index()
    df_failure_stat = df_failure_stat.rename(columns={"logit_0": "count"})
    df_failure_stat = df_failure_stat.reset_index(drop=True)

    # Make a facet-grid plot, make 1 plot per x_group value
    for x_group_val in df_failures[x_group].unique():
        df_failure_stat_x_group = df_failures[df_failures[x_group] == x_group_val]
        failure_plot = sns.FacetGrid(df_failure_stat_x_group, col=groupbyList[0], row=groupbyList[1], height=3, aspect=1.5)
        failure_plot.map(sns.countplot, 'failure')
        failure_plot_path = os.path.join(experiment_path, subfolder_name, f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}_{x_group}_{x_group_val}.pdf")
        plt.savefig(failure_plot_path)
        plt.close()

    failure_plot = sns.FacetGrid(df_failures, col=groupbyList[0], row=groupbyList[1], height=3, aspect=1.5)
    failure_plot.map(sns.countplot, 'failure')
    failure_plot_path = os.path.join(experiment_path, subfolder_name, f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}.pdf")
    plt.savefig(failure_plot_path)
    plt.close()

    # also save the failure statistics as csv
    df_failure_stat.to_csv(os.path.join(experiment_path, subfolder_name, f"failure_statistics_overview__{groupbyList[0]}_{groupbyList[1]}_{x_group}.csv"), index=False)


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
        text_list = [text.replace("\n", " ") for text in df[col_name].tolist()]
        predictions = fast_lang_detector.predict(text_list)
        post_preds = [(t[0][0].replace('__label__', ''), t[1][0]) for t in zip(predictions[0], predictions[1])]

        # Postprocess
        threshold = 0.5
        empty_val = "other"
        # compute the predicted language and assign it to the dataframe
        df["lang"] = [lang if score >= threshold else empty_val for lang, score in post_preds]
        df["lang_score"] = [score for lang, score in post_preds]
        return df

    else:
        # Use Spacy Language Detection
        lang_detector = get_de_lang_detector()
        threshold = 0.5
        empty_val = "other"

        tqdm.pandas()

        # generate dataframe column by df.apply
        df["lang"] = df[col_name].progress_apply(lambda x: get_lang_from_detector(lang_detector, x, threshold, empty_val))
        return df


def language_statistics(df, experiment_path, prompts, groupbyList=["model", "promptVersion"], promptVersionCol="promptVersion", modelCol="model"):
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
    df_lang_stat = df.groupby(groupbyList + ["lang"]).agg({"logit_0": "count"}).reset_index()

    # calculate the effect of the prompt being in the target language (german) on the target language being the same (german)
    df_prompt_langs = df.copy()
    df_prompt_langs['prompt_lang'] = df_prompt_langs.apply(lambda row: prompt_langs[f"{row[promptVersionCol]}"], axis=1)
    groupbyListLangEffect = list(set([col for col in groupbyList if col != promptVersionCol] + [modelCol]))  # exclude promptVersion from groupbyList
    df_prompt_lang_effect = df_prompt_langs.groupby(groupbyListLangEffect + ["prompt_lang", "lang"]).agg({"logit_0": "count"}).reset_index()

    # Make a language effect plot, showing the effect of the prompt language on the predicted language (for each model)
    # Grouping all other languages together
    df_prompt_lang_effect_plot = df_prompt_lang_effect.copy()
    df_prompt_lang_effect_plot["lang"] = df_prompt_lang_effect_plot["lang"].apply(lambda x: "other" if x != "de" else x)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.groupby([modelCol, "prompt_lang", "lang"]).agg({"logit_0": "sum"}).reset_index()
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.rename(columns={"logit_0": "count"})
    df_prompt_lang_effect_plot["count"] = df_prompt_lang_effect_plot["count"].astype(int)
    # Make the actual plot
    lang_effect_plot = sns.FacetGrid(df_prompt_lang_effect_plot, col="prompt_lang", row=modelCol, height=3, aspect=1.5)
    lang_effect_plot.map(sns.barplot, "lang", "count")
    token_distr_plot_path = os.path.join(experiment_path, "prompt_lang_effect_plot.pdf")
    plt.savefig(token_distr_plot_path)
    plt.close()
    #
    # sns.set_theme(style="whitegrid")
    # g = sns.catplot(
    #     data=df_prompt_lang_effect_plot, kind="bar",
    #     x="prompt_lang", y="count", hue="lang",
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
    df_prompt_lang_effect_plot["lang"] = df_prompt_lang_effect_plot["lang"].apply(lambda x: "other" if x not in plot_langs else x)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.groupby(["model", "prompt_lang", "lang"]).agg({"logit_0": "sum"}).reset_index()
    # fill missing combinations with 0 value (issue with plotting)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.groupby(["model", "prompt_lang", "lang"]).agg({"logit_0": "sum"}).reset_index()
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.pivot(index=["model", "prompt_lang"], columns="lang", values="logit_0").reset_index()
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.fillna(0)
    df_prompt_lang_effect_plot = df_prompt_lang_effect_plot.melt(id_vars=["model", "prompt_lang"], var_name="lang", value_name="logit_0")
    df_prompt_lang_effect_plot["logit_0"] = df_prompt_lang_effect_plot["logit_0"].astype(int)
    # plot
    lang_effect_distr_plot = sns.FacetGrid(df_prompt_lang_effect_plot, col="prompt_lang", row=modelCol, height=3, aspect=1.5, sharex=True)
    lang_effect_distr_plot.map(sns.barplot, "lang", "logit_0")
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


def num_sent_and_tokens(text: str, approximation: bool = True) -> Tuple[int, int]:
    sentences, tokens = sentence_splitting(text, approximation=approximation)
    return len(sentences), len(tokens)


def length_statistics(df, save_base_path, groupbyList=["model", "promptVersion"], approximation=True, file_suffix=""):
    if df.shape[0] == 0:
        return None, None, None
    print("Calculating length statistics...")

    pathlib.Path(os.path.join(save_base_path, "length-statistics")).mkdir(parents=True, exist_ok=True)

    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"

    # calculate the number of tokens and sentences per prediction
    df_len = df.copy()
    df_len["num_sentences"], df_len["num_tokens"] = zip(*df_len["logit_0"].apply(lambda x: num_sent_and_tokens(x, approximation=approximation)))

    # calculate the impact of the prompt on the number of tokens and sentences
    df_prompt_length_impact = df_len.groupby(groupbyList).agg({"num_sentences": "mean", "num_tokens": "mean"}).reset_index()

    for lbl, (a, b) in zip(["", "_flipped"], [(0, 1), (1, 0)]):
        # make plots showing the two distributions (with a subplot grid, one for each dimension in the group-by list) (just showing the number of tokens)
        # make a subplot grid with one plot for each dimension in the group-by list
        token_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        token_distr_plot.map(sns.histplot, "num_tokens", bins=20)
        token_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"token_distr_plot_hist{lbl}{file_suffix}.pdf")
        plt.savefig(token_distr_plot_path)
        plt.close()

        # make the same plot as violin plot
        token_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        token_distr_plot.map(sns.violinplot, "num_tokens", bins=20)
        token_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"token_distr_plot_violin{lbl}{file_suffix}.pdf")
        plt.savefig(token_distr_plot_path)
        plt.close()

        # make a subplot grid with one plot for each dimension in the group-by list
        sent_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        sent_distr_plot.map(sns.histplot, "num_sentences", bins=20)
        sent_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"sent_distr_plot_hist{lbl}{file_suffix}.pdf")
        plt.savefig(sent_distr_plot_path)
        plt.close()

        # make the same plot as violin plot
        sent_distr_plot = sns.FacetGrid(df_len, col=groupbyList[a], row=groupbyList[b], height=3, aspect=1.5)
        sent_distr_plot.map(sns.violinplot, "num_sentences", bins=20)
        sent_distr_plot_path = os.path.join(save_base_path, "length-statistics", f"sent_distr_plot_violin{lbl}{file_suffix}.pdf")
        plt.savefig(sent_distr_plot_path)
        plt.close()

    return df_prompt_length_impact, token_distr_plot_path, sent_distr_plot_path


def percentile_agg(percentile):
    return lambda x: np.percentile(x, percentile)


def bootstrap_CI(statistic, confidence_level=0.95, num_samples=10000):
    def CI(x):
        if len(x) <= 2:
            return math.nan, math.nan
        data = (np.array(x),)  # convert to sequence
        confidence_interval = bootstrap(data, statistic, n_resamples=num_samples, confidence_level=confidence_level, method='basic', vectorized=True).confidence_interval
        LOW_CI = round(confidence_interval.low, ROUND_TO_DECIMALS)
        HIGH_CI = round(confidence_interval.high, ROUND_TO_DECIMALS)
        return LOW_CI, HIGH_CI

    return CI


def statistics_overview(df, metric_names, groupbyList=["model", "promptVersion"]):
    if df.shape[0] == 0:
        return None, None, None
    confidence_level = 0.95

    print("Calculating statistics overview tables...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"
    agg_funcs = ["median", "std", "min", "max", percentile_agg(5), percentile_agg(95), bootstrap_CI(np.median, confidence_level=confidence_level)]
    agg_names = ["median", "std", "min", "max", "5th percentile", "95th percentile", f"Median {int(confidence_level * 100)}% CI"]
    agg_dict = {agg_name: agg_func for agg_name, agg_func in zip(agg_names, agg_funcs)}

    out_overview = []
    out_detail = []

    # make a table showing the median of each metric (one table per metric), grouped by groupbyList
    for metric_name in metric_names:
        # set the type of the metric column to float
        df[metric_name] = df[metric_name].astype(float, copy=True)
        # calculate the table
        df_metric = df.groupby(groupbyList).agg({metric_name: ["median", bootstrap_CI(np.median, confidence_level=confidence_level)]}).reset_index()
        col_new = groupbyList + ['median', f"Median {int(confidence_level * 100)}% CI"]
        df_metric.columns = col_new
        df_metric = df_metric.round(ROUND_TO_DECIMALS)
        out_overview.append({
            "name": f"median {metric_name}",
            "df": df_metric
        })

        # TODO: Make a plot showing this overview table -> showing median performance + confidence interval for each model and promptVersion

    # loop over models (looking at them separately), and over the metrics (looking at them separately)
    # each table showing the median, 10th percentile, 90th percentile, and the stderr for each metric
    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        for metric_name in metric_names:
            df_metric = df_model.groupby(groupbyList).agg({metric_name: agg_funcs}).reset_index()

            col_new = groupbyList + agg_names
            df_metric.columns = col_new
            df_metric = df_metric.round(ROUND_TO_DECIMALS)
            out_detail.append({
                "name": f"{model_name} {metric_name}",
                "df": df_metric
            })
    # and doing the same but looking first at the prompts individually, and then at the metrics within a given prompt
    for promptVersion in df["promptVersion"].unique():
        df_prompt = df[df["promptVersion"] == promptVersion]
        for metric_name in metric_names:
            df_metric = df_prompt.groupby(groupbyList).agg({metric_name: agg_funcs}).reset_index()

            col_new = groupbyList + agg_names
            df_metric.columns = col_new
            df_metric = df_metric.round(ROUND_TO_DECIMALS)
            out_detail.append({
                "name": f"{promptVersion} {metric_name}",
                "df": df_metric
            })

    return out_overview, out_detail, agg_names


def statistical_tests(df, metric_names, comparison_columns=['model', 'promptVersion'], prompt_category_group="model", promptVersionCol="promptVersion"):
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
        df_stat[promptCat] = df_stat["promptVersion"].apply(lambda x: 1 if x in promptCategoriesInv[promptCat] else 0)

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
                promptVersions_out = [promptVersion for promptVersion in df_group[promptVersionCol] if promptVersion not in promptVersions_in]

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


def find_inspect_examples(df, experiment_path, metric_names, groupbyList=["model", "promptVersion"], numExamples=2, suffix=""):
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

    # delete old folder if it exists
    base_folder = f"inspect-examples_{groupbyList[0]}_{groupbyList[1]}{suffix}"
    if os.path.exists(os.path.join(experiment_path, base_folder)):
        shutil.rmtree(os.path.join(experiment_path, base_folder))
    # make subfolders to store specific examples
    pathlib.Path(os.path.join(experiment_path, base_folder)).mkdir(parents=True, exist_ok=True)

    # Initialize
    def numExamples(category):
        if category == "all":
            return 10
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
                sample_population = df_groupA[(df_groupA[metric_name] >= percentile_values[0]) & (df_groupA[metric_name] <= percentile_values[1])]
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
                    sample_population = df_groupB[(df_groupB[metric_name] >= percentile_values[0]) & (df_groupB[metric_name] <= percentile_values[1])]
                    df_sample = sample_population.sample(min(sample_population.shape[0], numExamples(cat)))
                    # add the examples to the dict
                    out[f"{groupA}"][f"{groupB}"][metric_name][cat] = df_sample.to_dict(orient="records")

    # Gather the examples for the inspect metrics, and make side-by-side comparisons to the other combinations in the data
    # ... also sample 10 random examples and make a side-by-side comparison to the other combinations in the data
    inspect_metrics = ["bertscore_f1"]
    num_all_inspect_examples = 20
    inspect_exclude_columns = ['prompt_0', 'logit_0', "truth"]
    all_other_cols = [col for col in df.columns if col not in inspect_exclude_columns]
    html_base = """<!DOCTYPE html>
<html>
<body>

{table}

</body>
</html>"""
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
            pathlib.Path(os.path.join(experiment_path, base_folder, inspect_metric, cat)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(experiment_path, base_folder, inspect_metric, "random")).mkdir(parents=True, exist_ok=True)

        # for each document ID, get all the predictions for that document and save them in a html file
        for cat in interesting_docIds:
            for docID in interesting_docIds[cat]:
                doc_id_gt_summaries = df[df["doc_id"] == docID]["truth"].unique().tolist()
                # loop over all the gt-summaries, make 1 html file per gt-summary
                for summ_idx, gt_summary in enumerate(doc_id_gt_summaries):
                    df_docID = df[df['truth'].apply(lambda x: x.strip()) == gt_summary.strip()]

                    # make a new dataframe with the columns in inspect_exclude_columns, and the rest should be combined into a single column (JSON)
                    # df_docID["info"] = df_docID.apply(lambda row: json.dumps({col: row[col] for col in all_other_cols}), axis=1)
                    # df_docID = df_docID.drop(columns=all_other_cols)
                    df_docID = df_docID.transpose()

                    # save to html
                    html_data = df_docID.to_html().replace("\\n", "<br>").replace("\n", "<br>")
                    html_path = os.path.join(experiment_path, base_folder, inspect_metric, cat, f"{docID}__{summ_idx}.html")
                    html_page = html_base.format(table=html_data)
                    with open(html_path, "w") as f:
                        f.write(html_page)

        # also sample random document IDs and save them in a html file
        uniqueDocIDs = df["doc_id"].unique()
        random_docIDs = list(np.random.choice(uniqueDocIDs, size=min(len(uniqueDocIDs), num_all_inspect_examples), replace=False)) + [1, 85]
        for docID in random_docIDs:
            doc_id_gt_summaries = df[df["doc_id"] == docID]["truth"].unique().tolist()
            # loop over all the gt-summaries, make 1 html file per gt-summary
            for summ_idx, gt_summary in enumerate(doc_id_gt_summaries):
                df_docID = df[df['truth'].apply(lambda x: x.strip()) == gt_summary.strip()]
                # make a new dataframe with the columns in inspect_exclude_columns, and the rest should be combined into a single column (JSON)
                # df_docID["info"] = df_docID.apply(lambda row: json.dumps({col: row[col] for col in all_other_cols}), axis=1)
                # df_docID = df_docID.drop(columns=all_other_cols)
                df_docID = df_docID.transpose()

                # save to html
                html_data = df_docID.to_html().replace("\\n", "<br>").replace("\n", "<br>")
                html_path = os.path.join(experiment_path, base_folder, inspect_metric, "random", f"{docID}__{summ_idx}.html")
                html_page = html_base.format(table=html_data)
                with open(html_path, "w") as f:
                    f.write(html_page)

    return out


def make_metric_distribution_figures(df, save_base_path, metric_names, groupbyList=["model", "promptVersion"], file_suffix="") -> Tuple[List[List[str]], List[List[str]]]:
    if df.shape[0] == 0:
        return [], []
    print("Making metric distribution figures...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"

    prompt_plot_paths = []
    model_plot_paths = []

    def label_sort_key_func(x):
        # if there is a number in the label, sort by that number first, then the string order
        if isinstance(x, int):
            return (x, "")
        match = re.search(r'\d+', x)
        if match:
            return (int(match.group()), x)
        else:
            return (0, x)

    def get_sorted_labels(data_df, column):
        sorted_axis_labels = df[column].unique().tolist()
        sorted_axis_labels.sort(key=lambda x: label_sort_key_func(x))
        return sorted_axis_labels

    # make all subfolders
    for metric_name in metric_names:
        pathlib.Path(os.path.join(save_base_path, f"metric-{metric_name}")).mkdir(parents=True, exist_ok=True)
    for model_name in df["model"].unique():
        pathlib.Path(os.path.join(save_base_path, model_name)).mkdir(parents=True, exist_ok=True)
    for promptVersion in df["promptVersion"].unique():
        pathlib.Path(os.path.join(save_base_path, f"Prompt-{promptVersion}")).mkdir(parents=True, exist_ok=True)

    # sorted prompt versions (ordered as numbers, not strings)
    # promptVersions = df["promptVersion"].unique().tolist()
    # promptVersions.sort(key=lambda x: int(x))
    # ...
    residualGroupBy = [col for col in groupbyList if col not in ["promptVersion", "model"]]

    # Loop over the models -> making 1 figure per metric, comparing the prompts (on the same model)
    for model_name in df["model"].unique():
        out_paths = []
        df_model = df[df["model"] == model_name]

        for metric_name in metric_names:
            # make a violin plot showing the distribution of the metric values for each prompt
            if len(residualGroupBy) > 0:
                sorted_x_axis_labels = get_sorted_labels(df, "promptVersion")
                sorted_hue_labels = get_sorted_labels(df, residualGroupBy[0])
                violin_plot = sns.violinplot(data=df_model, x="promptVersion", hue=residualGroupBy[0], y=metric_name, order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                if metric_name in metric_0_1_range:
                    violin_plot.set_ylim(0, 1)
                # save
                violin_plot_path = os.path.join(save_base_path, model_name, f"{model_name}_{metric_name}_violin_plot{file_suffix}.pdf")
                plt.savefig(violin_plot_path)
                out_paths.append(violin_plot_path)
                plt.close()

                violin_plot = sns.violinplot(data=df_model, x=residualGroupBy[0], hue='promptVersion', y=metric_name, order=sorted_hue_labels, hue_order=sorted_x_axis_labels)
                if metric_name in metric_0_1_range:
                    violin_plot.set_ylim(0, 1)
                # save
                violin_plot_path = os.path.join(save_base_path, model_name, f"{model_name}_{metric_name}_R_violin_plot{file_suffix}.pdf")
                plt.savefig(violin_plot_path)
                out_paths.append(violin_plot_path)
                plt.close()
            else:
                sorted_x_axis_labels = get_sorted_labels(df, "promptVersion")
                violin_plot = sns.violinplot(data=df_model, x="promptVersion", y=metric_name, order=sorted_x_axis_labels)
                if metric_name in metric_0_1_range:
                    violin_plot.set_ylim(0, 1)
                # save
                violin_plot_path = os.path.join(save_base_path, model_name, f"{model_name}_{metric_name}_violin_plot{file_suffix}.pdf")
                plt.savefig(violin_plot_path)
                out_paths.append(violin_plot_path)
                plt.close()
        model_plot_paths.append(out_paths)

    # Loop over the prompts -> making 1 figure per metric, comparing the models (on the same prompt)
    for promptVersion in df["promptVersion"].unique():
        out_paths = []
        df_prompt = df[df["promptVersion"] == promptVersion]
        for metric_name in metric_names:
            # make a violin plot showing the distribution of the metric values for each model
            if len(residualGroupBy) > 0:
                sorted_x_axis_labels = get_sorted_labels(df, "model")
                sorted_hue_labels = get_sorted_labels(df, residualGroupBy[0])
                violin_plot = sns.violinplot(data=df_prompt, x="model", hue=residualGroupBy[0], y=metric_name, order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
                if metric_name in metric_0_1_range:
                    violin_plot.set_ylim(0, 1)
                # save
                violin_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}", f"Prompt_{promptVersion}_{metric_name}_violin_plot{file_suffix}.pdf")
                plt.savefig(violin_plot_path)
                out_paths.append(violin_plot_path)
                plt.close()

                violin_plot = sns.violinplot(data=df_prompt, x=residualGroupBy[0], hue="model", y=metric_name, order=sorted_hue_labels, hue_order=sorted_x_axis_labels)
                if metric_name in metric_0_1_range:
                    violin_plot.set_ylim(0, 1)
                # save
                violin_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}", f"Prompt_{promptVersion}_{metric_name}_R_violin_plot{file_suffix}.pdf")
                plt.savefig(violin_plot_path)
                out_paths.append(violin_plot_path)
                plt.close()
            else:
                sorted_x_axis_labels = get_sorted_labels(df, "model")
                violin_plot = sns.violinplot(data=df_prompt, x="model", y=metric_name, order=sorted_x_axis_labels)
                if metric_name in metric_0_1_range:
                    violin_plot.set_ylim(0, 1)
                # save
                violin_plot_path = os.path.join(save_base_path, f"Prompt-{promptVersion}", f"Prompt_{promptVersion}_{metric_name}_violin_plot{file_suffix}.pdf")
                plt.savefig(violin_plot_path)
                out_paths.append(violin_plot_path)
                plt.close()
        prompt_plot_paths.append(out_paths)

    # Loop over the metrics -> make 1 figure per metric, comparing groupByList[0] with groupByList[1] (with hue) -> make plot in both directions
    for metric_name in metric_names:
        # 0-1 and 1-0 plot
        for a, b in [(0, 1), (1, 0)]:
            sorted_x_axis_labels = get_sorted_labels(df, groupbyList[a])
            sorted_hue_labels = get_sorted_labels(df, groupbyList[b])

            violin_plot = sns.violinplot(data=df, x=groupbyList[a], hue=groupbyList[b], y=metric_name, points=500, cut=0, order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
            if metric_name in metric_0_1_range:
                violin_plot.set_ylim(0, 1)
            # save
            violin_plot_path = os.path.join(save_base_path, f"metric-{metric_name}", f"{metric_name}_{groupbyList[a]}_{groupbyList[b]}_violin_plot{file_suffix}.pdf")
            plt.savefig(violin_plot_path)
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

            box_plot = sns.boxplot(data=df, x=groupbyList[a], hue=groupbyList[b], y=metric_name, flierprops={"marker": "x"}, notch=True, bootstrap=1000, order=sorted_x_axis_labels, hue_order=sorted_hue_labels)
            if metric_name in metric_0_1_range:
                box_plot.set_ylim(0, 1)
            # save
            box_plot_path = os.path.join(save_base_path, f"metric-{metric_name}", f"{metric_name}_{groupbyList[a]}_{groupbyList[b]}_box_plot{file_suffix}.pdf")
            plt.savefig(box_plot_path)
            out_paths.append(box_plot_path)
            plt.close()

    return prompt_plot_paths, model_plot_paths


def get_metrics_info(df) -> Tuple[List[str], Dict[str, bool]]:
    """
    Returns a list of metric names and a dicti onary mapping each metric name to a list of percentiles to be calculated.
    :param df: DataFrame containing the results
    :return: metric_names, metric_ordering
        metric_names: List of metric names
        metric_ordering: Dictionary mapping each metric name a boolean indicating whether a larger metric value is better or not
    """
    exclude = [
        'doc_id', 'prompt_0', 'logit_0', 'truth', 'dataset', 'promptVersion', 'model', 'model-fullname', 'lang', 'lang_score',
        'temperature', 'precision', 'task_name', 'dataset-annotation', 'n-shot'
    ]
    metric_names = [col for col in list(df.columns) if col not in exclude]
    metric_ordering_all = {
        "rouge1": True,
        "rouge2": True,
        "rougeL": True,
        "bertscore_precision": True,
        "bertscore_recall": True,
        "bertscore_f1": True,
        "coverage": False,
        "density": False,
        "compression": False,
    }

    return metric_names, {metric_name: metric_ordering_all[metric_name] for metric_name in metric_names}


"""
    SELECT THE EXPERIMENT TO BUILD THE REPORT ON HERE
"""
# TODO
experiment_name = "mds-2stage-experiment"

"""
    ADD NEW EXPERIMENTS HERE
"""
# read the json file into a dictionary
experiment_config = {
    "few-shot-initial": {
        "groupByList": ["n-shot", "promptVersion"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"]
    },
    "least-to-most-prompting-stage1": {
        "groupByList": ["promptVersion", "model"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"]
    },
    "least-to-most-prompting-stage2": {
        "groupByList": ["promptVersion", "dataset-annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["20Minuten"]
    },
    "mds-baseline": {
        "groupByList": ["promptVersion", "dataset-annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-shuffling-and-annotation-experiment": {
        "groupByList": ["promptVersion", "dataset-annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-chunking-experiment": {
        "groupByList": ["promptVersion", "dataset-annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"]
    },
    "mds-2stage-experiment": {
        "groupByList": ["promptVersion", "dataset-annotation"],
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "datasets": ["Wikinews"],
        "additional_prompts": ["21", "22", "30", "31", "32", "33"]
    },
    "base-experiment": {
        "groupByList": ["promptVersion", "model"],
        "models": ["meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct", "bigscience/bloomz-7b1-mt"],
        "datasets": ["20Minuten"]
    },
    "base-experiment-temperature": {
        "groupByList": ["promptVersion", "temperature"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Llama-2-70b-chat-hf",
            "tiiuae/falcon-7b-instruct",
            # "tiiuae/falcon-40b-instruct",
            "bigscience/bloomz-7b1-mt"
        ],
        "datasets": ["20Minuten"]
    },
    "experiment-sizes": {
        "groupByList": ["promptVersion", "model"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            "garage-bAInd/Platypus2-70B-instruct",
        ],
        "datasets": ["20Minuten"]
    },
    "versions-experiment": {
        "groupByList": ["promptVersion", "model"],
        "models": [
            "gpt-4",
            "palm2",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "fangloveskari/ORCA_LLaMA_70B_QLoRA",
            "garage-bAInd/Platypus2-70B-instruct",
        ],
        "datasets": ["20Minuten"]
    },
    "versions-experiment-gpt4-only": {
        "groupByList": ["promptVersion", "model"],
        "models": ["gpt-4"],
        "datasets": ["20Minuten"]
    },
    "empty-experiment": {
        "groupByList": ["task_name", "promptVersion"],
        "models": [
            "meta-llama/Llama-2-7b-chat-hf",
        ],
        "datasets": ["20minSmol"]
    },
    "tmp": {}
}

RESULTS_BASE_PATH = 'results_bag'

groupByList = experiment_config[experiment_name]["groupByList"]
models = experiment_config[experiment_name]["models"]
datasets = experiment_config[experiment_name]["datasets"]
additional_prompts = experiment_config[experiment_name].get("additional_prompts", [])
RESULTS_PATH = os.path.join(RESULTS_BASE_PATH, experiment_name)
shortNames = {
    "gpt-4": "GPT 4",
    "palm2": "PaLM 2",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2  7b",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-2 13b",
    "meta-llama/Llama-2-70b-chat-hf": "Llama-2 70b",
    "tiiuae/falcon-7b-instruct": "Falcon  7b",
    "tiiuae/falcon-40b-instruct": "Falcon 40b",
    "bigscience/bloomz-7b1-mt": "BloomZ  7b",
    "fangloveskari/ORCA_LLaMA_70B_QLoRA": "OrcaLlama2 70B",
    "garage-bAInd/Platypus2-70B-instruct": "Platypus2 70B",
}
datasetNameMap = {
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
}
datasetAnnotationMap = {
    "20minTS250": "20Minuten, 250 samples",
    "20min0": "20Minuten, shard 1",
    "20min1": "20Minuten, shard 2",
    "20min2": "20Minuten, shard 3",
    "20min3": "20Minuten, shard 4",
    "20minLtm2p22S": "20Min, Prompt 22,\nPrompt at start",
    "20minLtm2p22E": "20Min, Prompt 22,\nPrompt at end",
    "20minLtm2p31S": "20Min, Prompt 31,\nPrompt at start",
    "20minLtm2p31E": "20Min, Prompt 31,\nPrompt at end",
    "20minLtm2p33S": "20Min, Prompt 33,\nPrompt at start",
    "20minLtm2p33E": "20Min, Prompt 33,\nPrompt at end",
    "Wikinews": "basic,\nfull articles,\noriginal order",
    "WikinewsClean": "cleaning,\nfull artices,\noriginal order",
    "WikinewsSimple": "no annotation,\noriginal order",
    "WikinewsSimpleS": "no annotation,\nrandom order",
    "WikinewsSimpleA": "article idx ann.,\noriginal order",
    "WikinewsSimpleAS": "article ids ann.,\nrandom order",
    "WikinewsSC32": "token prefix,\n 32 tokens",
    "WikinewsSC64": "token prefix,\n 64 tokens",
    "WikinewsSC128": "token prefix,\n128 tokens",
    "WikinewsSC256": "token prefix,\n256 tokens",
    "WikinewsSC512": "token prefix,\n512 tokens",
    "WikinewsSCS2": "sentence prefix,\n 2 sentences",
    "WikinewsSCS4": "sentence prefix,\n 4 sentences",
    "WikinewsSCS8": "sentence prefix,\n 8 sentences",
    "WikinewsSCS16": "sentence prefix,\n16 sentences",
    "WikinewsSCS32": "sentence prefix,\n32 sentences",
    "WikinewsSplit": "2-stage summary,\nstage 1",
    "WikinewsSplitS2O": "2-stage summary,\noriginal order",
    "WikinewsSplitS2S": "2-stage summary,\nrandom order",
}
metric_0_1_range = ["rouge1", "rouge2", "rougeL", "bertscore_precision", "bertscore_recall", "bertscore_f1", "coverage"]


# Main function
def main():
    # Aggregate results and create DataFrame
    df = load_all_results(models, shortNames)

    # TODO: Adapt here
    # df = df[df["promptVersion"].isin(["1", "2", "3", "5", "20"])]

    # Save DataFrame as CSV
    save_dataframe(df, experiment_name)

    metric_names, _ = get_metrics_info(df)

    # Get the prompts from the prompts_bag.json file for the given experiment
    prompts_bag_path = f"prompts_bag.json"
    with open(prompts_bag_path, "r") as f:
        prompts_bag = json.load(f)
        prompts = prompts_bag[experiment_name]

    # Create plots and overview report
    datasets = df["dataset"].unique()
    for dataset in datasets:
        df_dataset = df[df["dataset"] == dataset]

        report_name = f"{experiment_name}-{dataset}"
        report_path = os.path.join("reports", report_name)
        pathlib.Path(report_path).mkdir(parents=True, exist_ok=True)

        create_preprocessed_report(df_dataset, report_name, metric_names, prompts, skip_lang=False)


def make_report_plots():
    # Get the prompts from the prompts_bag.json file for the given experiment
    prompts_bag_path = f"prompts_bag.json"
    with open(prompts_bag_path, "r") as f:
        prompts_bag = json.load(f)
        prompts = prompts_bag[experiment_name]

    for dataset in datasets:
        experiment_path = os.path.join("reports", f"{experiment_name}-{dataset}")

        # Load all prepared data
        df = pd.read_csv(os.path.join(experiment_path, "df_filtered.csv"))
        df_all = pd.read_csv(os.path.join(experiment_path, "df_all.csv"))
        df_non_german = pd.read_csv(os.path.join(experiment_path, "df_non_german.csv"))
        df_empty = pd.read_csv(os.path.join(experiment_path, "df_empty.csv"))

        # make a prompt-html-file showing all the used prompts
        html_base = """<!DOCTYPE html>
        <html>
        <body>

        {table}

        </body>
        </html>"""
        df_prompts = pd.DataFrame(pd.DataFrame({"promptVersion": df_all['promptVersion'].unique().tolist() + additional_prompts}).drop_duplicates())
        df_prompts = pd.DataFrame(df_all['promptVersion'].drop_duplicates())
        df_prompts['prompt'] = df_prompts['promptVersion'].apply(lambda x: prompts[f"{x}"])
        df_prompts.sort_values(by='promptVersion', inplace=True)
        df_prompts.reset_index(inplace=True, drop=True)
        # df_prompts = df_prompts.transpose()
        html_data = df_prompts.to_html().replace("\\n", "<br>").replace("\n", "<br>")
        html_path = os.path.join(experiment_path, f"prompts_overview.html")
        html_page = html_base.format(table=html_data)
        with open(html_path, "w") as f:
            f.write(html_page)

        """
            EXPERIMENT COST ESTIMATE (IN TOKENS)
        """
        # Calculate the number of tokens per experiment
        df_dataset = df_all[df_all["dataset"] == dataset]
        df_dataset_model = df_dataset[df_dataset["model"] == shortNames[models[0]]]
        # concatenate the 'prompt_0' columns to 1 string
        prompt_list = df_dataset_model["prompt_0"].tolist()
        total_prompt = " ".join(prompt_list)
        # split into tokens (using approximation)
        tokens = approxTokenize(total_prompt)
        numTokens = len(tokens)
        # calculate the average summary size (for all entries that did work out)
        df_dataset = df[df["dataset"] == dataset]
        df_dataset_model = df_dataset[df_dataset["model"] == shortNames[models[0]]]
        summary_list = df_dataset_model["truth"].tolist()
        summary_tokens = [approxTokenize(summary) for summary in summary_list]
        summary_tokens = [len(tokens) for tokens in summary_tokens if tokens]
        if len(summary_tokens) == 0:
            avgSummarySize = 0
        else:
            avgSummarySize = sum(summary_tokens) / len(summary_tokens)
        # Print out the cost
        numRows = df_dataset_model.shape[0]
        numPrompts = len(df['promptVersion'].unique().tolist())
        print(f"Cost for {dataset} ({numPrompts} prompts): input {numTokens}, output {(numRows * avgSummarySize)}\n\tTokens: {numTokens}\n\tAvg. Summary Size: {avgSummarySize}\n\tNum. Rows: {numRows}\n\n")

        df_all_langs = pd.concat([df, df_non_german])

        metric_names, _ = get_metrics_info(df)
        inspect_examples = find_inspect_examples(df, experiment_path, metric_names, groupbyList=groupByList, suffix="")
        inspect_examples_all = find_inspect_examples(df_all, experiment_path, metric_names, groupbyList=groupByList, suffix="_all")

        # Make plots showing the failure rate
        failure_statistics_plot(df_all, experiment_path, groupbyList=groupByList, x_group="temperature")
        failure_statistics_plot(df_all, experiment_path, groupbyList=['promptVersion', 'model'], x_group="temperature")
        failure_statistics_plot(df_all, experiment_path, groupbyList=['promptVersion', 'temperature'], x_group="model")
        failure_statistics_plot(df_all, experiment_path, groupbyList=['temperature', 'model'], x_group="promptVersion")

        # make violin (distribution) plot showing distribution of metric values per model and prompt
        # ... group by model (comparing prompts)
        # ... group by prompt (comparing models)
        _ = make_metric_distribution_figures(df, experiment_path, metric_names, groupbyList=groupByList, file_suffix="")
        _ = make_metric_distribution_figures(df_all, experiment_path, metric_names, groupbyList=groupByList, file_suffix="_all")
        _ = make_metric_distribution_figures(df_non_german, experiment_path, metric_names, groupbyList=groupByList, file_suffix="_non_german")

        # re-do language statistics plots
        _ = language_statistics(df_all_langs, experiment_path, prompts)

        # create the statistics for the token lengths and number of sentences
        df_prompt_length_impact, token_distr_plot_path, sent_distr_plot_path = length_statistics(df, experiment_path, groupbyList=groupByList, approximation=True)
        _ = length_statistics(df_all, experiment_path, groupbyList=groupByList, approximation=True, file_suffix="_all")

        # per metric -> sample 2 documents with the worst performance and 2 documents with the best performance
        # ... and 2 documents with the median performance
        inspect_examples_mp = find_inspect_examples(df, experiment_path, metric_names, groupbyList=["model", "promptVersion"])

        if df_prompt_length_impact is not None:
            df_prompt_length_impact.to_csv(os.path.join(experiment_path, "df_prompt_length_impact.csv"), index=False)

        # calculate a statistics overview table (per model and prompt) -> calculate df, re-arrange for different views
        # ... showing median, 10th percentile, 90th percentile, and the stderr for each metric
        # ... showing 1 table for (model, prompt)
        # ... showing 1 table for (model) -> comparing prompts
        # ... showing 1 table for (prompt) -> comparing models
        tables_overview, tables_detail, agg_names = statistics_overview(df, metric_names, groupbyList=groupByList)

        pathlib.Path(os.path.join(experiment_path, "overview_table")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(experiment_path, "detail_table")).mkdir(parents=True, exist_ok=True)
        if tables_overview is not None:
            for table in tables_overview:
                table["df"].to_csv(os.path.join(experiment_path, "overview_table", f"overview_table_{table['name']}.csv"), index=False)
        if tables_detail is not None:
            for table in tables_detail:
                table["df"].to_csv(os.path.join(experiment_path, "detail_table", f"detail_table_{table['name']}.csv"), index=False)

        # save inspect examples in JSON
        with open(os.path.join(experiment_path, f"inspect_examples_{groupByList[0]}_{groupByList[1]}.json"), "w") as f:
            json.dump(inspect_examples, f, indent=4)
        with open(os.path.join(experiment_path, f"inspect_examples_{groupByList[0]}_{groupByList[1]}_all.json"), "w") as f:
            json.dump(inspect_examples_all, f, indent=4)
        with open(os.path.join(experiment_path, f"inspect_examples_model_promptVersion.json"), "w") as f:
            json.dump(inspect_examples_mp, f, indent=4)


if __name__ == "__main__":
    # if passing argument --full, run the main function
    if "--full" in sys.argv:
        main()
    make_report_plots()
