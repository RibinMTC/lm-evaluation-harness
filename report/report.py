import re
from tqdm import tqdm
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import jinja2
from typing import List, Dict, Union, Callable, Tuple
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


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
    prompt_version = split_filename[2]
    dataset_name = split_filename[1]
    return dataset_name, prompt_version


# Function to load results from JSON files to a dataframe
def load_results(model_name):
    model_name = rename_hf_model(model_name)
    results_path = f"../results/{model_name}"
    result_files = [f for f in os.listdir(results_path) if f.endswith(".json")]

    # Load the results from all JSON files and extend their fields as needed
    all_results = []
    for file in result_files:
        with open(os.path.join(results_path, file), "r") as f:
            result = json.load(f)

            dataset, promptVersion = extract_dataset_and_task_name(file)
            for entry in result:
                entry["dataset"] = dataset
                entry["promptVersion"] = promptVersion
                entry["model"] = model_name

            all_results.extend(result)

    out = pd.DataFrame(all_results)
    return out


# function receiving a list of model names calling load_results for each model and concatenating the results
def load_all_results(model_names, shortNames):
    dfs = []
    for model_name in model_names:
        df = load_results(model_name)
        dfs.append(df)

    df = pd.concat(dfs)
    df.rename(columns={"model": "model-fullname"}, inplace=True)
    df["model"] = df["model-fullname"].map(shortNames)
    return df


# Function to save DataFrame as CSV under the experiment name
def save_dataframe(df, experiment_name):
    report_folder = "data"
    pathlib.Path(report_folder).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(report_folder, f"{experiment_name}.csv")
    df.to_csv(csv_path, index=False)


# Function to make plots and create an overview report
def create_report(df, experiment_name, metric_names, prompts):
    experiment_path = os.path.join("reports", experiment_name)

    templateLoader = jinja2.FileSystemLoader(searchpath="./templates")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "report_template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render()
    # {% include ['my_template.html', 'another_template.html'] %}

    # prepare the subfolder for the given report to save the plots and the overview report to
    report_folder = "reports"
    pathlib.Path(os.path.join(report_folder, experiment_name)).mkdir(parents=True, exist_ok=True)

    # create the statistics for the empty predictions
    # ... calculate the success rate per prompt
    # ... calculate the top 5 documents with the worst success rate
    df_empty_stat, df_num_empty_docs, df_num_empty_prompts, worst_empty_docs, worst_empty_promptVersions = empty_statistics(df, groupbyList=["model", "promptVersion"])
    # Filter out the empty predictions
    df = df[df["logit_0"] != ""]

    # Calculate the language of the predicted text using spacy language detection
    # ... create a new column in the dataframe containing the predicted language
    # ... make a plot showing the distribution of the predicted languages per model and prompt
    df = lang_detect(df, "logit_0")
    df_lang_stat, df_prompt_lang_effect, df_non_german = language_statistics(df, prompts, groupbyList=["model", "promptVersion"])

    # create the statistics for the token lengths and number of sentences
    df_prompt_length_impact, token_distr_plot_path, sent_distr_plot_path = length_statistics(df, experiment_path, groupbyList=["model", "promptVersion"], approximation=True)

    # calculate a statistics overview table (per model and prompt) -> calculate df, re-arrange for different views
    # ... showing median, 10th percentile, 90th percentile, and the stderr for each metric
    # ... showing 1 table for (model, prompt)
    # ... showing 1 table for (model) -> comparing prompts
    # ... showing 1 table for (prompt) -> comparing models
    tables_overview, tables_detail, agg_names = statistics_overview(df, metric_names, groupbyList=["model", "promptVersion"])

    # per metric -> sample 2 documents with the worst performance and 2 documents with the best performance
    # ... and 2 documents with the median performance
    inspect_examples = find_inspect_examples(df, metric_names, groupbyList=["model", "promptVersion"])

    # make violin (distribution) plot showing distribution of metric values per model and prompt
    # ... group by model (comparing prompts)
    # ... group by prompt (comparing models)
    violin_figure_paths = make_metric_distribution_figures(df, experiment_path, metric_names, groupbyList=["model", "promptVersion"])

    # Save all stuff
    print("Saving results...")
    df.to_csv(os.path.join(experiment_path, "df_all.csv"), index=False)
    df_empty_stat.to_csv(os.path.join(experiment_path, "df_empty_stat.csv"), index=False)
    df_lang_stat.to_csv(os.path.join(experiment_path, "df_lang_stat.csv"), index=False)
    df_non_german.to_csv(os.path.join(experiment_path, "df_non_german.csv"), index=False)
    df_num_empty_docs.to_csv(os.path.join(experiment_path, "df_num_empty_docs.csv"), index=False)
    df_num_empty_prompts.to_csv(os.path.join(experiment_path, "df_num_empty_prompts.csv"), index=False)
    df_prompt_lang_effect.to_csv(os.path.join(experiment_path, "df_prompt_lang_effect.csv"), index=False)
    df_prompt_length_impact.to_csv(os.path.join(experiment_path, "df_prompt_length_impact.csv"), index=False)
    for model in inspect_examples:
        inspect_examples[model].to_csv(os.path.join(experiment_path, f"inspect_examples_{model}.csv"), index=False)


    # TODO: Use prompts


def empty_statistics(df, groupbyList=["model", "promptVersion"], text_col="logit_0", topN=5, docCol="doc_id", promptCol="promptVersion") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def lang_detect(df, col_name):
    print("Detecting Languages...")
    lang_detector = get_de_lang_detector()
    threshold = 0.5
    empty_val = "other"

    # generate dataframe column by df.apply
    df["lang"] = df[col_name].apply(lambda x: get_lang_from_detector(lang_detector, x, threshold, empty_val))
    return df


def language_statistics(df, prompts, groupbyList=["model", "promptVersion"], promptVersionCol="promptVersion"):
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
    df_prompt_langs['prompt_lang'] = df_prompt_langs['promptVersion'].map(prompt_langs)
    groupbyListLangEffect = [col for col in groupbyList if col != promptVersionCol]  # exclude promptVersion from groupbyList
    df_prompt_lang_effect = df_prompt_langs.groupby(groupbyListLangEffect + ["prompt_lang", "lang"]).agg({"logit_0": "count"}).reset_index()

    # extract all non-german predictions
    df_non_german = df[df["lang"] != "de"]

    return df_lang_stat, df_prompt_lang_effect, df_non_german


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


def length_statistics(df, save_base_path, groupbyList=["model", "promptVersion"], approximation=True):
    print("Calculating length statistics...")

    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"

    # calculate the number of tokens and sentences per prediction
    df_len = df.copy()
    df_len["num_sentences"], df_len["num_tokens"] = zip(*df_len["logit_0"].apply(lambda x: num_sent_and_tokens(x, approximation=approximation)))

    # calculate the impact of the prompt on the number of tokens and sentences
    df_prompt_length_impact = df_len.groupby(groupbyList).agg({"num_sentences": "mean", "num_tokens": "mean"}).reset_index()

    # make plots showing the two distributions (with a subplot grid, one for each dimension in the group-by list) (just showing the number of tokens)
    # make a subplot grid with one plot for each dimension in the group-by list
    token_distr_plot = sns.FacetGrid(df_len, col=groupbyList[0], row=groupbyList[1], height=3, aspect=1.5)
    # plot the distribution of the number of tokens in each group
    token_distr_plot.map(sns.histplot, "num_tokens", bins=20)
    # save
    token_distr_plot_path = os.path.join(save_base_path, "token_distr_plot.png")
    plt.savefig(token_distr_plot_path)
    plt.close()

    # make a subplot grid with one plot for each dimension in the group-by list
    sent_distr_plot = sns.FacetGrid(df_len, col=groupbyList[0], row=groupbyList[1], height=3, aspect=1.5)
    # plot the distribution of the number of sentences in each group
    sent_distr_plot.map(sns.histplot, "num_sentences", bins=20)
    # save
    sent_distr_plot_path = os.path.join(save_base_path, "sent_distr_plot.png")
    plt.savefig(sent_distr_plot_path)
    plt.close()

    return df_prompt_length_impact, token_distr_plot_path, sent_distr_plot_path


def percentile_agg(percentile):
    return lambda x: np.percentile(x, percentile)


def statistics_overview(df, metric_names, groupbyList=["model", "promptVersion"]):
    print("Calculating statistics overview tables...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"
    agg_funcs = ["median", "std", "min", "max", percentile_agg(5), percentile_agg(95)]
    agg_names = ["median", "std", "min", "max", "5th percentile", "95th percentile"]
    agg_dict = {agg_name: agg_func for agg_name, agg_func in zip(agg_names, agg_funcs)}

    out_overview = []
    out_detail = []

    # make a table showing the median of each metric (one table per metric), grouped by groupbyList
    for metric_name in metric_names:
        # set the type of the metric column to float
        df[metric_name] = df[metric_name].astype(float, copy=True)
        # calculate the table
        df_metric = df.groupby(groupbyList).agg({metric_name: "median"}).reset_index()
        out_overview.append({
            "name": f"median {metric_name}",
            "df": df_metric
        })

    # loop over models (looking at them separately), and over the metrics (looking at them separately)
    # each table showing the median, 10th percentile, 90th percentile, and the stderr for each metric
    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]
        for metric_name in metric_names:
            df_metric = df_model.groupby(groupbyList).agg({metric_name: agg_funcs}).reset_index()

            col_new = groupbyList + agg_names
            df_metric.columns = col_new
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
            out_detail.append({
                "name": f"{promptVersion} {metric_name}",
                "df": df_metric
            })

    return out_overview, out_detail, agg_names


def find_inspect_examples(df, metric_names, groupbyList=["model", "promptVersion"], numExamples=2):
    print("Finding examples to inspect...")
    # percentile ranges to sample from
    sample_ranges = {
        "worst": [0, 5],
        "median": [47.5, 52.5],
        "best": [95, 100]
    }

    # Initialize
    numExamples = 2
    out = {}
    for model in df["model"].unique():
        out[f"{model}"] = {}
        out[f"{model}"]["general"] = {
            f"{metric_name}": {
                "best": [],
                "worst": [],
                "median": []
            } for metric_name in metric_names
        }
        for promptVersion in df["promptVersion"].unique():
            out[f"{model}"][f"{promptVersion}"] = {
                f"{metric_name}": {
                    "best": [],
                    "worst": [],
                    "median": []
                } for metric_name in metric_names
            }

    # fill the dict with examples
    for model in df["model"].unique():
        # get the df for the current model
        df_model = df[df["model"] == model]
        # in general, compute the percentiles for each metric and sample the examples
        for metric_name in metric_names:
            for cat in out[model]["general"][metric_name]:
                # calculate the percentiles
                percentile_range = sample_ranges[cat]
                percentile_values = np.percentile(df_model[metric_name], percentile_range)
                # sample the examples
                sample_population = df_model[(df_model[metric_name] >= percentile_values[0]) & (df_model[metric_name] <= percentile_values[1])]
                df_sample = sample_population.sample(min(sample_population.shape[0], numExamples))
                # add the examples to the dict
                out[model]["general"][metric_name][cat] = df_sample
        # for each prompt, compute the percentiles for each metric and sample the examples
        for promptVersion in df["promptVersion"].unique():
            # get the df for the current prompt
            df_prompt = df_model[df_model["promptVersion"] == promptVersion]
            for metric_name in metric_names:
                for cat in out[model][promptVersion][metric_name]:
                    # calculate the percentiles
                    percentile_range = sample_ranges[cat]
                    percentile_values = np.percentile(df_prompt[metric_name], percentile_range)
                    # sample the examples
                    sample_population = df_prompt[(df_prompt[metric_name] >= percentile_values[0]) & (df_prompt[metric_name] <= percentile_values[1])]
                    df_sample = sample_population.sample(min(sample_population.shape[0], numExamples))
                    # add the examples to the dict
                    out[model][promptVersion][metric_name][cat] = df_sample

    return out


def make_metric_distribution_figures(df, save_base_path, metric_names, groupbyList=["model", "promptVersion"]) -> Tuple[List[List[str]], List[List[str]]]:
    print("Making metric distribution figures...")
    assert len(groupbyList) == 2, "groupbyList must contain exactly 2 elements"

    prompt_plot_paths = []
    model_plot_paths = []

    # Loop over the models -> making 1 figure per metric, comparing the prompts (on the same model)
    for model_name in df["model"].unique():
        out_paths = []
        df_model = df[df["model"] == model_name]
        for metric_name in metric_names:
            # make a violin plot showing the distribution of the metric values for each prompt
            violin_plot = sns.violinplot(data=df_model, x="promptVersion", y=metric_name)
            # save
            violin_plot_path = os.path.join(save_base_path, f"{model_name}_{metric_name}_violin_plot.png")
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
            violin_plot = sns.violinplot(data=df_prompt, x="model", y=metric_name)
            # save
            violin_plot_path = os.path.join(save_base_path, f"Prompt_{promptVersion}_{metric_name}_violin_plot.png")
            plt.savefig(violin_plot_path)
            out_paths.append(violin_plot_path)
            plt.close()
        prompt_plot_paths.append(out_paths)

    return prompt_plot_paths, model_plot_paths


def get_metrics_info(df) -> Tuple[List[str], Dict[str, bool]]:
    """
    Returns a list of metric names and a dictionary mapping each metric name to a list of percentiles to be calculated.
    :param df: DataFrame containing the results
    :return: metric_names, metric_ordering
        metric_names: List of metric names
        metric_ordering: Dictionary mapping each metric name a boolean indicating whether a larger metric value is better or not
    """
    exclude = ['doc_id', 'prompt_0', 'logit_0', 'truth', 'dataset', 'promptVersion', 'model', 'model-fullname']
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


# Main function
def main():
    experiment_name = "base-experiment"
    models = ["meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct", "bigscience/bloomz-7b1-mt"]  # List of LLM models
    # "bigscience/bloomz-7b1-mt", "tiiuae/falcon-7b-instruct", "google/flan-t5-xl" "meta-llama/Llama-2-7b-chat-hf"

    # NOTE: left-hand-name must replace '/' with '-' to be a valid filename
    shortNames = {
        "meta-llama-Llama-2-7b-chat-hf": "Llama-2 7b",
        "tiiuae-falcon-7b-instruct": "Falcon 7b",
        "bigscience-bloomz-7b1-mt": "BloomZ 7b",
    }

    # Aggregate results and create DataFrame
    df = load_all_results(models, shortNames)

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
        create_report(df_dataset, f"{experiment_name}-{dataset}", metric_names, prompts)

    # create_report(df, experiment_name, metric_names)


if __name__ == "__main__":
    main()
