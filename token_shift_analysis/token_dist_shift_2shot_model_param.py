import gc

import jsonargparse
import pandas as pd
import random
import os

import randomname
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm
import scipy
import nltk
from nltk import pos_tag, word_tokenize
import numpy

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="Dataset for analysis, must be in [arxiv, pubmed]")
    parser.add_argument("--model_1", required=True, type=str, help="Name of first model")
    parser.add_argument("--model_2", required=True, type=str, help="Name of second model",
                        default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--samples_to_compute", required=True, type=int, help="Number of samples to analyze [1,10]",
                        default=1)
    parser.add_argument('--config', action=jsonargparse.ActionConfigFile)

    return parser.parse_args()


def add_prompt(article):
    sys_prompt = "You are an expert at summarization. Proceed to summarize the following text. TEXT: "
    prompt = sys_prompt + article
    return prompt


def prepare_context(df, context: str = '0-shot', row: int = None, idx: int = None) -> List:
    if not row:
        row = random.randint(0, len(df))
    sample = df.iloc[row]

    pred = sample['prediction']
    if context == '2-shot':
        prompt = sample['prompt_0']
    else:
        article = sample['prompt_0'].split('TEXT:')[-1]
        prompt = add_prompt(article)
    if not idx:
        idx = [random.randint(0, len(pred.split())) for i in range(10)]

    pred_idx = [' '.join(map(str, pred.split()[:i])) for i in idx]
    model_input = [f"{prompt} SUMMARY: {p}" for p in pred_idx]
    return model_input, idx


def find_csv_files(folder_path, context='2-shot'):
    # List all files in the given folder
    files = os.listdir(folder_path)

    if context == '2-shot':
        # Filter files that contain '2-shot' and end with '.csv'
        csv_files = [file for file in files if '2-SHOT' in file and file.endswith('.csv')]
    else:
        csv_files = [file for file in files if '0-SHOT' in file and file.endswith('.csv')]

    # Print the filtered CSV file names
    for csv_file in csv_files:
        print(csv_file)
    return csv_files


def get_indices_by_pos(text, pos_tag_to_find):
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    indices = [i - 1 for i, (word, pos) in enumerate(tagged_words) if pos.startswith(pos_tag_to_find)]
    return indices


def find_indices_of_nouns_verbs_adjectives(text):
    noun_indices = get_indices_by_pos(text, 'N')
    verb_indices = get_indices_by_pos(text, 'V')
    adjective_indices = get_indices_by_pos(text, 'J')

    return noun_indices + adjective_indices + verb_indices


def get_token_idx(df, num_items_to_select=None, random_samples=False, use_all_tokens=False, use_domain_words=False):
    # define the tokens for which the distribution should be derived
    predictions_after = df['prediction'].tolist()
    pred = predictions_after[row_idx]
    idx = []

    # Or if you want to test across every nth samples of the text
    if random_samples:
        summary_words = len(pred.split())
        idx = [i for i in range(0, summary_words, 7)]

    elif use_all_tokens:
        summary_words = len(pred.split())
        idx = [i for i in range(0, summary_words)]
    # compute only for the words that are part of the domain vocabulary
    elif use_domain_words:
        # read the domain vocabulary
        path = f"{input_dir}/{ds}_top10000_vocabulary.txt"

        if not os.path.exists(path):
            print("Provided domain vocabulary path doesn't exist")
        else:
            with open(path, 'r') as file:
                # Create an empty list to store the lines
                domain_vocab = []
                # Iterate over the lines of the file
                for line in file:
                    # Remove the newline character at the end of the line
                    line = line.strip()
                    # Append the line to the list
                    domain_vocab.append(str(line))
            summary_words = pred.split()
            idx = [i for i, word in enumerate(summary_words) if word in domain_vocab]

    else:
        # selecting N random words from the sample for evaluating the distribution shift
        idx = find_indices_of_nouns_verbs_adjectives(pred)

    if num_items_to_select:
        if len(idx) > num_items_to_select:
            idx = idx[:num_items_to_select]
    return idx


def get_token_distribution(inputs, model_name: str, top_k=5, max_len=2048, truncation=True, verbose=False,
                           automatic_evaluation=False):
    top_k_values_all = []
    predicted_tokens_all = []
    probability_distribution_all = []

    model_hf_key = model_key[model_name]
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_hf_key)
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_hf_key, device_map="auto", torch_dtype=torch.bfloat16)

    for input in tqdm(inputs):
        # Tokenize input text
        input_ids = tokenizer.encode(input, return_tensors="pt", max_length=max_len, truncation=truncation).to("cuda")

        # Generate probabilities for the next token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Take the logits for the last token
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probability_distribution_all.append(probabilities)

        # Get the top-k predicted tokens and their probabilities

        top_k_values, top_k_indices = torch.topk(probabilities, k=top_k)

        # Convert indices back to tokens
        predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices[0].tolist())

        top_k_values_all.append(top_k_values)
        predicted_tokens_all.append(predicted_tokens)
    del model, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return top_k_values_all, predicted_tokens_all, probability_distribution_all


def token_dist_model1_model2(model_1, model_2, ds, num_items_to_select=None, random_samples=False, context='2-shot',
                             use_all_tokens=False, use_domain_words=False):
    # Read the data
    file_path = \
    df_runs.loc[(df_runs['dataset'] == ds) & (df_runs['model'] == "meta-llama-Llama-2-70b-chat-hf")].file_path.values[0]

    df_after = pd.read_csv(file_path)

    # Get samples across which tokens should be predicted
    idx = get_token_idx(df_after, num_items_to_select=num_items_to_select, random_samples=random_samples,
                        use_all_tokens=use_all_tokens, use_domain_words=use_domain_words)

    # prepare input for the model
    inputs, token_idx = prepare_context(df=df_after, context=context, row=row_idx, idx=idx)

    # Set the device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Getting the token distribution
    top_k_values_before, predicted_tokens_before, probability_distribution_all_before = get_token_distribution(inputs,
                                                                                                               model_name=model_1,
                                                                                                               top_k=50,
                                                                                                               max_len=4096,
                                                                                                               truncation=True,
                                                                                                               verbose=False,
                                                                                                               automatic_evaluation=True)

    # Getting the token distribution
    top_k_values_after, predicted_tokens_after, probability_distribution_all_after = get_token_distribution(inputs,
                                                                                                            model_name=model_2,
                                                                                                            top_k=50,
                                                                                                            max_len=4096,
                                                                                                            truncation=True,
                                                                                                            verbose=False,
                                                                                                            automatic_evaluation=True)

    # Automatic Evaluation for Token Distribution Shift
    eval_scores = dict()

    # Calculate KL Divergence
    kl_divergence = []
    frequently_shift_tokens = []
    # compute for all tokens of the sample
    for prob_dist_bef, prob_dist_aft in zip(probability_distribution_all_before, probability_distribution_all_after):
        kl_div = scipy.special.kl_div(prob_dist_bef.cpu(), prob_dist_aft.cpu())
        kl_div = numpy.sum(kl_div.numpy())
        # print("KL Divergence: ", kl_div)
        kl_divergence.append(kl_div)

    eval_scores['kl_divergence'] = kl_divergence
    eval_scores['kl_divergence_mean'] = numpy.average(kl_divergence)

    token_shift = []
    base_rank = []
    base_prob = []

    unshifted = 0
    marginal_shift = 1
    shifted = 2
    for predicted_token_before, top_k_value_before, predicted_token_after, top_k_value_after in zip(
            predicted_tokens_before, top_k_values_before, predicted_tokens_after, top_k_values_after):

        top_k_value_before = top_k_value_before[0]
        top_k_value_after = top_k_value_after[0]
        # print( top_k_values_before)
        # print (predicted_tokens_before)

        # print (top_k_values_after)
        # print (predicted_tokens_after)

        # Calculate token shift rate
        if predicted_token_before[0] == predicted_token_after[0]:
            # print(predicted_token_before[0], predicted_token_after[0])
            # print("Unshifted")
            token_shift.append(unshifted)
        elif predicted_token_after[0] in predicted_token_before[:3]:
            # print(predicted_token_after[0], predicted_token_before[:3])
            # print("marginal shift")
            token_shift.append(marginal_shift)
            frequently_shift_tokens.append(predicted_token_after[0])
        else:
            # print(predicted_token_after[:5], predicted_token_before[:5])
            token_shift.append(shifted)
            # print("shifted")

        # Base rank of token
        try:
            rank = predicted_token_before.index(predicted_token_after[0])
        except:
            rank = -1
        base_rank.append(rank)

        # Base Probability of token
        if rank == -1:
            base_prob.append(0)
        else:
            base_prob.append(top_k_value_before[rank].item())

    eval_scores['token_shift'] = token_shift
    eval_scores['token_shift_rate'] = token_shift.count(shifted) / len(token_shift)

    eval_scores['base_rank'] = base_rank
    eval_scores['base_rank_mean'] = numpy.average(base_rank)

    eval_scores['base_prob'] = base_prob
    eval_scores['base_prob_mean'] = numpy.average(base_prob)

    eval_scores['token_idx'] = token_idx
    eval_scores['freq_shifted_tokens'] = frequently_shift_tokens

    return eval_scores


# def connect_to_wandb():
#     wandb_token_key: str = "WANDB_TOKEN"
#
#     load_dotenv()
#     # wandb setup
#     wandb_tok = os.environ.get(wandb_token_key)
#     assert wandb_tok and wandb_tok != "<wb_token>", "Wandb token is not defined"
#     wandb.login(key=wandb_tok)


def push_to_wandb(eval_scores: dict, wandb_project_name="domain-adaptation-token-distribution-shift",
                  entity='background-tool'):
    # Prepare config
    model = f"{model_1}-{model_2}"
    sample_id = row_idx
    task = "2-SHOT"
    model_family = model if len(model.split('-')) == 1 else ''.join(model.split('-')[:2])
    config = {'model': model, 'dataset': ds, 'task': task, 'model_family': model_family}
    wandb_run_name = '_'.join([model, task, ds, str(sample_id)])
    wandb_run_name = run_random_name + '_' + wandb_run_name

    wandb_mode = "online"

    wandb.init(project=wandb_project_name, entity=entity, config=config, name=wandb_run_name,
               mode=wandb_mode, group=ds)

    # compute plots for all

    # Token dist shift
    df_token_shift = pd.DataFrame(eval_scores['token_shift'])
    table = wandb.Table(dataframe=df_token_shift)
    wandb.log({f"token dist shift table": table}, commit=True)

    data = [[label, val] for (label, val) in dict(Counter(eval_scores['token_shift'])).items()]
    table = wandb.Table(data=data, columns=["label", "value"])
    wandb.log(
        {
            "token_dist_shift": wandb.plot.bar(
                table, "label", "value", title="Token Distribution Shift"
            )
        }
    )

    # Freq shifted tokens
    df_freq_shifted_tokens = pd.DataFrame(eval_scores['freq_shifted_tokens'])
    table = wandb.Table(dataframe=df_freq_shifted_tokens)
    wandb.log({f"frequently shifted tokens table": table}, commit=True)

    # plot KL divergence
    data = [[x, y] for (x, y) in zip(eval_scores["token_idx"], eval_scores["kl_divergence"])]
    table = wandb.Table(data=data, columns=["Position", "KL Divergence"])
    wandb.log({"KL Div plot": wandb.plot.scatter(table, "Position", "KL Divergence")}, commit=True)

    # Plot base rank
    data = [[x, y] for (x, y) in zip(eval_scores["token_idx"], eval_scores["base_rank"])]
    table = wandb.Table(data=data, columns=["Position", "Base Rank"])
    wandb.log({"Base Rank plot": wandb.plot.scatter(table, "Position", "Base Rank")}, commit=True)

    # plot base prob
    data = [[x, y] for (x, y) in zip(eval_scores["token_idx"], eval_scores["base_prob"])]
    table = wandb.Table(data=data, columns=["Position", "Base Prob"])
    wandb.log({"Base Probability plot": wandb.plot.scatter(table, "Position", "Base Prob")}, commit=True)

    # Delete these as logging them as lists would create problems
    eval_scores.pop("token_shift")
    eval_scores.pop("base_rank")
    eval_scores.pop("base_prob")
    eval_scores.pop("kl_divergence")
    eval_scores.pop("token_idx")
    eval_scores.pop('freq_shifted_tokens')

    df = pd.DataFrame(eval_scores, index=[0])
    df.insert(0, "model", [model])
    df.insert(1, "dataset", [ds])
    df.insert(2, "task", [task])
    df.insert(3, "sample_id", [sample_id])
    table = wandb.Table(dataframe=df)
    wandb.log({f"eval_scores_sample_{sample_id}_table": table}, commit=True)

    wandb.finish()


args = parse_args()

load_dotenv()
# connect_to_wandb()

run_random_name = randomname.get_name()
ds = args.dataset
context = '2-shot'
samples_to_compute = args.samples_to_compute
# Set the folder path
input_dir = 'token_shift_analysis/token_shift_analysis_config'
model_1 = args.model_1
model_2 = args.model_2

# Call the function to find and print the matching CSV file names
two_shot_runs = find_csv_files(input_dir, context=context)

df_runs = pd.DataFrame(columns=['run_id', 'model', 'model_family', 'dataset', 'task', 'file_path'])

for i, run in enumerate(two_shot_runs):
    run_id = run.split("_")[0]
    model = run.split("_")[2]
    model_family = model if len(model.split('-')) == 1 else ''.join(model.split('-')[:2])
    dataset = run.split("_")[4]
    task = run.split("_")[5]
    file_path = os.path.join(input_dir, run)

    df_runs.loc[i] = [run_id, model, model_family, dataset, task, file_path]

model_key = {
    "meta-llama-Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama-Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama-Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",
    "mistralai-Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "lmsys-vicuna-7b-v1.5-16k": "lmsys/vicuna-7b-v1.5-16k",
    "lmsys-vicuna-13b-v1.5-16k": "lmsys/vicuna-13b-v1.5-16k",
}

# randomly selecting one sample each dataset across which all models are selected
pubmed_two_shot_samples = 426
arxiv_two_shot_samples = 97
pubmed_samples = 6592
arxiv_samples = 6438
govreport_samples = 972

if ds == 'govreport':
    num_samples = govreport_samples
elif context == '2-shot':
    num_samples = arxiv_two_shot_samples if ds == 'arxiv' else pubmed_two_shot_samples
else:
    num_samples = arxiv_samples if ds == 'arxiv' else pubmed_samples

s = list(range(num_samples))
random.shuffle(s)
samples = s[-samples_to_compute:]

for i in range(len(samples)):
    row_idx = i
    print(ds, row_idx)

    eval_scores = token_dist_model1_model2(model_1=model_1, model_2=model_2, ds=ds, use_domain_words=True)
    # use_all_tokens=True,  num_items_to_select=3, random_samples = False,)num_items_to_select=5, random_samples = True)
    push_to_wandb(eval_scores)
