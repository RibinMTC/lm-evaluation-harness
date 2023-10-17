import collections
import itertools
import numpy as np
import random
# import lm_eval.metrics
# import lm_eval.models
# import lm_eval.tasks
# import lm_eval.base
# from lm_eval.utils import positional_deprecated, run_task_tests
import json
import pathlib


exclude_folders = ['BACKUP-.*', 'OLD-BAG']
SRC_FOLDER = '../results_bag'
DST_FOLDER = '../results_bag_extended'

models = [
    'mtc/NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged',
    'mtc/NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged',
    'mtc/NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged'
]

base_prompt_train = "### Instruction:\n{sub_task_prompt}\nSummary:{{summary}}\n\n### Assistant:\n{{label}}"
base_prompt = "### Instruction:\n{sub_task_prompt}\nSummary:{{summary}}\n\n### Assistant:\n"
subtask_prompts = {
    "attribution": "Given a German article and its summary, determine if all information in the summary originates from the article. Return True if it is, and False if not.\nArticle: {article}",
    "concise": "Is the provided German summary a concise and accurate representation of the given German article? Return True if it is, and False if not.\nArticle: {article}",
    "main_ideas": "Does the provided German summary capture the main ideas of the German article? Return True if the summary captures the main ideas, and False if not.\nArticle: {article}",
}


if __name__ == "__main__":
    # list the source-folder, print the subfolders along with enumeration, and ask the user to select one
    src_folder = pathlib.Path(SRC_FOLDER)
    dst_folder = pathlib.Path(DST_FOLDER)
    dst_folder.mkdir(parents=True, exist_ok=True)
    src_folders = [x for x in src_folder.iterdir() if x.is_dir() and not any([x.match(y) for y in exclude_folders])]
    print("Select a folder:")
    for i, folder in enumerate(src_folders):
        number = "[%2s]" % i
        print(f"{number}: {folder}")
    folder_index = int(input("Enter the number of the folder: "))
    src_folder = src_folders[folder_index]
    dst_folder = dst_folder / src_folder.name

    print("Hello World")
