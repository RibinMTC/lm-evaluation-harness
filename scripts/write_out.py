import argparse

import jsonargparse
import numpy as np
import json
import os
import random

import lm_eval
from lm_eval import tasks, utils
from lm_eval.utils import join_iters, TaskConfig

EXAMPLE_DIVIDER = "!!@@##@@!! -- Example {i}\n"


def parse_args():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--task_configs", default=None)
    parser.add_argument("--prompt_version_per_task", type=str, default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fewshot_sampling", type=str, default="")
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--start_range", type=int, default=None,
                        help="The start index of the sample from which evaluation should begin")
    parser.add_argument("--end_range", type=int, default=None,
                        help="The end index of the sample from which evaluation should end")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--wandb_on", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--wandb_entity_name", type=str, default=None)
    parser.add_argument('--config', action=jsonargparse.ActionConfigFile)

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    model_args_dict = utils.simple_parse_args_string(args.model_args)
    if args.model in ["gpt3.5", "gpt4"]:
        model_id = model_args_dict["engine"]
    else:
        model_id = model_args_dict["pretrained"]

    task_config_list = []
    for task_config in args.task_configs:
        task_config_list.append(TaskConfig(**task_config))
    task_dict = lm_eval.tasks.get_task_dict_from_task_config(task_config_list=task_config_list, model_id=model_id)

    start_index = 0
    if args.start_range:
        start_index = args.start_range
    if args.end_range:
        end_index = args.end_range
    else:
        end_index = 1
        print("No end index specified, therefore only 1 sample will be generated.")
    num_examples = end_index - start_index

    if not args.output_path:
        args.output_path = "write_out_test"
        print(f"No output path specified, saving to default path {args.output_path}")

    os.makedirs(args.output_path, exist_ok=True)
    for task_name, task in task_dict.items():
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []

        # if task.has_training_docs():
        #     docs = task.training_docs()
        #     iters.append(docs)
        # if task.has_validation_docs():
        #     docs = task.validation_docs()
        #     iters.append(docs)
        if task.has_test_docs():
            docs = task.test_docs()
            iters.append(docs)

        docs = join_iters(iters)

        with open(os.path.join(args.output_path, task_name + f"_num_shot_{str(args.num_fewshot)}"), "w") as f:
            for i, doc in (
                    zip(range(num_examples), docs)
                    if num_examples > 0
                    else enumerate(docs)
            ):
                f.write(EXAMPLE_DIVIDER.format(i=i))
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                    rnd=rnd,
                    description=None,
                    fewshot_sampling=args.fewshot_sampling
                )
                f.write(ctx + "\n")


if __name__ == "__main__":
    main()
