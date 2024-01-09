#!/bin/bash

# NOTE:
# - This script assumes that ssh keys have been set up for the remote host
# - ... and that the remote host is called "euler"

remote_path="/cluster/home/roysc/roysc/lm-evaluation-harness/results"
outputs_path="/cluster/home/roysc/roysc/lm-evaluation-harness/euler_outputs"
local_folder="results"
outputs_folder="euler_outputs"

# Special folders
attribution_path="mtc-NousResearch-Llama-2-7b-hf-attribution-with-target-modules-qlora-4bit-merged"
conciseness_path="mtc-NousResearch-Llama-2-7b-hf-conciseness-with-target-modules-qlora-4bit-merged"
mainIdeas_path="mtc-NousResearch-Llama-2-7b-hf-main-ideas-with-target-modules-qlora-4bit-merged"
scratch_path="/cluster/home/roysc/scratch/results"


# Using rsync to synchronize
# rsync -ruav "euler:${remote_path}/" "${local_folder}/"
rsync -ruav "euler:${scratch_path}/" "${local_folder}/"
rsync -ruav "euler:${outputs_path}/" "${outputs_folder}/"

