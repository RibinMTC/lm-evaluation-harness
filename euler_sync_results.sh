#!/bin/bash

# NOTE:
# - This script assumes that ssh keys have been set up for the remote host
# - ... and that the remote host is called "euler"

remote_path="/cluster/home/roysc/roysc/lm-evaluation-harness/results"
outputs_path="/cluster/home/roysc/roysc/lm-evaluation-harness/euler_outputs"
local_folder="results"
outputs_folder="euler_outputs"

# Using rsync to synchronize
rsync -r "euler:${remote_path}/" "${local_folder}/"
rsync -r "euler:${outputs_path}/" "${outputs_folder}/"
