#!/bin/zsh

# Given names of experiments, run the report.py script for each of them.

logfilename="run_reports.log"

#    versions-experiment \
#    few-shot-initial \
#    base-experiment-temperature \
#    base-experiment-separator-and-german-only \
#    prompt-experiment-large-variants-only \
#    prompt-experiment-large-llama2-vs-leolm \
#    prompt-experiment-large-all-llama-comparison-good-prompts \
#    least-to-most-prompting-stage2 \
#    mds-summarization-chain \
#    mds-2stage-experiment \
#    mds-baseline \
#    mds-cluster-chunks-experiment \
#    mds-cluster-chunks-vs-2stage-experiment \
#    mds-ordered-chunks-initial-1sentence \
#    mds-ordered-chunks-initial-overview \
#    mds-prefix-experiment \
#    mds-shuffling-and-annotation-experiment \
#    mds-summarization-chain \
#    few-shot-experiment-main-1536 \
#    few-shot-experiment-clustering \
#    few-shot-experiment-clustering-WikinewsExamples \
#    few-shot-experiment-full-20MinutesExamples \
#    few-shot-experiment-full-1024 \
#    few-shot-experiment-full-1536 \
#    few-shot-experiment-full-2048 \
#    few-shot-experiment-full \
#    few-shot-experiment-full-WikinewsExamples

for name in \
    mds-summarization-chain \
    mds-baseline
do
    # if an error occurred -> write the name into the log-file with [ERROR] in front of it
    # otherwise, write the name into the log-file with [SUCCESS] in front of it

    # if the log-file does not exist, create it
    if [ ! -f $logfilename ]; then
        touch $logfilename
    fi
    echo $(date) >> $logfilename
    echo "Starting $name" >> $logfilename

    # run python report.py --full --reload --name $name and check whether it was successful
    python report.py --full --reload --name $name
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $name" >> $logfilename
    else
        echo "[ERROR] $name" >> $logfilename
    fi
done