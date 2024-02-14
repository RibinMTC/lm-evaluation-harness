#!/bin/zsh

# Given names of experiments, run the report.py script for each of them.

# Unused experiment names
#   mds-baseline-german

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
#    mds-baseline-german \
#    mds-cluster-chunks-experiment \
#    mds-cluster-chunks-vs-2stage-experiment \
#    mds-ordered-chunks-initial-1sentence \
#    mds-ordered-chunks-initial-overview \
#    mds-prefix-experiment \
#    mds-shuffling-and-annotation-experiment \
#    few-shot-experiment-main-1536 \
#    few-shot-experiment-clustering \
#    few-shot-experiment-clustering-WikinewsExamples \
#    few-shot-experiment-full-20MinutesExamples \
#    few-shot-experiment-full-1024 \
#    few-shot-experiment-full-1536 \
#    few-shot-experiment-full-2048 \
#    few-shot-experiment-full \
#    few-shot-experiment-full-WikinewsExamples

#    mds-summarization-chain \
#    mds-2stage-experiment \
#    mds-baseline \
#    mds-cluster-chunks-experiment \
#    mds-cluster-chunks-vs-2stage-experiment \
#    mds-ordered-chunks-initial-1sentence \
#    mds-ordered-chunks-initial-overview \
#    mds-prefix-experiment \
#    mds-shuffling-and-annotation-experiment \
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
    mds-summarization-chain-comparison \
    few-shot-experiment-main # \
#    few-shot-experiment-clustering \
#    few-shot-experiment-clustering-BEST \
#    few-shot-experiment-distMMR \
#    few-shot-experiment-distMMR-BEST \
#    few-shot-experiment-20Min-vs-Wiki-Examples \
#    few-shot-experiment-baselines
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
#    python report.py --full --reload  --skip_catplots --skip_violins --skip_cost_estimate --skip_metric_plots --skip_seahorse_plots --skip_length_statistics --skip_inspect_examples --skip_language_statistics --skip_failure_statistics --name $name
    python report.py --full --reload  --skip_catplots --skip_violins --skip_cost_estimate --skip_length_statistics --skip_inspect_examples --skip_language_statistics --skip_failure_statistics --name $name
    #    python report.py --full --reload --skip_catplots --skip_violins --skip_cost_estimate --skip_language_statistics --name $name
    #    python report.py --full --reload  --skip_catplots --skip_violins --skip_cost_estimate --skip_length_statistics --skip_inspect_examples --name $name
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $name" >> $logfilename
    else
        echo "[ERROR] $name" >> $logfilename
    fi
done