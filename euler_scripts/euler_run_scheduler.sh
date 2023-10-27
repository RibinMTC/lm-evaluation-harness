#!/bin/bash

#7b models: models=("LeoLM-leo-mistral-hessianai-7b" "NousResearch-Llama-2-7b-hf" "LeoLM-leo-hessianai-7b")
#13b models: models=("NousResearch-Llama-2-13b-hf" "LeoLM-leo-hessianai-13b" "mtc-LeoLM-leo-hessianai-13b-all-labels-classification-one-epoch-qlora-4bit-merged") #"mtc-LeoLM-leo-hessianai-13b-all-labels-classification-english-one-epoch-qlora-4bit-merged"
#models=("LeoLM-leo-hessianai-13b") #("NousResearch-Llama-2-7b-hf" "NousResearch-Llama-2-13b-hf" "google-flan-ul2" "LeoLM-leo-mistral-hessianai-7b" "LeoLM-leo-hessianai-7b" "LeoLM-leo-hessianai-13b" "mtc-LeoLM-leo-hessianai-13b-all-labels-classification-one-epoch-qlora-4bit-merged")
#num_shots=("0" "3") # ("0" "3")
#fewshot_strategies=("stratified")
#seeds=(5 10)
models=($1)
num_shots=($2)
fewshot_strategies=($3)
seeds=($4)
BASE_CONFIG_DIR="configs/faithfulness_benchmark_final_configs/{model}/num_fewshot_{num_shots}/fewshot_strategy_{fewshot_strategy}/seed_{seed}"

for model in "${models[@]}"; do
  model=${model//\//-}
  for num_shot in "${num_shots[@]}"; do
    for fewshot_strategy in "${fewshot_strategies[@]}"; do
      for seed in "${seeds[@]}"; do
         # Format the base string with the 'model' and 'num_shot' variables
        formatted_config_dir=${BASE_CONFIG_DIR//\{model\}/$model}
        formatted_config_dir=${formatted_config_dir//\{num_shots\}/$num_shot}
        formatted_config_dir=${formatted_config_dir//\{fewshot_strategy\}/$fewshot_strategy}
        formatted_config_dir=${formatted_config_dir//\{seed\}/$seed}

        if [ ! -d "$formatted_config_dir" ]; then
            echo "!! Directory does not exist: $formatted_config_dir"
            continue
        fi
        # Iterate over each generated config file in the directory
        for config_file in $formatted_config_dir/*; do
            # Run the run_eulers.sh script with the config file as an argument
            cd euler_scripts
            bash run_euler.sh lm_eval_euler_config.json $config_file
            # Wait for the script to complete before moving on to the next config file
            wait
            cd ..
        done
      done
    done
  done
done

