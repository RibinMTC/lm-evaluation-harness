#!/bin/bash

#7b models: models=("LeoLM-leo-mistral-hessianai-7b" "NousResearch-Llama-2-7b-hf" "LeoLM-leo-hessianai-7b")
#13b models: models=("NousResearch-Llama-2-13b-hf" "LeoLM-leo-hessianai-13b" "mtc-LeoLM-leo-hessianai-13b-all-labels-classification-one-epoch-qlora-4bit-merged") #"mtc-LeoLM-leo-hessianai-13b-all-labels-classification-english-one-epoch-qlora-4bit-merged"
models=("mtc-LeoLM-leo-hessianai-13b-faithfulness-only-classification-english-one-epoch-qlora-4bit-merged") #("NousResearch-Llama-2-7b-hf" "NousResearch-Llama-2-13b-hf" "google-flan-ul2" "LeoLM-leo-mistral-hessianai-7b" "LeoLM-leo-hessianai-7b" "LeoLM-leo-hessianai-13b" "mtc-LeoLM-leo-hessianai-13b-all-labels-classification-one-epoch-qlora-4bit-merged")
num_shots=("0") # ("0" "3")
BASE_CONFIG_DIR="../configs/faithfulness_benchmark_final_configs/{model}/num_fewshot_{num_shots}"

for model in "${models[@]}"; do
   if [[ "$model" == */* ]]; then
    echo "Error: Model name '$model' contains a forward slash (/), which is not allowed."
    exit 1
  fi
  for num_shot in "${num_shots[@]}"; do
     # Format the base string with the 'model' and 'num_shot' variables
    formatted_config_dir=${BASE_CONFIG_DIR//\{model\}/$model}
    formatted_config_dir=${formatted_config_dir//\{num_shots\}/$num_shot}
    echo "Formatted string: $formatted_config_dir"
    # Iterate over each generated config file in the directory
    for config_file in $formatted_config_dir/*; do
        # Run the run_eulers.sh script with the config file as an argument
        cleaned_path=${config_file#../}
        bash run_euler.sh lm_eval_euler_config.json $cleaned_path
        # Wait for the script to complete before moving on to the next config file
        wait
    done
  done
done

