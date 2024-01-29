#!/bin/bash

selected_config_files=($1)

for selected_config_file in "${selected_config_files[@]}"; do

  if [ ! -f "$selected_config_file" ]; then
      echo "!! Config file does not exist: $selected_config_file"
      continue
  fi
  # Iterate over each generated config file in the directory
  cd euler_scripts
  bash run_euler.sh lm_eval_euler_config.json $selected_config_file
  # Wait for the script to complete before moving on to the next config file
  wait
  cd ..


done

