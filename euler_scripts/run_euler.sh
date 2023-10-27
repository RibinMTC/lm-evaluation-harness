# Provide as first argument the path to the euler config file


if [ -z "$1" ]
  then
    echo "No euler_config.json path provided"
    exit 1
fi

CURR_BRANCH=$(git rev-parse --abbrev-ref HEAD)

LOAD_MODULES=$(python3 utils/euler_config_parser.py \
                       --euler_config_path "$1" \
                       --retreive_key load_modules)

PROJECT_PATH=$(python3 utils/euler_config_parser.py \
                       --euler_config_path "$1" \
                       --retreive_key project_path)

CODE_PATH=$(python3 utils/euler_config_parser.py \
                --euler_config_path "$1" \
                --retreive_key code_path)

if [ "$#" -ge 2 ]; then
    CONFIG_FILE=$2
else
    CONFIG_FILE=$(python3 utils/euler_config_parser.py \
                --euler_config_path "$1" \
                --retreive_key config_file)
fi


scp "$1" euler:$CODE_PATH/euler_scripts
#scp ../lm_eval/tasks/seahorse_classification.py euler:$CODE_PATH/lm_eval/tasks/seahorse_classification.py
#scp ../lm_eval/tasks/swisstext23_faithfulness_classification.py euler:$CODE_PATH/lm_eval/tasks/swisstext23_faithfulness_classification.py
#scp ../lm_eval/tasks/swisstext23_summarization.py euler:$CODE_PATH/lm_eval/tasks/swisstext23_summarization.py
#scp ../lm_eval/tasks/faithfulness_classification_base_task.py euler:$CODE_PATH/lm_eval/tasks/faithfulness_classification_base_task.py
#scp ../lm_eval/evaluator.py euler:$CODE_PATH/lm_eval/evaluator.py
#scp ../configs/prompt_templates/swisstext23_summarization.json euler:$CODE_PATH/configs/prompt_templates/swisstext23_summarization.json
#scp ../configs/prompt_templates/seahorse_classification.json euler:$CODE_PATH/configs/prompt_templates/seahorse_classification.json
#scp ../configs/prompt_templates/faithfulness_benchmark_final_swisstext23_multi_label.json euler:$CODE_PATH/configs/prompt_templates/faithfulness_benchmark_final_swisstext23_multi_label.json
#scp ../configs/prompt_templates/faithfulness_benchmark_final_swisstext23_benchmark.json euler:$CODE_PATH/configs/prompt_templates/faithfulness_benchmark_final_swisstext23_benchmark.json
PARENT_DIR=$(dirname "$CONFIG_FILE")
#echo "parent dir: $PARENT_DIR"
echo "config file: $CONFIG_FILE"
ssh euler "
    if [[ ! -d $CODE_PATH/$PARENT_DIR ]]; then
        mkdir -p $CODE_PATH/$PARENT_DIR && echo 'Directory created'
    else
        echo 'Directory already exists'
    fi
  "
scp ../"$CONFIG_FILE" euler:$CODE_PATH/$CONFIG_FILE

ssh euler ARG1=\"$1\" \
          ARG4=\"$LOAD_MODULES\" \
          ARG5=\"$CODE_PATH\" \
          ARG6=\"$CURR_BRANCH\" \
          ARG7=\"$PROJECT_PATH\" \
          ARG8=\"$CONFIG_FILE\" \
          'bash -s' <<'ENDSSH'

    # Change to work dir
    echo "### Changing to project dir..."
    cd "$ARG7" || exit

    # Export to avoid relativ folder import errors in python
    echo "### Add to python path: $ARG7/"
    export PYTHONPATH="${PYTHONPATH}:$ARG7/"

    # Load all updates
#    echo "### Pulling commits..."
#    echo ""
#    git pull
#    git checkout "$ARG6"
#    echo ""

    # LOAD MODULES AFTER ACTIVATING ENVIRONMENT TO AVOID LIBRARY ERRORS!
    echo "### Loading modules..."
    eval module load "$ARG4"
    # UNLOADING ETH_PROXY MODULES FOR WANDB
    # echo "Unloading eth_proxy module for wandb"
    # eval module unload eth_proxy
    echo ""

    # Activate environment
    echo "### Activating environment..."
    source env/bin/activate
    echo ""

    # Change to code dir
    echo "### Changing to code dir..."
    cd "$ARG5" || exit
    echo ""

    # Get commands
    echo "### Retrieving commands to execute..."
    echo "$ARG1"
    STR_COMMAND=$(python3 euler_scripts/utils/bash_euler_commands_helper.py --config_path euler_scripts/"$ARG1")
    echo ""

    # Run all the commands
    echo "### Running command..."
    CMD="$STR_COMMAND $ARG8 \""
    echo "$CMD"
    eval "$CMD"
    echo ""
    sleep 1

ENDSSH