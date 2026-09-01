#!/bin/bash

# This script specifically extracts --config and --values arguments,
# appends the fixed seed to the --values argument, and runs the command repeatedly.

CONFIG_ARG=""
VALUES_BASE=""
OTHER_ARGS="" # for other args 

# Temporary array to hold the command-line arguments for parsing
ARGS=("$@")

while [[ ${#ARGS[@]} -gt 0 ]]; do
    case "${ARGS[0]}" in
        --config)
            CONFIG_ARG="--config ${ARGS[1]}"
            ARGS=("${ARGS[@]:2}") # Consume two arguments
            ;;
        --values)
            VALUES_BASE="${ARGS[1]}"
            ARGS=("${ARGS[@]:2}") # Consume two arguments
            ;;
        *)
            OTHER_ARGS+="${ARGS[0]} "
            ARGS=("${ARGS[@]:1}") # Consume one argument
            ;;
    esac
done

if [ -z "$CONFIG_ARG" ] && [ -z "$VALUES_BASE" ] && [ -z "$OTHER_ARGS" ]; then
    echo "⚠️  Note: No extra arguments passed to the script."
else
    echo "✅ Arguments successfully parsed."
    [ -n "$CONFIG_ARG" ] && echo "   - Config file: $(echo "$CONFIG_ARG" | cut -d ' ' -f 2)"
    [ -n "$VALUES_BASE" ] && echo "   - Base values: $VALUES_BASE"
    [ -n "$OTHER_ARGS" ] && echo "   - Other arguments: $OTHER_ARGS"
fi


execute_python_script() {
    local seed_value="$1"

    echo "🔥 Running experiment with seed: $seed_value"

    local FINAL_VALUES_ARG=""
    if [ -n "$VALUES_BASE" ]; then
        FINAL_VALUES_ARG="--values $VALUES_BASE seed=$seed_value"
    else
        FINAL_VALUES_ARG="--values seed=$seed_value"
    fi
    
    FULL_COMMAND="python train.py $CONFIG_ARG $FINAL_VALUES_ARG $OTHER_ARGS"

    # echo "SIMULATING: $FULL_COMMAND"
    $FULL_COMMAND
}


echo "--- Starting batch experiments over fixed seeds ---"

SEEDS=(42 101 3927 374024391 1702442591 751238365 1593226693 217519846 183184942 456748450)

for seed in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "🌱 Preparing run for seed: $seed"

    execute_python_script "$seed"

    # Pause between runs to handle system load
    sleep 1
done

echo "----------------------------------------"
echo "✨ All experiments finished successfully for ${#SEEDS[@]} seeds."
