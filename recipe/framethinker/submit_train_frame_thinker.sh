#!/bin/bash
# Ray job submission script for FrameThinker training
# Usage:
#   EXP_NAME=framethinker_test bash examples/framethinker/submit_train_frame_thinker.sh
#   MODEL_PATH=path/to/model EXP_NAME=framethinker_exp bash examples/framethinker/submit_train_frame_thinker.sh

set -x

# Set Ray address (default to local cluster)
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8300"}

# Set experiment name
EXP_NAME=${EXP_NAME:-framethinker_baseline_v2}
export EXP_NAME

# Set model path if provided
if [ -n "$MODEL_PATH" ]; then
    export MODEL_PATH
fi

# Set data files if provided
if [ -n "$TRAIN_FILES" ]; then
    export TRAIN_FILES
fi
if [ -n "$VAL_FILES" ]; then
    export VAL_FILES
fi

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Path to runtime environment YAML (relative to project root)
RUNTIME_ENV_YAML="$PROJECT_ROOT/../runtime_env.yaml"

# Verify runtime environment file exists
if [ ! -f "$RUNTIME_ENV_YAML" ]; then
    echo "Error: Runtime environment file not found at $RUNTIME_ENV_YAML"
    exit 1
fi

# Submit the job to Ray
# Note: runtime_env.yaml already sets working_dir to verl/, so no need to cd
RAY_ADDRESS="$RAY_ADDRESS" ray job submit \
    --runtime-env "$RUNTIME_ENV_YAML" \
    --submission-id "framethinker-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)" \
    -- bash -c "EXP_NAME=$EXP_NAME bash examples/framethinker/train_frame_thinker_vhonly.sh"
