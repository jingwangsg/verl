#!/bin/bash
# Ray job submission script for FrameThinker debug/testing
# Usage:
#   bash examples/framethinker/submit_train_debug.sh
#   EXP_NAME=my_test bash examples/framethinker/submit_train_debug.sh

set -x

# Set Ray address (default to local cluster)
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8300"}

# Set experiment name for debug
EXP_NAME=${EXP_NAME:-framethinker_debug}
export EXP_NAME

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Path to runtime environment YAML
RUNTIME_ENV_YAML="$PROJECT_ROOT/../runtime_env.yaml"

# Verify runtime environment file exists
if [ ! -f "$RUNTIME_ENV_YAML" ]; then
    echo "Error: Runtime environment file not found at $RUNTIME_ENV_YAML"
    exit 1
fi

echo "Submitting debug job to Ray cluster at $RAY_ADDRESS"
echo "Experiment name: $EXP_NAME"
echo "Runtime environment: $RUNTIME_ENV_YAML"

# Submit the job to Ray
# Note: runtime_env.yaml already sets working_dir to verl/, so no need to cd
RAY_ADDRESS="$RAY_ADDRESS" ray job submit \
    --runtime-env "$RUNTIME_ENV_YAML" \
    --submission-id "framethinker-debug-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)" \
    -- bash -c "EXP_NAME=$EXP_NAME bash examples/framethinker/train_frame_thinker_debug.sh"
