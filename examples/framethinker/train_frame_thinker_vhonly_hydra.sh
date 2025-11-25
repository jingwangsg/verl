#!/bin/bash
# FrameThinker Video-Holmes Training Script (Using Hydra Config)
# This variant uses Hydra configuration for better maintainability

set -x

# Set ulimit to avoid "too many open files" error
ulimit -n 65535

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/framethinker/config"
TOOL_CONFIG_PATH="$PROJECT_DIR/recipe/framethinker/video_think_tool_config.yaml"

# Data paths (can be overridden with environment variables)
TRAIN_FILES=${TRAIN_FILES:-/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train.parquet}
VAL_FILES=${VAL_FILES:-/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test.parquet}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
PROJECT_NAME=${PROJECT_NAME:-video_holmes_rl}
EXP_NAME=${EXP_NAME:-framethinker_baseline_v2}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-/mnt/amlfs-02/shared/datasets/checkpoints/jingwang/video_reason/}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='framethinker_grpo' \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    $@
