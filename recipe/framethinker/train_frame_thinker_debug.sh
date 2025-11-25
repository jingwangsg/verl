#!/bin/bash
# FrameThinker Debug Training Script
# Minimal configuration for testing the training setup
# Uses very small batch sizes and limited data for quick validation
swanlab login --api-key iPPnOFirR3dXCBgOBPqiB

set -x
ulimit -n 65535

# Get absolute path to project root
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/framethinker/config"
TOOL_CONFIG_PATH="$CONFIG_PATH/video_think_tool_config.yaml"

# Data paths (can be overridden with environment variables)
TRAIN_FILES=${TRAIN_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/train.parquet}
VAL_FILES=${VAL_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/test.parquet}
MEDIA_DIA=${MEDIA_DIA:-$PROJECT_DIR/data/video_reason/}

# Model and save paths
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/model_weights/Qwen2.5-VL-7B-Instruct}
PROJECT_NAME=${PROJECT_NAME:-video_holmes_verl}
EXP_NAME=${EXP_NAME:-framethinker_verl_debug}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/video_reason/}

# Hyperparams
DTYPE=float16
MODEL_DTYPE=float16

# Debug configuration: minimal resources
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.media_dir=${MEDIA_DIA} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=8 \
    data.message_template=framethinker_default \
    data.media_reading_kwargs.enable=True \
    data.media_reading_kwargs.num_frames=8 \
    data.media_reading_kwargs.size=360 \
    data.media_reading_kwargs.sampling_mode=uniform \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.limit_images=128 \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=8 \
    actor_rollout_ref.rollout.multi_turn.format=think_with_video \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1 \
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.nframes=8 \
    +custom_reward_function.reward_kwargs.lambda_gfn=0.2 \
    +custom_reward_function.reward_kwargs.lambda_cf=0.0 \
    +custom_reward_function.reward_kwargs.lambda_zoom=0.1 \
    $@
