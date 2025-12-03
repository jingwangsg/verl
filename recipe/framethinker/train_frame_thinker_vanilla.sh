#!/bin/bash
# FrameThinker Debug Training Script
# Minimal configuration for testing the training setup
# Uses very small batch sizes and limited data for quick validation

# use swanlab
swanlab login --api-key <api_key>

# use wandb
# wandb login <api_key>
# export WANDB_ENTITY="<entity_name>"

set -x
ulimit -n 65535

ray stop
ray start --head --resources='{"drivers": 1}'

# Get absolute path to project root
PROJECT_DIR="$(pwd)"
MEDIA_DIA=$PROJECT_DIR/data/video_reason/

# Data paths (can be overridden with environment variables)
vh_train_path=$PROJECT_DIR/data/video_reason/Video-Holmes/train.parquet
TRAIN_FILES="['$vh_train_path']"

vh_test_path=$PROJECT_DIR/data/video_reason/Video-Holmes/test.parquet
VAL_FILES="['$vh_test_path']"

# Hyperparameters
train_batch_size=32
num_frames=8
lr=2e-6
message_template=default

if [[ "$message_template" == "framethinker_add_zoomin" ]]; then
    name_part="zoomin"
elif [[ "$message_template" == "framethinker_default" ]]; then
    name_part="default"
elif [[ "$message_template" == "default" ]]; then
    name_part="grpo"
else
    echo "Error: unknown message_template: $message_template"
    exit 1
fi

# Model and save paths
MODEL_PATH=$PROJECT_DIR/model_weights/Qwen2.5-VL-7B-Instruct
PROJECT_NAME=framethinker_verl
EXP_NAME=framethinker_${name_part}_bsz${train_batch_size}_${num_frames}frames_${lr}
SAVE_CHECKPOINT_DIR=$PROJECT_DIR/checkpoints/video_reason/

# Hyperparams
DTYPE=float16
MODEL_DTYPE=fp32

# Debug configuration: minimal resources
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.media_dir=${MEDIA_DIA} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=8 \
    data.message_template=${message_template} \
    data.media_reading_kwargs.enable=True \
    data.media_reading_kwargs.num_frames=${num_frames} \
    data.media_reading_kwargs.size=360 \
    data.media_reading_kwargs.sampling_mode=uniform \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.dtype=$DTYPE \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=20 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.eps=1e-15 \
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward_vanilla.py \
    custom_reward_function.name=compute_score \
    $@