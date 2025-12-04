set -ex

BENCHMARK_ROOT="/mnt/amlfs-02/shared/datasets/s3/video_reason/BENCHMARK/"

# BENCHMARKS (MCQ-only)
LVR="${BENCHMARK_ROOT}/LongVideoReason/test.parquet"
LVB="${BENCHMARK_ROOT}/LongVideoBench/test.parquet"
VH="${BENCHMARK_ROOT}/Video-Holmes/test.parquet"
MMRV="${BENCHMARK_ROOT}/MMR-V/test.parquet"
MVB="${BENCHMARK_ROOT}/MVBench/test.parquet"
VRB="${BENCHMARK_ROOT}/VRBench/test.parquet"
TEMP="${BENCHMARK_ROOT}/TempCompass/test.parquet"
VIDEOMME="${BENCHMARK_ROOT}/Video-MME/test.parquet"

# MODEL PATH
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "data.train_files=[${LVR}]" \
    "data.val_files=[${MMRV}]" \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.media_dir=/mnt/amlfs-02/shared/datasets/s3/video_reason/ \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=8 \
    data.media_reading_kwargs.enable=True \
    data.media_reading_kwargs.num_frames=8 \
    data.media_reading_kwargs.size=360 \
    data.media_reading_kwargs.sampling_mode=uniform \
    algorithm.adv_estimator=grpo \
    \
    data.message_template=default \
    custom_reward_function.path=verl/utils/reward_score/video_vanilla.py \
    custom_reward_function.name=compute_score \
    \
    data.val_batch_size=32 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.logger='["console"]' \
    \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2