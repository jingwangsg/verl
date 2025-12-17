set BENCHMARK_ROOT /mnt/amlfs-02/shared/datasets/s3/video_reason/BENCHMARK/
set TRAIN_ROOT /mnt/amlfs-02/shared/datasets/s3/video_reason/TRAIN/
# BENCHMARKS (MCQ-only)
set LVBEN $BENCHMARK_ROOT/LVBench/test.parquet
set LVR $BENCHMARK_ROOT/LongVideoReason/test.parquet
set LVB $BENCHMARK_ROOT/LongVideoBench/test.parquet
set VH $BENCHMARK_ROOT/Video-Holmes/test.parquet
set MMRV $BENCHMARK_ROOT/MMR-V/test.parquet
set MVB $BENCHMARK_ROOT/MVBench/test.parquet
set VRB $BENCHMARK_ROOT/VRBench/test.parquet
set TEMP $BENCHMARK_ROOT/TempCompass/test.parquet
set VMME $BENCHMARK_ROOT/VideoMME/test.parquet
set VMMMU_MCQ $BENCHMARK_ROOT/VideoMMMU/test_mcq.parquet
set VMQA $BENCHMARK_ROOT/VideoMathQA/test.parquet
set MLVU_MCQ $BENCHMARK_ROOT/MLVU/test_mcq.parquet

# TRAIN
set LVR_FULL $TRAIN_ROOT/LongVideoReason/train.parquet
set LVR_3K $TRAIN_ROOT/LongVideoReason/train_mcq_3k.parquet
set LVR_10K $TRAIN_ROOT/LongVideoReason/train_mcq_10k.parquet
set VH_TRAIN $TRAIN_ROOT/Video-Holmes/train.parquet
set VR1_VIDEO_20K $TRAIN_ROOT/Video-R1/train_video_mcq_20k.parquet
# set LVR_20K $BENCHMARK_ROOT/LongVideoReason/train_mcq_20k.parquet

EXP=grpo_baseline_vh+lvr10k+vr1_20k_n8 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env/train.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP \
    TRAIN_FILES=\"$VH_TRAIN,$LVR_10K,$VR1_VIDEO_20K\" \
    VAL_FILES=\"$LVR,$LVB,$LVBEN,$MVB,$VRB,$TEMP,$VH,$MMRV,$VMME,$VMMMU_MCQ,$VMQA,$MLVU_MCQ\" \
    bash examples/framethinker/train_grpo_baselines.sh \
    trainer.test_freq=100
    """

EXP=grpo_baseline_vh+lvrfull+vr1_20k_n16 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env/train.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP \
    TRAIN_FILES=\"$VH_TRAIN,$LVR_FULL,$VR1_VIDEO_20K\" \
    VAL_FILES=\"$LVR,$LVB,$LVBEN,$MVB,$VRB,$TEMP,$VH,$MMRV,$VMME,$VMMMU_MCQ,$VMQA,$MLVU_MCQ\" \
    bash examples/framethinker/train_grpo_baselines.sh \
    data.media_reading_kwargs.num_frames=16 \
    trainer.test_freq=100
    """

EXP=grpo_baseline_vh+lvrfull+vr1_20k_n32 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env/train.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP \
    TRAIN_FILES=\"$VH_TRAIN,$LVR_FULL,$VR1_VIDEO_20K\" \
    VAL_FILES=\"$LVR,$LVB,$LVBEN,$MVB,$VRB,$TEMP,$VH,$MMRV,$VMME,$VMMMU_MCQ,$VMQA,$MLVU_MCQ\" \
    bash examples/framethinker/train_grpo_baselines.sh \
    data.max_prompt_length=16384 \
    data.max_response_length=16384 \
    data.media_reading_kwargs.num_frames=32 \
    trainer.test_freq=100 \
    trainer.nnodes=8
    """

EXP=framethinker_vh+lvrfull+vr1_20k_n8 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env/train.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP \
    TRAIN_FILES=\"$VH_TRAIN,$LVR_FULL,$VR1_VIDEO_20K\" \
    VAL_FILES=\"$LVR,$LVB,$LVBEN,$MVB,$VRB,$TEMP,$VH,$MMRV,$VMME,$VMMMU_MCQ,$VMQA,$MLVU_MCQ\" \
    bash examples/framethinker/train_frame_thinker.sh \
    data.media_reading_kwargs.num_frames=16 \
    trainer.test_freq=100
    """


EXP=framethinker_vh+lvrfull+vr1_20k_n16 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env/train.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP \
    TRAIN_FILES=\"$VH_TRAIN,$LVR_FULL,$VR1_VIDEO_20K\" \
    VAL_FILES=\"$LVR,$LVB,$LVBEN,$MVB,$VRB,$TEMP,$VH,$MMRV,$VMME,$VMMMU_MCQ,$VMQA,$MLVU_MCQ\" \
    bash examples/framethinker/train_frame_thinker.sh \
    data.media_reading_kwargs.num_frames=16 \
    trainer.test_freq=100 \
    trainer.nnodes=8
    """

EXP=framethinker_vh+lvrfull+vr1_20k_n32 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env/train.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP \
    TRAIN_FILES=\"$VH_TRAIN,$LVR_FULL,$VR1_VIDEO_20K\" \
    VAL_FILES=\"$LVR,$LVB,$LVBEN,$MVB,$VRB,$TEMP,$VH,$MMRV,$VMME,$VMMMU_MCQ,$VMQA,$MLVU_MCQ\" \
    bash examples/framethinker/train_frame_thinker.sh \
    data.max_prompt_length=16384 \
    data.max_response_length=16384 \
    data.media_reading_kwargs.num_frames=32 \
    trainer.test_freq=100 \
    trainer.nnodes=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    """

