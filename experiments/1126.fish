set DATA_DIR /mnt/amlfs-02/shared/datasets/s3/video_reason/
set VH_TRAIN $DATA_DIR/Video-Holmes/train.parquet
set VH_TEST $DATA_DIR/Video-Holmes/test.parquet
set LVR_3K $DATA_DIR/LongVideoReason/train_mcq_3k.parquet
set LVR_10K $DATA_DIR/LongVideoReason/train_mcq_10k.parquet
set LVR_TEST $DATA_DIR/LongVideoReason/test.parquet

EXP=framethinker_coldstart_lr3e-6 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    "EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
    actor_rollout_ref.actor.optim.lr=3e-6"

EXP=framethinker_coldstart_TIS_thr2_lr3e-6 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    actor_rollout_ref.actor.optim.lr=3e-6"""

EXP=framethinker_coldstart_TIS_thr2_isbn_lr3e-6 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=true \
    actor_rollout_ref.actor.optim.lr=3e-6"""





# EXP=framethinker_coldstart \
# ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
#     --submission-id $EXP -- bash -c \
#     "EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh"


# EXP=framethinker_coldstart_TIS_thr2 \
# ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
#     --submission-id $EXP -- bash -c \
#     """EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
#     algorithm.rollout_correction.rollout_is=token \
#     algorithm.rollout_correction.rollout_is_threshold=2.0"""

# EXP=framethinker_coldstart_TIS_thr2_isbn \
# ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
#     --submission-id $EXP -- bash -c \
#     """EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
#     algorithm.rollout_correction.rollout_is=token \
#     algorithm.rollout_correction.rollout_is_threshold=2.0 \
#     algorithm.rollout_correction.rollout_is_batch_normalize=true"""






EXP=framethinker_coldstart_vh+lvr3k \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker.sh \
    data.train_files=[$VH_TRAIN,$LVR_3K] \
    data.val_files=[$VH_TEST,$LVR_TEST] \
    """

EXP=framethinker_coldstart_vh+lvr10k \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker.sh \
    data.train_files=[$VH_TRAIN,$LVR_10K] \
    data.val_files=[$VH_TEST,$LVR_TEST] \
    """

EXP=framethinker_coldstart_vh+lvr3k_TIS_thr2 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker.sh \
    data.train_files=[$VH_TRAIN,$LVR_3K] \
    data.val_files=[$VH_TEST,$LVR_TEST] \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0
    """

EXP=framethinker_coldstart_vh+lvr10k_TIS_thr2 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker.sh \
    data.train_files=[$VH_TRAIN,$LVR_10K] \
    data.val_files=[$VH_TEST,$LVR_TEST] \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0
    """




EXP=framethinker_coldstart_vh+lvr3k_TIS_thr2_isbn \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker.sh \
    data.train_files=[$VH_TRAIN,$LVR_3K] \
    data.val_files=[$VH_TEST,$LVR_TEST] \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=true \
    """

EXP=framethinker_coldstart_vh+lvr10k_TIS_thr2_isbn \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker.sh \
    data.train_files=[$VH_TRAIN,$LVR_10K] \
    data.val_files=[$VH_TEST,$LVR_TEST] \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=true \
    """

