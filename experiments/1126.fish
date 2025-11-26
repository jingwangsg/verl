EXP=framethinker_coldstart \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    "EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh"

EXP=framethinker_coldstart_TIS_thr2 \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0"""

EXP=framethinker_coldstart_TIS_thr2_isbn \
ray_job_submit --no-wait --skip-exists --runtime-env ../runtime_env.yaml \
    --submission-id $EXP -- bash -c \
    """EXP=$EXP bash examples/framethinker/train_frame_thinker_vhonly.sh \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=true"""

