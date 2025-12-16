set -ex
TIMESTAMP=$(date +%s%N)
RAY_ADDRESS="http://localhost:8300" ray job submit \
    --submission-id "rm_vllm_${TIMESTAMP}" \
    --runtime-env ../runtime_env/train.yaml \
    -- python examples/deploy_rm/deploy_vllm_ray.py \
    --model_path "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tensor_parallel_size 4 \
    --num_replicas 4
