#!/bin/bash
set -ex

FILE_SYSTEM=$(df -h | grep /mnt/ | awk '{print $6}' | grep -v /home/ | sort | head -n 1)

if [[ "$FILE_SYSTEM" == *"/aws-lfs-01/"* ]]; then
    export HOME=/mnt/aws-lfs-01/shared/jingwang/
elif [[ "$FILE_SYSTEM" == *"/amlfs-04/"* ]]; then
    export HOME=/mnt/amlfs-04/home/jingwang/
elif [[ "$FILE_SYSTEM" == *"/lfs/"* ]]; then
    export HOME=/mnt/lfs/home/jingwang/
elif [[ "$FILE_SYSTEM" == *"/amlfs-01/"* ]]; then
    export HOME=/mnt/amlfs-03/shared/jingwang/
else
    echo "HOME directory not found"
fi

echo $HOME
BASH_ONLY=1 source $HOME/.bashrc

# --- unset proxy ---
export HTTP_PROXY=
export HTTPS_PROXY=
export http_proxy=
export https_proxy=
export ALL_PROXY=
export all_proxy=

unset PIP_CONSTRAINT

# --- NCCL ---
# FIXME: should turn on later
export NCCL_NVLS_ENABLE=0

# To prevent NFS deadlock on flashinfer-cubin
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_extensions_$(hostname)


if [ -n "$VENV" ]; then
    echo "Using virtual environment: $VENV"
    source $VENV/bin/activate
    UV_SYSTEM=""
else
    echo "Using system environment"
    export PATH=/usr/bin:/usr/local/bin:$HOME/homebrew/bin:$PATH
    UV_SYSTEM="--system --break-system-packages"
fi

RAY_VERSION=${RAY_VERSION:-2.40.0}
export WANDB_MODE=online

# Cluster configuration - passed from raystart.sh
RAY_CLUSTER_ID=${RAY_CLUSTER_ID:-0}
RAY_HEAD_IP=${RAY_HEAD_IP}
RAY_ROLE=${RAY_ROLE}
RAY_PORT=${RAY_PORT:-29512}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

echo "=== Ray Cluster Configuration ==="
echo "CLUSTER_ID: $RAY_CLUSTER_ID"
echo "HEAD_IP: $RAY_HEAD_IP"
echo "ROLE: $RAY_ROLE"
echo "PORT: $RAY_PORT"
echo "NODE_RANK: $NODE_RANK"

# Verify role by checking if current node's IP matches head IP
CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "Current node IP: $CURRENT_IP"

# Double-check role assignment
if [ "$RAY_HEAD_IP" = "$CURRENT_IP" ]; then
    if [ "$RAY_ROLE" != "head" ]; then
        echo "WARNING: IP matches head but role is $RAY_ROLE, correcting to head"
        RAY_ROLE="head"
    fi
    IS_HEAD=true
else
    if [ "$RAY_ROLE" = "head" ]; then
        echo "WARNING: Role is head but IP doesn't match ($CURRENT_IP != $RAY_HEAD_IP), correcting to worker"
        RAY_ROLE="worker" 
    fi
    IS_HEAD=false
fi

echo "Final role determination: $RAY_ROLE (IS_HEAD=$IS_HEAD)"

# Setup Ray temporary directory and init flag

set -x

pip_install() {
    pip install uv
    which uv
    uv pip install $UV_SYSTEM boto3 --upgrade
    uv pip install $UV_SYSTEM smart_open --upgrade
    uv pip install $UV_SYSTEM -U ray==$RAY_VERSION
    uv pip install $UV_SYSTEM 'click<8.3'
    uv pip install $UV_SYSTEM ray"[train,default,tune]"
}

if [ -d /mnt/amlfs-02/shared/ckpts/ ]; then
    export HF_HUB_CACHE=/mnt/amlfs-02/shared/ckpts/
elif [ -d /mnt/aws-lfs-01/shared/ckpts/ ]; then
    export HF_HUB_CACHE=/mnt/aws-lfs-01/shared/ckpts/
fi

export AWS_ACCESS_KEY_ID=$(jq -r '.default.aws_access_key_id' ~/.gear/data_credentials)
export AWS_SECRET_ACCESS_KEY=$(jq -r '.default.aws_secret_access_key' ~/.gear/data_credentials)
export AWS_DEFAULT_REGION=$(jq -r '.default.region' ~/.gear/data_credentials)
export AWS_ENDPOINT_URL=$(jq -r '.default.endpoint_url' ~/.gear/data_credentials)
export no_proxy=127.0.0.1,localhost

export RAY_gcs_server_request_timeout_seconds=1800
cd /workspace/

RAY_DEBUG=${RAY_DEBUG:-1}
RAY_NO_HEAD_WORKER=${RAY_NO_HEAD_WORKER:-0}
if [ "$RAY_DEBUG" = "legacy" ]; then
    opt="--ray-debugger-external"
else
    opt=""
fi


# ============================================================================ #
# Ray Variables
export RAY_memory_usage_threshold=0.90
export RAY_num_heartbeats_timeout=600

# ============================================================================ #
# Print Important Environment Variables
echo "PYTHONPATH: $PYTHONPATH"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LD_PRELOAD: $LD_PRELOAD"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"


RAY_TMP_DIR=$HOME/.tmp/.ray/$WORKFLOW_ID

# Clean up GPU memory
# bash $HOME/WORKSPACE/GearLaunchV2/scripts/clean_gpu.sh

# filter if SKIP_FILTER is not 1 and healthy_gpu_ids.txt exists
if [ "$SKIP_FILTER" != "1" ] && [ -f /etc/healthy_gpu_ids.txt ]; then
    export CUDA_VISIBLE_DEVICES=$(cat /etc/healthy_gpu_ids.txt)
fi

# Execute based on determined role
if [ "$IS_HEAD" = "true" ]; then
    echo "Starting as HEAD node for cluster $RAY_CLUSTER_ID"
    
    pip_install
    which ray
    ray stop --force

    if [ "$RAY_NO_HEAD_WORKER" = "1" ]; then
        ARGS_CPU="--num-cpus 0 --num-gpus 0"
    else
        ARGS_CPU=""
    fi
    
    ray start --head --port=${RAY_PORT} --include-dashboard=true --dashboard-port=$RAY_DASHBOARD_PORT $opt $ARGS_CPU --resources='{"drivers": 1}'
    MASTER_INIT_FLAG=$RAY_TMP_DIR/init_flag_master
    touch $MASTER_INIT_FLAG
    
    sleep 1
    echo "Head node setup complete for cluster $RAY_CLUSTER_ID"

    # Setup monitoring
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        ARCH="arm64"
    else
        ARCH="amd64"
    fi

    rm -rf /tmp/ray/*.tar.gz
    wget https://github.com/prometheus/prometheus/releases/download/v2.47.0/prometheus-2.47.0.linux-$ARCH.tar.gz -O /tmp/ray/prometheus-2.47.0.linux-$ARCH.tar.gz
    tar -xzf /tmp/ray/prometheus-2.47.0.linux-$ARCH.tar.gz -C /tmp/ray/

    wget https://dl.grafana.com/oss/release/grafana-10.2.2.linux-$ARCH.tar.gz -O /tmp/ray/grafana-v10.2.2.linux-$ARCH.tar.gz
    tar -xzf /tmp/ray/grafana-v10.2.2.linux-$ARCH.tar.gz -C /tmp/ray/

    RAY_MONITOR_DIR="/tmp/ray/"
    # cancel if exist
    if tmux list-sessions | grep -q "grafana"; then
        tmux kill-session -t grafana > /dev/null 2>&1
    fi
    if tmux list-sessions | grep -q "prometheus"; then
        tmux kill-session -t prometheus > /dev/null 2>&1
    fi
    ps aux | grep -v grep | grep "grafana" | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1 || true
    ps aux | grep -v grep | grep "prometheus" | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1 || true

    tmux new-session -d -s grafana "$RAY_MONITOR_DIR/grafana-v10.2.2/bin/grafana server --config=/tmp/ray/session_latest/metrics/grafana/grafana.ini --homepath=$RAY_MONITOR_DIR/grafana-v10.2.2/"
    tmux new-session -d -s prometheus "$RAY_MONITOR_DIR/prometheus-2.47.0.linux-$ARCH/prometheus --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml --web.listen-address=0.0.0.0:9090"

    # if all init flags are created, exit
    # Only source NNODES from /etc/workflow_env
    if [ -f /etc/workflow_env ]; then
        NNODES=$(grep "^export NNODES=" /etc/workflow_env | cut -d'=' -f2)
    else
        echo "Warning: /etc/workflow_env not found, defaulting NNODES to 1"
        NNODES=1
    fi
    set +ex
    TIMEOUT=30
    PRINT_INTERVAL=5
    WAIT_START=$(date +%s)
    LAST_PRINT_TIME=$WAIT_START

    while [ $(ls $RAY_TMP_DIR/init_flag_* 2>/dev/null | wc -l) -lt "$NNODES" ]; do
        sleep 1
        NOW=$(date +%s)
        ELAPSED=$((NOW - WAIT_START))

        # Print progress every PRINT_INTERVAL seconds
        if [ $((NOW - LAST_PRINT_TIME)) -ge $PRINT_INTERVAL ]; then
            CONNECTED=$(ls $RAY_TMP_DIR/init_flag_* 2>/dev/null | wc -l)
            MISSING=$((NNODES - CONNECTED))
            echo "Waiting for workers... ($CONNECTED/$NNODES connected, $MISSING missing, elapsed: ${ELAPSED}s)"

            # Show what we have connected
            echo "Connected nodes:"
            if [ -f "$RAY_TMP_DIR/init_flag_master" ]; then
                echo "  ✓ HEAD (IP: $RAY_HEAD_IP)"
            fi
            for flag_file in $RAY_TMP_DIR/init_flag_worker_*; do
                [ -e "$flag_file" ] || continue
                FLAG_NAME=$(basename "$flag_file")
                if [[ "$FLAG_NAME" =~ init_flag_worker_([0-9]+) ]]; then
                    NODE_RANK=${BASH_REMATCH[1]}
                    if [ -f /etc/osmo_hosts.txt ]; then
                        TOTAL_HOSTS=$(wc -l < /etc/osmo_hosts.txt)
                        if [ "$NODE_RANK" -lt "$TOTAL_HOSTS" ]; then
                            WORKER_IP=$(sed -n "$((NODE_RANK + 1))p" /etc/osmo_hosts.txt)
                            echo "  ✓ WORKER NODE_RANK=$NODE_RANK (IP: $WORKER_IP)"
                        fi
                    fi
                fi
            done
            LAST_PRINT_TIME=$NOW
        fi

        if [ $ELAPSED -ge $TIMEOUT ]; then
            CONNECTED=$(ls $RAY_TMP_DIR/init_flag_* 2>/dev/null | wc -l)
            MISSING=$((NNODES - CONNECTED))
            echo "Warning: Timeout reached after ${TIMEOUT} seconds, continuing anyway... ($CONNECTED/$NNODES connected, $MISSING missing)"

            # Show final status
            echo "Final connection status:"
            if [ -f "$RAY_TMP_DIR/init_flag_master" ]; then
                echo "  ✓ HEAD (IP: $RAY_HEAD_IP)"
            else
                echo "  ✗ HEAD (IP: $RAY_HEAD_IP) - MISSING"
            fi
            for flag_file in $RAY_TMP_DIR/init_flag_worker_*; do
                [ -e "$flag_file" ] || continue
                FLAG_NAME=$(basename "$flag_file")
                if [[ "$FLAG_NAME" =~ init_flag_worker_([0-9]+) ]]; then
                    NODE_RANK=${BASH_REMATCH[1]}
                    if [ -f /etc/osmo_hosts.txt ]; then
                        TOTAL_HOSTS=$(wc -l < /etc/osmo_hosts.txt)
                        if [ "$NODE_RANK" -lt "$TOTAL_HOSTS" ]; then
                            WORKER_IP=$(sed -n "$((NODE_RANK + 1))p" /etc/osmo_hosts.txt)
                            echo "  ✓ WORKER NODE_RANK=$NODE_RANK (IP: $WORKER_IP)"
                        fi
                    fi
                fi
            done
            break
        fi
    done
    set -ex

    if [ "$RAY_DAEMON" = "1" ]; then
        cd $HOME/WORKSPACE/GearLaunchV2/scripts/
        RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --working-dir . --submission-id "ray_daemon" --runtime-env-json '{"env_vars": {"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING":"1", "RAY_DEDUP_LOGS": "0"}}' -- python ray_daemon.py
    fi
else
    echo "Starting as WORKER node for cluster $RAY_CLUSTER_ID"
    
    if [ -z "$VENV" ]; then
        pip_install
    fi
    
    # Wait for head node to be ready
    echo "Waiting for head node at $RAY_HEAD_IP to be ready..."
    while [ ! -f $RAY_INIT_FLAG ]; do
        sleep 1
    done
    
    sleep 5
    which ray
    ray stop --force
    
    echo "Connecting to head at $RAY_HEAD_IP:$RAY_PORT"
    ray start --address=$RAY_HEAD_IP:$RAY_PORT $opt --resources='{"drivers": 1}'

    WORKER_INIT_FLAG=$RAY_TMP_DIR/init_flag_worker_${NODE_RANK}
    touch $WORKER_INIT_FLAG
fi

sleep infinity
