#!/bin/bash

# Single node setup script for cluster initialization
# Usage: setup_node.sh --workflow-id WORKFLOW_ID --nnodes NNODES --node-rank NODE_RANK --master-addr MASTER_ADDR

set -e

# Default values
WORKFLOW_ID=""
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT="29501"

green_print() {
    echo -e "\033[32m$1\033[0m"
}

red_print() {
    echo -e "\033[31m$1\033[0m"
}

write_workflow_env() {
    local target_file="$1"
    local nnodes_value="$2"
    cat > "$target_file" << EOF
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NNODES=$nnodes_value
export NODE_RANK=$NODE_RANK
export WORKFLOW_ID=$WORKFLOW_ID
export WORKFLOW_NAME=$WORKFLOW_ID
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --workflow-id)
        WORKFLOW_ID="$2"
        shift 2
        ;;
    --nnodes)
        NNODES="$2"
        shift 2
        ;;
    --node-rank)
        NODE_RANK="$2"
        shift 2
        ;;
    --master-addr)
        MASTER_ADDR="$2"
        shift 2
        ;;
    --master-port)
        MASTER_PORT="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# if user not exist, add user
if ! id -u "jingwang" >/dev/null 2>&1; then
    USER_GID=1021
    USER_UID=1021
    groupadd -g "$USER_GID" "jingwang" || true
    adduser --home "$HOME" --uid "$USER_UID" --gid "$USER_GID" --disabled-password --gecos "" "jingwang" || true
fi

# Validate required parameters
if [[ -z "$WORKFLOW_ID" || -z "$NNODES" || -z "$NODE_RANK" || -z "$MASTER_ADDR" ]]; then
    red_print "Error: Missing required parameters"
    echo "Usage: $0 --workflow-id WORKFLOW_ID --nnodes NNODES --node-rank NODE_RANK --master-addr MASTER_ADDR"
    exit 1
fi

green_print "Setting up node with:"
green_print "  WORKFLOW_ID: $WORKFLOW_ID"
green_print "  NNODES: $NNODES"
green_print "  NODE_RANK: $NODE_RANK"
green_print "  MASTER_ADDR: $MASTER_ADDR"
green_print "  MASTER_PORT: $MASTER_PORT"

# Set HOME directory based on available mount points
if [ -d "/mnt/aws-lfs-01/" ]; then
    export HOME=/mnt/aws-lfs-01/shared/jingwang
elif [ -d "/mnt/amlfs-04/" ]; then
    export HOME=/mnt/amlfs-04/home/jingwang
elif [ -d "/mnt/lfs/" ]; then
    export HOME=/mnt/lfs/home/jingwang
elif [ -d "/mnt/amlfs-03/" ]; then
    export HOME=/mnt/amlfs-03/shared/jingwang
else
    red_print "Error: No suitable home directory found"
    exit 1
fi

green_print "Using HOME: $HOME"

WORKFLOW_DIR="$HOME/.osmo/$WORKFLOW_ID"
ACTUAL_NNODES_FILE="$WORKFLOW_DIR/nnodes_actual.txt"

# 1. Install necessary packages
green_print "Installing necessary packages..."
ps aux | grep -v grep | grep "apt" | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1 || true
apt update
apt install -y openssh-server tmux iproute2 dnsutils pssh rsync fd-find telnet
export DEBIAN_FRONTEND=noninteractive
apt install -y tzdata
rm -rf /var/lib/apt/lists/*

# 2. Configure SSH (exactly from default.yaml)
green_print "Configuring SSH..."
mkdir -p /var/run/sshd
SSHD_CONFIG="/etc/ssh/sshd_config"

# Configure SSH to allow passwordless root login
echo "Configuring SSH for passwordless root login..."
sed -i '/^#\?PermitRootLogin/s/.*/PermitRootLogin yes/' "$SSHD_CONFIG"
sed -i '/^#\?PasswordAuthentication/s/.*/PasswordAuthentication yes/' "$SSHD_CONFIG"
sed -i '/^#\?PermitEmptyPasswords/s/.*/PermitEmptyPasswords yes/' "$SSHD_CONFIG"
sed -i '/^#\?PubkeyAuthentication/s/.*/PubkeyAuthentication yes/' "$SSHD_CONFIG"
sed -i '/^#\?PubkeyAcceptedKeyTypes/s/.*/PubkeyAcceptedKeyTypes ssh-ed25519,ssh-ed25519-cert-v01@openssh.com/' "$SSHD_CONFIG"

# Set empty password for root and jingwang
passwd -d root
passwd -d jingwang || true  # jingwang user might not exist yet

# Start sshd in tmux session (check if already exists)
if tmux has-session -t "ssh" 2>/dev/null; then
    green_print "SSH tmux session already exists, skipping SSH daemon start"
else
    tmux new-session -d -s "ssh" "/usr/sbin/sshd -D"
    green_print "SSH daemon started in tmux session 'ssh'"
fi

# 3. Create /etc/workflow_env
green_print "Creating /etc/workflow_env..."
write_workflow_env /etc/workflow_env "$NNODES"

green_print "/etc/workflow_env created successfully"

# 4. Create workflow directory and collect IP information
mkdir -p "$WORKFLOW_DIR/hosts"

# Write this node's IP to the hosts directory (wait until available)
HOST_IP=""
while [ -z "$HOST_IP" ]; do
    HOST_IP=$(hostname -I | awk '{print $1}')
    if [ -z "$HOST_IP" ]; then
        echo "hostname -I returned empty result, retrying..."
        sleep 1
    fi
done
echo "$HOST_IP" > "$WORKFLOW_DIR/hosts/host_${NODE_RANK}.txt"
green_print "Node IP ($HOST_IP) written to host_${NODE_RANK}.txt"

# 5. Wait for all hosts and create osmo_hosts.txt (master node only)
if [ "$NODE_RANK" -eq 0 ]; then
    green_print "Master node: waiting for all host files..."
    rm -f "$ACTUAL_NNODES_FILE"
    # Wait until all host_*.txt are created
    TIMEOUT=120
    WAIT_START=$(date +%s)
    while [ $(ls "$WORKFLOW_DIR/hosts/host_"*.txt 2>/dev/null | wc -l) -lt "$NNODES" ]; do
        current_count=$(ls "$WORKFLOW_DIR/hosts/host_"*.txt 2>/dev/null | wc -l)
        missing_ranks=""
        for i in $(seq 0 $((NNODES-1))); do
            if [ ! -f "$WORKFLOW_DIR/hosts/host_${i}.txt" ]; then
                if [ -z "$missing_ranks" ]; then
                    missing_ranks="$i"
                else
                    missing_ranks="$missing_ranks,$i"
                fi
            fi
        done
        echo "Waiting for all host_*.txt to be created ($current_count/$NNODES). Missing node_ranks: [$missing_ranks]"
        ELAPSED=$(( $(date +%s) - WAIT_START ))
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo "Warning: Timeout reached after ${TIMEOUT} seconds, continuing anyway..."
            echo "Missing host files:"
            for i in $(seq 0 $((NNODES - 1))); do
                if [ ! -f "$WORKFLOW_DIR/hosts/host_${i}.txt" ]; then
                    echo "  host_${i}.txt (NODE_RANK: $i)"
                fi
            done
            break
        fi
        sleep 2
    done
    green_print "Creating osmo_hosts.txt with available hosts..."
    
    # Create hosts files with available hosts only
    touch "$WORKFLOW_DIR/hosts_tmp.txt"
    touch "$WORKFLOW_DIR/hosts_tmp2.txt"
    AVAILABLE_RANKS=()
    
    for i in $(seq 0 $((NNODES-1))); do
        if [ -f "$WORKFLOW_DIR/hosts/host_${i}.txt" ]; then
            # Strip redundant whitespace and newlines
            CUR_IP=$(cat "$WORKFLOW_DIR/hosts/host_${i}.txt" | tr -d '\n' | tr -d ' ')
            echo "$CUR_IP" >> "$WORKFLOW_DIR/hosts_tmp.txt"
            echo "${CUR_IP} slots=8" >> "$WORKFLOW_DIR/hosts_tmp2.txt"
            AVAILABLE_RANKS+=("$i")
        fi
    done
    
    mv "$WORKFLOW_DIR/hosts_tmp.txt" "$WORKFLOW_DIR/hosts.txt"
    mv "$WORKFLOW_DIR/hosts_tmp2.txt" "$WORKFLOW_DIR/hosts_mpi.txt"
    
    # Copy to /etc/
    cp "$WORKFLOW_DIR/hosts.txt" /etc/osmo_hosts.txt
    cp "$WORKFLOW_DIR/hosts_mpi.txt" /etc/osmo_hosts_mpi.txt
    
    green_print "Created /etc/osmo_hosts.txt and /etc/osmo_hosts_mpi.txt"
    green_print "Host list:"
    cat /etc/osmo_hosts.txt

    ACTUAL_NNODES=${#AVAILABLE_RANKS[@]}
    if [ "$ACTUAL_NNODES" -eq 0 ]; then
        ACTUAL_NNODES=1
    fi
    echo "$ACTUAL_NNODES" > "$ACTUAL_NNODES_FILE"
    write_workflow_env /etc/workflow_env "$ACTUAL_NNODES"
    green_print "Updated /etc/workflow_env with actual NNODES=$ACTUAL_NNODES"
else
    green_print "Worker node: waiting for master to create hosts.txt..."
    
    # Wait for hosts.txt to be created by master
    while [ ! -f "$WORKFLOW_DIR/hosts.txt" ]; do
        echo "Waiting for hosts.txt to be created by master..."
        sleep 2
    done
    
    while [ ! -f "$WORKFLOW_DIR/hosts_mpi.txt" ]; do
        echo "Waiting for hosts_mpi.txt to be created by master..."
        sleep 2
    done
    
    # Copy from workflow dir to /etc/
    cp "$WORKFLOW_DIR/hosts.txt" /etc/osmo_hosts.txt
    cp "$WORKFLOW_DIR/hosts_mpi.txt" /etc/osmo_hosts_mpi.txt
    
    green_print "Copied osmo_hosts.txt from master"

    while [ ! -f "$ACTUAL_NNODES_FILE" ]; do
        echo "Waiting for nnodes_actual.txt from master..."
        sleep 2
    done
    ACTUAL_NNODES=$(cat "$ACTUAL_NNODES_FILE")
    write_workflow_env /etc/workflow_env "$ACTUAL_NNODES"
    green_print "Updated /etc/workflow_env with actual NNODES=$ACTUAL_NNODES"
fi

green_print "Node setup completed successfully!"
green_print "You can now source /etc/workflow_env to use the environment variables."
