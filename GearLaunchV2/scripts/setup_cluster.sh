#!/bin/bash

# Cluster setup script - distributes node setup to all nodes in workflow
# Usage: setup_cluster.sh --workflow-id WORKFLOW_ID [--master-port PORT]

# Based on obash.sh but modified for cluster setup distribution

set -e

# Default values
WORKFLOW_ID=""
MASTER_PORT="29501"
LOGDIR=""
SESSION="cluster_setup"

green_print() {
    echo -e "\033[32m$1\033[0m"
}

red_print() {
    echo -e "\033[31m$1\033[0m"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --workflow-id)
        WORKFLOW_ID="$2"
        shift 2
        ;;
    --master-port)
        MASTER_PORT="$2"
        shift 2
        ;;
    --logdir)
        LOGDIR="$2"
        shift 2
        ;;
    --session)
        SESSION="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --workflow-id WORKFLOW_ID [--master-port PORT] [--logdir DIR] [--session NAME]"
        exit 1
        ;;
    esac
done

# Validate required parameters
if [[ -z "$WORKFLOW_ID" ]]; then
    red_print "Error: --workflow-id is required"
    echo "Usage: $0 --workflow-id WORKFLOW_ID [--master-port PORT]"
    exit 1
fi

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

# Get all nvidia tasks from osmo (must run as jingwang)
green_print "Getting all nvidia tasks from osmo..."
mkdir -p /tmp/osmo-client-logs
chmod -R 777 /tmp/osmo-client-logs
OSMO_TASKS=$(su - jingwang -c "OSMO_LOG_FILE_DIR=/tmp/osmo-client-logs /osmo/usr/bin/osmo task list -a -c 1000 -w $WORKFLOW_ID | grep nvidia | grep RUNNING")

if [[ -z "$OSMO_TASKS" ]]; then
    red_print "Error: Could not find any nvidia tasks for workflow $WORKFLOW_ID"
    exit 1
fi

# Parse task names and create sorted list (master first, then workers by number)
TASK_NAMES=
while IFS= read -r line; do
    TASK_NAMES+=($(echo $line | awk '{print $3}'))
done <<< "$OSMO_TASKS"


# Sort tasks: master first, then workers by numerical order
SORTED_TASKS=()
MASTER_TASK=""
WORKER_TASKS=()

for task in "${TASK_NAMES[@]}"; do
    if [[ "$task" == "master" ]]; then
        MASTER_TASK="$task"
    elif [[ "$task" =~ ^worker_([0-9]+)$ ]]; then
        WORKER_TASKS+=("$task")
    fi
done

# Sort worker tasks numerically
IFS=$'
' WORKER_TASKS=($(sort -V <<< "${WORKER_TASKS[*]}"))
unset IFS

# Build final sorted task list
if [[ -n "$MASTER_TASK" ]]; then
    SORTED_TASKS=("$MASTER_TASK")
fi
SORTED_TASKS+=("${WORKER_TASKS[@]}")

NNODES=${#SORTED_TASKS[@]}

green_print "Found tasks in order:"
for i in "${!SORTED_TASKS[@]}"; do
    green_print "  NODE_RANK=$i -> ${SORTED_TASKS[$i]}"
done

if [[ -z "$NNODES" || "$NNODES" -eq 0 ]]; then
    red_print "Error: Could not determine NNODES or workflow not found"
    red_print "Make sure workflow $WORKFLOW_ID exists and has nvidia tasks"
    exit 1
fi

# Get master address (current node)
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Set current node rank to 0 (master)
NODE_RANK=0

WORKFLOW_DIR="$HOME/.osmo/$WORKFLOW_ID"
LOGDIR="$WORKFLOW_DIR/cluster_setup/"

rm -rfv $WORKFLOW_DIR/hosts/
rm -rfv $WORKFLOW_DIR/hosts.txt
rm -rfv $WORKFLOW_DIR/hosts_mpi.txt

green_print "Cluster setup configuration:"
green_print "  WORKFLOW_ID: $WORKFLOW_ID"
green_print "  NNODES: $NNODES"
green_print "  MASTER_ADDR: $MASTER_ADDR"
green_print "  MASTER_PORT: $MASTER_PORT"
green_print "  LOGDIR: $LOGDIR"

# Create log directory
mkdir -p "$LOGDIR"
chmod -R 777 "$LOGDIR"

ENTRY_SESSION="entry_${SESSION}"

# Kill existing tmux session if it exists
if tmux has-session -t "$ENTRY_SESSION" 2>/dev/null; then
    echo "Session '$ENTRY_SESSION' already exists. Killing existing session..."
    tmux kill-session -t "$ENTRY_SESSION"
fi

# Get current working directory and setup_node.sh path
PWD_DIR=$HOME/WORKSPACE/GearLaunchV2
SETUP_NODE_SCRIPT="$PWD_DIR/scripts/setup_node.sh"

if [[ ! -f "$SETUP_NODE_SCRIPT" ]]; then
    red_print "Error: setup_node.sh not found at $SETUP_NODE_SCRIPT"
    exit 1
fi

# Create temporary directory and script
TMP_DIR="$HOME/.tmp"
mkdir -p "$TMP_DIR"
TMP_BASH="$TMP_DIR/${WORKFLOW_ID}_cluster_setup.sh"

# Always generate fresh setup script with latest setup_node.sh version
# Remove existing temp script if it exists to force regeneration
rm -f "$TMP_BASH"

# Generate a simple wrapper script that directly calls the latest setup_node.sh
cat > "$TMP_BASH" << EOF
#!/bin/bash
# Auto-generated cluster setup wrapper script
# This script always uses the latest setup_node.sh from GearLaunchV2

set -e

# Set environment based on available mount points
if [ -d /mnt/aws-lfs-01/ ]; then
    export PATH=/usr/bin:/usr/local/bin:/mnt/aws-lfs-01/shared/jingwang/homebrew/bin:\$PATH
    export HOME=/mnt/aws-lfs-01/shared/jingwang/
elif [ -d /mnt/amlfs-04/ ]; then
    export PATH=/usr/bin:/usr/local/bin:/mnt/amlfs-04/home/jingwang/homebrew/bin:\$PATH
    export HOME=/mnt/amlfs-04/home/jingwang/
elif [ -d /mnt/lfs/ ]; then
    export PATH=/usr/bin:/usr/local/bin:/mnt/lfs/home/jingwang/homebrew/bin:\$PATH
    export HOME=/mnt/lfs/home/jingwang/
elif [ -d /mnt/amlfs-03/ ]; then
    export PATH=/usr/bin:/usr/local/bin:/mnt/amlfs-03/shared/jingwang/homebrew/bin:\$PATH
    export HOME=/mnt/amlfs-03/shared/jingwang/
fi

# if user not exist, add user
if ! id -u "jingwang" >/dev/null 2>&1; then
    USER_GID=1021
    USER_UID=1021
    groupadd -g "$USER_GID" "jingwang" || true
    adduser --home "$HOME" --uid "$USER_UID" --gid "$USER_GID" --disabled-password --gecos "" "jingwang" || true
fi

export TERM=xterm-256color
export WANDB_MODE=disabled

if [ -f \$HOME/.bashrc ]; then
    source \$HOME/.bashrc
fi

# Execute the setup with proper parameters
exec > >(tee -a "$LOGDIR/node_\${NODE_RANK}.log") 2>&1

echo "Starting cluster setup on node \${NODE_RANK}..."

# Always use the latest setup_node.sh from GearLaunchV2 shared location
SETUP_SCRIPT="$PWD_DIR/scripts/setup_node.sh"

# Verify setup_node.sh exists and is accessible
if [ ! -f "\$SETUP_SCRIPT" ]; then
    echo "Error: setup_node.sh not found at \$SETUP_SCRIPT"
    echo "Make sure GearLaunchV2 is accessible from all nodes"
    exit 1
fi

echo "Using setup script: \$SETUP_SCRIPT"

# Execute the latest setup_node.sh directly
bash "\$SETUP_SCRIPT" \\
    --workflow-id "$WORKFLOW_ID" \\
    --nnodes "$NNODES" \\
    --node-rank "\${NODE_RANK}" \\
    --master-addr "$MASTER_ADDR" \\
    --master-port "$MASTER_PORT"

echo "Cluster setup completed on node \${NODE_RANK}"
sleep infinity
EOF

# Set permissions
chmod 777 "$TMP_BASH"

green_print "Generated setup script: $TMP_BASH (always uses latest setup_node.sh)"


# Create new tmux session
tmux new-session -d -s "$ENTRY_SESSION"
green_print "Created tmux session: $ENTRY_SESSION"

# Get osmo executable path
OSMO_EXECUTABLE="/osmo/usr/bin/osmo"

green_print "Using osmo executable: $OSMO_EXECUTABLE"

# Execute setup on all nodes
green_print "Distributing cluster setup to $NNODES nodes..."

# Process nodes in batches  f 8
for ((k = 0; k< 10; k++)); do
for ((i = 0; i < NNODES; i += 8)); do

    green_print "Processing batch starting at node $i..."
    ALL_SKIP=1
    
    for ((j = 0; j < 8; j++)); do
        CUR_NODE_RANK=$((i + j))
        ALL_SKIP=0

        # if tmux window is alive, skip
        if tmux list-windows -t "$ENTRY_SESSION" -F '#{window_name}' | grep -q "^node${CUR_NODE_RANK}$"; then
            continue
        fi

        if [[ $CUR_NODE_RANK -ge $NNODES ]]; then
            break
        fi
        
        # Get the actual task name for this NODE_RANK
        if [[ $CUR_NODE_RANK -lt ${#SORTED_TASKS[@]} ]]; then
            TASK_NAME="${SORTED_TASKS[$CUR_NODE_RANK]}"
        else
            red_print "Error: NODE_RANK $CUR_NODE_RANK exceeds available tasks"
            continue
        fi
        
        if [[ $CUR_NODE_RANK -eq $NODE_RANK ]]; then
            # Execute setup directly on current node (master)
            green_print "Setting up master node (rank $CUR_NODE_RANK, task: $TASK_NAME)..."
            tmux new-window -t "$ENTRY_SESSION" -n "node${CUR_NODE_RANK}" \
                "NODE_RANK=$CUR_NODE_RANK WORKFLOW_ID=$WORKFLOW_ID NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT bash $TMP_BASH"
        else
            # Execute setup on remote node using osmo workflow exec
            
            green_print "Setting up remote node $CUR_NODE_RANK (task: $TASK_NAME)..."
            
            # Create a wrapper script for remote execution with environment variables
            REMOTE_WRAPPER="$TMP_DIR/${WORKFLOW_ID}_node_${CUR_NODE_RANK}.sh"
            cat > "$REMOTE_WRAPPER" << EOF
#!/bin/bash
export NODE_RANK=$CUR_NODE_RANK
export WORKFLOW_ID=$WORKFLOW_ID
export NNODES=$NNODES
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export LOGDIR=$LOGDIR
export PWD_DIR=$PWD_DIR

bash $TMP_BASH
EOF
            chmod +x "$REMOTE_WRAPPER"
            
            # Execute on remote node
            tmux new-window -t "$ENTRY_SESSION" -n "node${CUR_NODE_RANK}" \
                "su - jingwang -c 'OSMO_LOG_FILE_DIR=/tmp/osmo-client-logs $OSMO_EXECUTABLE workflow exec $WORKFLOW_ID $TASK_NAME --entry $REMOTE_WRAPPER'"
        fi
    done
    
    # Brief pause between batches
    if [[ $ALL_SKIP -eq 0 ]]; then
        sleep 2
    fi
done
sleep 10
done

green_print ""
green_print "Cluster setup distribution completed!"
green_print ""
green_print "To monitor progress:"
green_print "  tmux attach -t $ENTRY_SESSION"
green_print ""
green_print "Log files are in: $LOGDIR"
green_print ""
green_print "Once setup is complete, all nodes will have:"
green_print "  - /etc/workflow_env with environment variables"
green_print "  - /etc/osmo_hosts.txt with all node IPs"
green_print "  - SSH daemon running for passwordless access"
green_print ""
green_print "You can then use the original obash.sh script for distributed commands."