#!/bin/bash
set -e

source /etc/workflow_env # use latest workflow env

# Default values for backward compatibility
if [ -z "${RAY_CLUSTER_SIZES+x}" ]; then
    RAY_CLUSTER_SIZES_WAS_PROVIDED=0
else
    RAY_CLUSTER_SIZES_WAS_PROVIDED=1
fi
RAY_HEADS=${RAY_HEADS:-0}
RAY_CLUSTER_SIZES=${RAY_CLUSTER_SIZES:-$NNODES}

echo "RAY_HEADS=$RAY_HEADS"
echo "RAY_CLUSTER_SIZES=$RAY_CLUSTER_SIZES"

# Convert comma-separated strings to arrays
IFS=',' read -ra HEADS_ARRAY <<< "$RAY_HEADS"
IFS=',' read -ra SIZES_ARRAY <<< "$RAY_CLUSTER_SIZES"

# Validate input
if [ ${#HEADS_ARRAY[@]} -ne ${#SIZES_ARRAY[@]} ]; then
    echo "Error: RAY_HEADS and RAY_CLUSTER_SIZES must have the same number of elements"
    exit 1
fi

# Parse EXCLUDE_NODES into array for easier processing
declare -a EXCLUDED_NODES
if [ -n "$EXCLUDE_NODES" ]; then
    IFS=',' read -ra EXCLUDED_NODES <<< "$EXCLUDE_NODES"
    echo "Setting EXCLUDE_NODES=$EXCLUDE_NODES"
fi

# Function to check if a node is excluded
is_excluded() {
    local node_rank=$1
    for excluded in "${EXCLUDED_NODES[@]}"; do
        if [ "$node_rank" = "$excluded" ]; then
            return 0  # is excluded
        fi
    done
    return 1  # not excluded
}

RAY_TMP_DIR=$HOME/.tmp/.ray/$WORKFLOW_ID
rm -rf $RAY_TMP_DIR/init_flag_*
mkdir -p $RAY_TMP_DIR

PSSH_SCRIPT=$(dirname $0)/pssh.sh

WORKFLOW_HOSTS_DIR="$HOME/.osmo/$WORKFLOW_ID/hosts"
declare -a ALL_NODE_RANKS=()
declare -A NODE_IP_MAP

if [ -d "$WORKFLOW_HOSTS_DIR" ]; then
    while IFS= read -r host_file; do
        host_base=$(basename "$host_file")
        host_name="${host_base%.txt}"
        rank="${host_name#host_}"
        if [[ "$rank" =~ ^[0-9]+$ ]]; then
            ip=$(head -n 1 "$host_file" | tr -d '[:space:]')
            if [ -n "$ip" ]; then
                ALL_NODE_RANKS+=("$rank")
                NODE_IP_MAP[$rank]=$ip
            fi
        fi
    done < <(find "$WORKFLOW_HOSTS_DIR" -maxdepth 1 -name 'host_*.txt' | sort -V)
fi

if [ ${#ALL_NODE_RANKS[@]} -eq 0 ]; then
    TOTAL_NODES=$(wc -l < /etc/osmo_hosts.txt)
    for ((node_rank=0; node_rank<TOTAL_NODES; node_rank++)); do
        ALL_NODE_RANKS+=("$node_rank")
        NODE_IP_MAP[$node_rank]=$(sed -n "$((node_rank + 1))p" /etc/osmo_hosts.txt)
    done
else
    echo "Using node ranks from host metadata: ${ALL_NODE_RANKS[*]}"
fi

ACTUAL_NODE_COUNT=${#ALL_NODE_RANKS[@]}
if [ "$ACTUAL_NODE_COUNT" -eq 0 ]; then
    echo "Error: Could not determine any available nodes"
    exit 1
fi
echo "Detected $ACTUAL_NODE_COUNT nodes for workflow $WORKFLOW_ID"

if [ "$RAY_CLUSTER_SIZES_WAS_PROVIDED" -eq 0 ] && [ ${#SIZES_ARRAY[@]} -eq 1 ]; then
    echo "Adjusting cluster size to actual node count: $ACTUAL_NODE_COUNT"
    SIZES_ARRAY[0]=$ACTUAL_NODE_COUNT
fi

# Generate a unified script that all nodes will run
RAY_UNIFIED_SCRIPT="$RAY_TMP_DIR/ray_unified_startup.sh"

cat > "$RAY_UNIFIED_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash
set -e

# Source workflow environment to get correct NODE_RANK
source /etc/workflow_env

# Node configuration determination based on NODE_RANK
echo "Node $NODE_RANK starting Ray configuration..."

SCRIPT_EOF

# First pass: collect all node assignments to ensure no conflicts
declare -A NODE_ASSIGNMENTS  # node_rank -> "cluster_id:role:head_ip:port:dashboard_port"

# Create hash set of head nodes for fast lookup
declare -A HEAD_NODES
for head in "${HEADS_ARRAY[@]}"; do
    HEAD_NODES[$head]=1
done

# Build list of available worker nodes (all nodes - head nodes - excluded nodes)
AVAILABLE_WORKERS=()
for node_rank in "${ALL_NODE_RANKS[@]}"; do
    # Skip excluded nodes
    if is_excluded $node_rank; then
        continue
    fi
    
    # Skip head nodes (fast hash lookup)
    if [ -z "${HEAD_NODES[$node_rank]}" ]; then
        AVAILABLE_WORKERS+=("$node_rank")
    fi
done

echo "Available worker nodes: ${AVAILABLE_WORKERS[*]}"
WORKER_INDEX=0

for i in "${!HEADS_ARRAY[@]}"; do
    HEAD_RANK=${HEADS_ARRAY[$i]}
    CLUSTER_SIZE=${SIZES_ARRAY[$i]}
    
    
    # Check if head node is excluded
    if is_excluded $HEAD_RANK; then
        echo "Warning: Head node $HEAD_RANK for cluster $i is in EXCLUDE_NODES, skipping this cluster"
        continue
    fi
    
    echo "=== Planning Ray cluster $i: head=$HEAD_RANK, target size=$CLUSTER_SIZE ==="
    
    # Get head node IP
    HEAD_IP=${NODE_IP_MAP[$HEAD_RANK]}
    if [ -z "$HEAD_IP" ]; then
        HEAD_IP=$(sed -n "$((HEAD_RANK + 1))p" /etc/osmo_hosts.txt)
    fi
    if [ -z "$HEAD_IP" ]; then
        echo "Error: Cannot get IP for head node $HEAD_RANK"
        continue
    fi
    
    # Use same ports for all clusters (since nodes don't overlap)
    # Default to 6379 which is the traditional Ray port
    CLUSTER_PORT=${RAY_PORT:-6379}
    CLUSTER_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
    
    # Check if head node is already assigned
    if [ -n "${NODE_ASSIGNMENTS[$HEAD_RANK]}" ]; then
        echo "Error: Node $HEAD_RANK is already assigned to another cluster. Cannot assign as head for cluster $i"
        continue
    fi
    
    # Assign head node
    NODE_ASSIGNMENTS[$HEAD_RANK]="$i:head:$HEAD_IP:$CLUSTER_PORT:$CLUSTER_DASHBOARD_PORT"
    unset CLUSTER_NODES
    CLUSTER_NODES=($HEAD_RANK)
    
    # Assign worker nodes from available worker pool
    WORKERS_NEEDED=$((CLUSTER_SIZE - 1))
    for ((w=0; w<WORKERS_NEEDED && WORKER_INDEX<${#AVAILABLE_WORKERS[@]}; w++)); do
        worker_node=${AVAILABLE_WORKERS[$WORKER_INDEX]}
        NODE_ASSIGNMENTS[$worker_node]="$i:worker:$HEAD_IP:$CLUSTER_PORT:$CLUSTER_DASHBOARD_PORT"
        CLUSTER_NODES+=($worker_node)
        echo "  Added worker node $worker_node to cluster $i"
        WORKER_INDEX=$((WORKER_INDEX + 1))
    done
    
    ACTUAL_CLUSTER_SIZE=${#CLUSTER_NODES[@]}
    if [ "$ACTUAL_CLUSTER_SIZE" -lt "$CLUSTER_SIZE" ]; then
        echo "Warning: Only found $ACTUAL_CLUSTER_SIZE nodes for cluster $i (requested $CLUSTER_SIZE) - not enough available workers"
    fi
    echo "  Final cluster $i nodes: ${CLUSTER_NODES[*]}"
    echo "  Head IP: $HEAD_IP, Port: $CLUSTER_PORT, Dashboard: $CLUSTER_DASHBOARD_PORT"
done

# Second pass: generate unified script with non-conflicting assignments
echo "# Cluster assignments" >> "$RAY_UNIFIED_SCRIPT"
echo "source /etc/workflow_env" >> "$RAY_UNIFIED_SCRIPT"
echo "export NNODES=$ACTUAL_NODE_COUNT" >> "$RAY_UNIFIED_SCRIPT"

for node_rank in "${!NODE_ASSIGNMENTS[@]}"; do
    IFS=':' read -ra ASSIGNMENT <<< "${NODE_ASSIGNMENTS[$node_rank]}"
    CLUSTER_ID=${ASSIGNMENT[0]}
    ROLE=${ASSIGNMENT[1]}
    HEAD_IP=${ASSIGNMENT[2]}
    CLUSTER_PORT=${ASSIGNMENT[3]}
    CLUSTER_DASHBOARD_PORT=${ASSIGNMENT[4]}
    
    # Add node-specific condition to unified script
    cat >> "$RAY_UNIFIED_SCRIPT" << NODE_CONDITION_EOF
if [ "\$NODE_RANK" = "$node_rank" ]; then
    echo "Node \$NODE_RANK: Starting Ray cluster $CLUSTER_ID as $ROLE"
    export RAY_CLUSTER_ID=$CLUSTER_ID
    export RAY_HEAD_IP="$HEAD_IP"
    export RAY_ROLE="$ROLE"
    export RAY_PORT=$CLUSTER_PORT
    export RAY_DASHBOARD_PORT=$CLUSTER_DASHBOARD_PORT
    export RAY_DAEMON=$RAY_DAEMON
NODE_CONDITION_EOF

    # Add environment variables conditionally
    if [ -n "$RAY_VERSION" ]; then
        echo "    export RAY_VERSION=\"$RAY_VERSION\"" >> "$RAY_UNIFIED_SCRIPT"
    fi
    if [ -n "$RAY_DEBUG" ]; then
        echo "    export RAY_DEBUG=\"$RAY_DEBUG\"" >> "$RAY_UNIFIED_SCRIPT"
    fi
    if [ -n "$RAY_NO_HEAD_WORKER" ]; then
        echo "    export RAY_NO_HEAD_WORKER=\"$RAY_NO_HEAD_WORKER\"" >> "$RAY_UNIFIED_SCRIPT"
    fi
    if [ -n "$VENV" ]; then
        echo "    export VENV=\"$VENV\"" >> "$RAY_UNIFIED_SCRIPT"
    fi
    if [ -n "SKIP_FILTER" ]; then
        echo "    export SKIP_FILTER=\"$SKIP_FILTER\"" >> "$RAY_UNIFIED_SCRIPT"
    fi
    
    echo "    exec bash $(dirname $0)/_raystart.sh" >> "$RAY_UNIFIED_SCRIPT"
    echo "fi" >> "$RAY_UNIFIED_SCRIPT"
    echo "" >> "$RAY_UNIFIED_SCRIPT"
done

# Add excluded nodes handling
if [ ${#EXCLUDED_NODES[@]} -gt 0 ]; then
    echo "# Excluded nodes handling" >> "$RAY_UNIFIED_SCRIPT"
    for excluded in "${EXCLUDED_NODES[@]}"; do
        cat >> "$RAY_UNIFIED_SCRIPT" << EXCLUDED_CONDITION_EOF
if [ "\$NODE_RANK" = "$excluded" ]; then
    echo "Node \$NODE_RANK: Excluded from Ray clusters, exiting"
    exit 0
fi

EXCLUDED_CONDITION_EOF
    done
fi

# Add fallback for unassigned nodes
cat >> "$RAY_UNIFIED_SCRIPT" << 'FALLBACK_EOF'
# Fallback for unassigned nodes
echo "Node $NODE_RANK: Not assigned to any Ray cluster, exiting"
exit 0
FALLBACK_EOF

# Make the script executable
chmod +x "$RAY_UNIFIED_SCRIPT"

echo "Generated unified Ray startup script: $RAY_UNIFIED_SCRIPT"
echo "Starting all Ray clusters in parallel..."

# Execute the unified script on all nodes in parallel
eval "bash $PSSH_SCRIPT --timeout 30 --session ray --cmd 'bash $RAY_UNIFIED_SCRIPT'"

# Save cluster count for status/stop scripts
echo "${#HEADS_ARRAY[@]}" > "$RAY_TMP_DIR/num_clusters.txt"
echo "Started ${#HEADS_ARRAY[@]} Ray clusters in parallel"

echo ""

# Always kill existing port forward tmux session first
TMUX_SESSION="ray_port_forward"
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing tmux session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
fi

# Set up port forwarding if requested
if [ "$PORT_FORWARD" = "1" ]; then
    echo "=== Setting up Dashboard Port Forwarding ==="

    # Create tmux session for port forwarding
    tmux new-session -d -s "$TMUX_SESSION" -c "$(pwd)"
    sleep 0.5  # Give tmux time to initialize

    # Create temporary file to store head node info with cluster IDs
    HEAD_NODES_FILE=$(mktemp)
    trap "rm -f $HEAD_NODES_FILE $TEMP_FILE" EXIT

    # Extract head node information from assignments
    CLUSTER_ID=0
    for node_rank in "${!NODE_ASSIGNMENTS[@]}"; do
        IFS=':' read -ra ASSIGNMENT <<< "${NODE_ASSIGNMENTS[$node_rank]}"
        ASSIGNMENT_CLUSTER_ID=${ASSIGNMENT[0]}
        ROLE=${ASSIGNMENT[1]}
        HEAD_IP=${ASSIGNMENT[2]}
        RAY_PORT=${ASSIGNMENT[3]}
        CLUSTER_DASHBOARD_PORT=${ASSIGNMENT[4]}
        
        if [ "$ROLE" = "head" ]; then
            echo "$ASSIGNMENT_CLUSTER_ID:$HEAD_IP:$RAY_PORT:$CLUSTER_DASHBOARD_PORT" >> "$HEAD_NODES_FILE"
        fi
    done

    # Set up port forwarding in tmux windows
    DASHBOARD_LOCAL_PORT=9000
    RAY_LOCAL_PORT=10000
    WINDOW_INDEX=0
    
    # Sort by cluster ID and set up port forwarding
    sort -n "$HEAD_NODES_FILE" | while IFS=':' read -r CLUSTER_ID HEAD_IP RAY_PORT DASHBOARD_PORT; do
        echo "Setting up port forwards for Cluster $CLUSTER_ID:"
        echo "  Dashboard: localhost:$DASHBOARD_LOCAL_PORT -> $HEAD_IP:$DASHBOARD_PORT"
        echo "  Ray Port:  localhost:$RAY_LOCAL_PORT -> $HEAD_IP:$RAY_PORT"
        
        WINDOW_NAME="cluster_${CLUSTER_ID}_pf"
        
        if [ $WINDOW_INDEX -eq 0 ]; then
            # Use the initial window for first port forward
            if tmux list-windows -t "$TMUX_SESSION" | grep -q "^0:"; then
                tmux rename-window -t "$TMUX_SESSION:0" "$WINDOW_NAME"
                tmux send-keys -t "$TMUX_SESSION:$WINDOW_NAME" "ssh -L $DASHBOARD_LOCAL_PORT:localhost:$DASHBOARD_PORT -L $RAY_LOCAL_PORT:localhost:$RAY_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $HEAD_IP -N" Enter
            else
                echo "Warning: Initial window not found, creating new window"
                tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME"
                tmux send-keys -t "$TMUX_SESSION:$WINDOW_NAME" "ssh -L $DASHBOARD_LOCAL_PORT:localhost:$DASHBOARD_PORT -L $RAY_LOCAL_PORT:localhost:$RAY_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $HEAD_IP -N" Enter
            fi
        else
            # Create new window for additional port forwards
            tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME"
            tmux send-keys -t "$TMUX_SESSION:$WINDOW_NAME" "ssh -L $DASHBOARD_LOCAL_PORT:localhost:$DASHBOARD_PORT -L $RAY_LOCAL_PORT:localhost:$RAY_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $HEAD_IP -N" Enter
        fi
        
        echo "✓ Port forwards started in tmux window: $WINDOW_NAME"
        
        DASHBOARD_LOCAL_PORT=$((DASHBOARD_LOCAL_PORT + 1))
        RAY_LOCAL_PORT=$((RAY_LOCAL_PORT + 1))
        WINDOW_INDEX=$((WINDOW_INDEX + 1))
    done

    echo "Port forwards running in tmux session: $TMUX_SESSION"
    echo "Use 'tmux attach -t $TMUX_SESSION' to view port forward status"
    echo ""
fi

echo "=== Ray Cluster Connection Information ==="

# Create temporary file to store head node info
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

# Extract head node IPs from the unified script
grep -A2 "Starting Ray cluster.*as head" "$RAY_UNIFIED_SCRIPT" | \
grep "RAY_HEAD_IP" | \
sed 's/.*export RAY_HEAD_IP="\([^"]*\)".*/\1/' > "$TEMP_FILE"

# Get the Ray port (same for all clusters)
CLUSTER_RAY_PORT=${RAY_PORT:-6379}

if [ "$PORT_FORWARD" = "1" ]; then
    echo "Query to clusters using (Direct):"
    
    # Show direct Ray connection info
    CLUSTER_ID=0
    while read -r HEAD_IP; do
        echo "  Cluster $CLUSTER_ID: $HEAD_IP:$CLUSTER_RAY_PORT"
        CLUSTER_ID=$((CLUSTER_ID + 1))
    done < "$TEMP_FILE"
    
    echo ""
    echo "Query to clusters using (Port Forwarded):"
    
    # Show port forwarded Ray ports
    CLUSTER_ID=0
    RAY_LOCAL_PORT=10000
    while read -r HEAD_IP; do
        echo "  Cluster $CLUSTER_ID: localhost:$RAY_LOCAL_PORT"
        RAY_LOCAL_PORT=$((RAY_LOCAL_PORT + 1))
        CLUSTER_ID=$((CLUSTER_ID + 1))
    done < "$TEMP_FILE"
    
    echo ""
    echo "Connect to dashboards using (Direct):"
    
    # Show direct dashboard URLs
    CLUSTER_ID=0
    while read -r HEAD_IP; do
        echo "  Cluster $CLUSTER_ID: http://$HEAD_IP:$CLUSTER_DASHBOARD_PORT"
        CLUSTER_ID=$((CLUSTER_ID + 1))
    done < "$TEMP_FILE"
    
    echo ""
    echo "Connect to dashboards using (Port Forwarded):"
    
    # Show port forwarded dashboard URLs
    CLUSTER_ID=0
    DASHBOARD_LOCAL_PORT=9000
    while read -r HEAD_IP; do
        echo "  Cluster $CLUSTER_ID: http://localhost:$DASHBOARD_LOCAL_PORT"
        CLUSTER_ID=$((CLUSTER_ID + 1))
        DASHBOARD_LOCAL_PORT=$((DASHBOARD_LOCAL_PORT + 1))
    done < "$TEMP_FILE"
    
else
    echo "Query to clusters using:"

    # Sort by cluster ID and print connection info
    CLUSTER_ID=0
    while read -r HEAD_IP; do
        echo "$HEAD_IP:$CLUSTER_RAY_PORT"
        CLUSTER_ID=$((CLUSTER_ID + 1))
    done < "$TEMP_FILE"

    echo ""
    echo "Connect to dashboards using:"

    # Sort by cluster ID and print connection info
    CLUSTER_ID=0
    while read -r HEAD_IP; do
        echo "http://$HEAD_IP:$CLUSTER_DASHBOARD_PORT"
        CLUSTER_ID=$((CLUSTER_ID + 1))
    done < "$TEMP_FILE"
fi

echo ""
