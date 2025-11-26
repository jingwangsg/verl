set -ex
source /etc/workflow_env
WORKFLOW_DIR="$HOME/.osmo/${WORKFLOW_ID}"
PWD_DIR=$(pwd)

SESSION="default"
VERBOSE=0
CMD="sleep infinity"
TIMEOUT=0

while [[ $# -gt 0 ]]; do
    case $1 in
    -s | --session)
        SESSION="$2"
        shift 2
        ;;
    -v | --verbose)
        VERBOSE=1
        shift
        ;;
    --timeout)
        TIMEOUT="$2"
        shift 2
        ;;
    --node-ranks)
        NODE_RANKS="$2"
        shift 2
        ;;
    --cmd)
        # Collect all remaining arguments as the command
        shift
        CMD="$*"
        break
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

RUN_DIR="${WORKFLOW_DIR}/pssh/$SESSION/$(date +%s)"
RUN_BASH="${RUN_DIR}/run.sh"

mkdir -p "${RUN_DIR}"

OUTPUT_FILE_PREFIX="${RUN_DIR}/pssh_${SESSION}"
FILE_SYSTEM=$(df -h | grep /mnt/ | awk '{print $6}' | grep -v /home/ | sort | head -n 1)
if [[ "$FILE_SYSTEM" == *"/aws-lfs-01/"* ]]; then
    export HOME=/mnt/aws-lfs-01/shared/jingwang/
elif [[ "$FILE_SYSTEM" == *"/amlfs-04/"* ]]; then
    export HOME=/mnt/amlfs-04/home/jingwang/
elif [[ "$FILE_SYSTEM" == *"/lfs/"* ]]; then
    export HOME=/mnt/lfs/home/jingwang/
elif [[ "$FILE_SYSTEM" == *"/amlfs-01/"* ]]; then
    export HOME=/mnt/amlfs-03/shared/jingwang/
fi 

cat >"${RUN_BASH}" <<EOF
source /etc/workflow_env
export HOME=$HOME

BASH_ONLY=1 source \$HOME/.bashrc

cd ${PWD_DIR}

tmux kill-session -t "$SESSION" > /dev/null 2>&1
tmux new-session -d -s "$SESSION" "source /etc/workflow_env && BASH_ONLY=1 bash"

tmux pipe-pane -t "$SESSION" 'cat > ${OUTPUT_FILE_PREFIX}_\$(grep NODE_RANK /etc/workflow_env | cut -d= -f2).txt'
tmux send-keys -t "$SESSION" '${CMD}' C-m

if [ "${VERBOSE}" = "1" ]; then
    sleep 2
    cat ${OUTPUT_FILE_PREFIX}_\${NODE_RANK}.txt
fi
EOF

if [ -n "${NODE_RANKS}" ]; then
    TMP_HOST_FILE="${RUN_DIR}/hosts.txt"
    NODE_RANKS_ARRAY=(${NODE_RANKS//,/ })
    for NODE_RANK in ${NODE_RANKS_ARRAY[@]}; do
        # select row from /etc/osmo_hosts.txt
        HOST_FILE="${WORKFLOW_DIR}/hosts/host_${NODE_RANK}.txt"
        cat ${HOST_FILE} >>${TMP_HOST_FILE}
    done
    HOST_FILE="${TMP_HOST_FILE}"
else
    HOST_FILE="/etc/osmo_hosts.txt"
fi

NNODES=$(wc -l ${HOST_FILE} | awk '{print $1}')

set -ex
if [ "$TIMEOUT" -gt 0 ]; then
    parallel-ssh -P -t $TIMEOUT -h ${HOST_FILE} -i -x "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" "bash ${RUN_BASH}"
else
    parallel-ssh -P -h ${HOST_FILE} -i -x "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" "bash ${RUN_BASH}"
fi
set +ex

echo "RunDir: ${RUN_DIR}"
echo "Command: ${CMD}"
