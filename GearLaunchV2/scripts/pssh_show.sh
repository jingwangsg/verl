source /etc/workflow_env
SESSION=default
LINES=-1

while [[ $# -gt 0 ]]; do
    case $1 in
    -s|--session)
        SESSION="$2"
        shift 2
        ;;
    -l|--lines)
        LINES="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

WORKFLOW_DIR="$HOME/.osmo/${WORKFLOW_ID}"

SESSION_DIR="${WORKFLOW_DIR}/pssh/$SESSION/"
# sort files alphabetically, and get the last one
LATEST_DIR=$(ls -v ${SESSION_DIR}/ | tail -n 1)
RUN_DIR="${SESSION_DIR}/${LATEST_DIR}"

echo "RunDir: ${RUN_DIR}"

red_print() {
    echo -e "\033[31m$1\033[0m"
}

# strip empty lines and print out last 20 lines
for FILE in ${RUN_DIR}/pssh_*.txt; do
    NODE_RANK=$(basename $FILE | cut -d. -f1 | cut -d_ -f3)
    red_print "[Node Rank: $NODE_RANK]"
    if [ $LINES -ne -1 ]; then
        cat $FILE | tail -n $LINES
    else
        cat $FILE
    fi
    red_print "----------------------------------------"
    echo ""
done