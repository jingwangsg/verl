#!/bin/bash

source /etc/workflow_env


VENV=${1:-"$HOME/WORKSPACE/playground/notebook/.venv"}
PORT=${PORT:-9000}

source $VENV/bin/activate
set -ex

uv pip install jupyter jupytext

if tmux has-session -t jupyter 2>/dev/null; then
    tmux kill-session -t jupyter
fi
tmux new-session -d -s jupyter "jupyter notebook --no-browser --ip=0.0.0.0 --port=$PORT --NotebookApp.allow_remote_access=True --FileContentsManager.delete_to_trash=False --NotebookApp.token=''; sleep infinity"