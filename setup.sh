pip install uv
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements_custom.txt
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip uninstall pynvml

sudo apt update
sudo apt install ffmpeg -y
sudo apt install nodejs npm -y
sudo npm install -g n
curl -fsSL https://claude.ai/install.sh | bash

PROJECT_DIR="$(pwd)"

DATASET_DIR=add_path_here
if [[ "$DATASET_DIR" == "add_path_here" ]]; then
    DATASET_DIR="${PROJECT_DIR}/data"
fi

MODEL_WEIGHTS=add_path_here
if [[ "$MODEL_WEIGHTS" == "add_path_here" ]]; then
    MODEL_WEIGHTS="${PROJECT_DIR}/model_weights"
fi

# Download datasets
mkdir -p "$DATASET_DIR"
if [ ! -d "${DATASET_DIR}/Video-Holmes" ]; then
    hf download k-nick/vr_train_vh --repo-type dataset --local-dir ${DATASET_DIR}/vr_train_vh
    mv ${DATASET_DIR}/vr_train_vh ${DATASET_DIR}/Video-Holmes
    cp ${PROJECT_DIR}/recipe/framethinker/data_preprocess/decompress.py ${DATASET_DIR}/Video-Holmes
    cd ${DATASET_DIR}/Video-Holmes
    python decompress.py
    cd ${PROJECT_DIR}
fi

if [ ! -d "${DATASET_DIR}/vr_train_vr1" ]; then
    # huggingface-cli download --repo-type dataset --resume-download k-nick/vr_train_vr1 --local-dir data/video_reason/vr_train_vr1
    hf download k-nick/vr_train_vr1 --repo-type dataset --local-dir ${DATASET_DIR}/vr_train_vr1
    cp ${PROJECT_DIR}/recipe/framethinker/data_preprocess/decompress.py ${DATASET_DIR}/vr_train_vr1
    cd ${DATASET_DIR}/vr_train_vr1
    python decompress.py
    cd ${PROJECT_DIR}
fi

if [ ! -d "${DATASET_DIR}/vr_train_lvr" ]; then
    hf download k-nick/vr_train_lvr --repo-type dataset --local-dir ${DATASET_DIR}/vr_train_lvr
    cp ${PROJECT_DIR}/recipe/framethinker/data_preprocess/decompress.py ${DATASET_DIR}/vr_train_lvr
    cd ${DATASET_DIR}/vr_train_lvr
    python decompress.py
    cd ${PROJECT_DIR}
fi

# preprocess datasets to parquet format
cd "$PROJECT_DIR"
chmod a+x recipe/framethinker/data_preprocess/convert_to_rl_parquet.sh
bash "recipe/framethinker/data_preprocess/convert_to_rl_parquet.sh" "$DATASET_DIR"

# download model weights
mkdir -p "$MODEL_WEIGHTS"
if [ ! -d "model_weights/Qwen2.5-VL-7B-Instruct" ]; then
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ${MODEL_WEIGHTS}/Qwen2.5-VL-7B-Instruct
fi

# if [ ! -d "model_weights/ft_coldstart" ]; then
#     hf download k-nick/ft_coldstart --repo-type dataset --local-dir ${MODEL_WEIGHTS}/ft_coldstart
# fi
