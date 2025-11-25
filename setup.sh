pip install uv
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements_custom.txt
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip uninstall pynvml
# uv pip install nvidia-ml-py

sudo apt update
sudo apt install ffmpeg -y
sudo apt install nodejs npm -y
sudo npm install -g n
curl -fsSL https://claude.ai/install.sh | bash

# download dataset
mkdir -p data
mkdir -p data/video_reason
if [ ! -d "data/video_reason/Video-Holmes" ]; then
    huggingface-cli download --repo-type dataset --resume-download k-nick/video_reason --local-dir data/video_reason/Video-Holmes
fi

cp recipe/framethinker/data_preprocess/decompress.py data/video_reason/Video-Holmes
cd data/video_reason/Video-Holmes
python decompress.py
cd ../../../

chmod a+x recipe/framethinker/data_preprocess/convert_video_holmes.sh
./recipe/framethinker/data_preprocess/convert_video_holmes.sh

# download model weights
mkdir -p model_weights
if [ ! -d "model_weights/Qwen2.5-VL-7B-Instruct" ]; then
    huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir model_weights/Qwen2.5-VL-7B-Instruct
fi

if [ ! -d "model_weights/ft_coldstart" ]; then
    huggingface-cli download --repo-type dataset --resume-download k-nick/ft_coldstart --local-dir model_weights/ft_coldstart
fi

# start to train the model
# ray start --head --resources='{"drivers": 1}'
# chmod a+x recipe/framethinker/train_frame_thinker_debug.sh
# ./recipe/framethinker/train_frame_thinker_debug.sh