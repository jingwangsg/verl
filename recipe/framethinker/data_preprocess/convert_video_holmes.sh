
PROJECT_DIR="$(pwd)"

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/Video-Holmes/test.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/Video-Holmes/test.parquet --data-source TencentARC/Video-Holmes

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/Video-Holmes/train.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/Video-Holmes/train.parquet --data-source TencentARC/Video-Holmes