PROJECT_DIR="$(pwd)"

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/LongVideoReason/train_mcq_10k.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/LongVideoReason/train_mcq_10k.parquet --data-source LongVideo-Reason

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/LongVideoReason/train_mcq_3k.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/LongVideoReason/train_mcq_3k.parquet --data-source LongVideo-Reason

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/LongVideoReason/test.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/LongVideoReason/test.parquet --data-source LongVideo-Reason