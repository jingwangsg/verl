PROJECT_DIR="$(pwd)"

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/BENCHMARKS/eval_mmvu.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/BENCHMARKS/mmvu.parquet --data-source mmvu

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/BENCHMARKS/eval_videommmu_mcq.json --num-frames 8 --num-proc 16 --media-dir "$PROJECT_DIR/data/video_reason/" -o $PROJECT_DIR/data/video_reason/BENCHMARKS/videommmu_mcq.parquet --data-source videommmu_mcq

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py $PROJECT_DIR/data/video_reason/BENCHMARKS/eval_vsibench_mcq.json --num-frames 8 --num-proc 16 --media-dir $PROJECT_DIR/data/video_reason/ -o $PROJECT_DIR/data/video_reason/BENCHMARKS/vsibench_mcq.parquet --data-source vsibench_mcq