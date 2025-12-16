#!/usr/bin/env bash
set -euo pipefail

# Run from agent_rl/verl
MEDIA_ROOT="/mnt/amlfs-02/shared/datasets/s3/video_reason"
BENCH_DIR="$MEDIA_ROOT/BENCHMARK"
export DECORD_EOF_RETRY_MAX=40960

# Explicit list of dataset splits to convert (MCQ-only variant for VideoMMMU included)
DATA_FILES=(
  "$BENCH_DIR/CG-AV-Counting/test.json"
  "$BENCH_DIR/LVBench/test.json"
  "$BENCH_DIR/LongVideoBench/test.json"
  "$BENCH_DIR/LongVideoReason/test.json"
  "$BENCH_DIR/MLVU/test.json"
  "$BENCH_DIR/MLVU/test_mcq.json"
  "$BENCH_DIR/MMR-V/test.json"
  "$BENCH_DIR/MMVU/test.json"
  "$BENCH_DIR/MVBench/test.json"
  "$BENCH_DIR/TempCompass/test.json"
  "$BENCH_DIR/VCRBench/test.json"
  "$BENCH_DIR/VRBench/test.json"
  "$BENCH_DIR/VSI-Bench/test.json"
  "$BENCH_DIR/Video-Holmes/test.json"
  "$BENCH_DIR/VideoMME/test.json"
  "$BENCH_DIR/VideoMMMU/test.json"
  "$BENCH_DIR/VideoMMMU/test_mcq.json"
  "$BENCH_DIR/VideoMathQA/test.json"
)

for json_file in "${DATA_FILES[@]}"; do
  dataset_dir=$(basename "$(dirname "$json_file")")
  file_base=$(basename "$json_file")

  data_source="$dataset_dir"
  if [[ "$file_base" != "test.json" ]]; then
    stem=${file_base%.*}
    data_source="${dataset_dir}_${stem}"
  fi

  output="${json_file%.*}.parquet"

  python examples/data_preprocess/convert_to_rl_parquet.py \
    "$json_file" \
    --media-dir "$MEDIA_ROOT/" \
    -o "$output" \
    --data-source "$data_source" \
    --num-proc 64
done
