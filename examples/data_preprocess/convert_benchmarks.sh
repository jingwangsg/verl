#!/usr/bin/env bash
set -euo pipefail

# Run from agent_rl/verl
MEDIA_ROOT="/mnt/amlfs-02/shared/datasets/s3/video_reason"
BENCH_DIR="$MEDIA_ROOT/BENCHMARK"

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/MMVU/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/MMVU/test.parquet" \
  --data-source "MMVU" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/Video-MMMU/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/Video-MMMU/test.parquet" \
  --data-source "Video-MMMU" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/VSI-Bench/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/VSI-Bench/test.parquet" \
  --data-source "VSI-Bench" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/VRBench/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/VRBench/test.parquet" \
  --data-source "VRBench" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/TempCompass/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/TempCompass/test.parquet" \
  --data-source "TempCompass" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/CG-Bench/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/CG-Bench/test.parquet" \
  --data-source "CG-Bench" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/Video-Holmes/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/Video-Holmes/test.parquet" \
  --data-source "Video-Holmes" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/MMR-V/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/MMR-V/test.parquet" \
  --data-source "MMR-V" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/MLVU/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/MLVU/test.parquet" \
  --data-source "MLVU" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/LongVideoReason/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/LongVideoReason/test.parquet" \
  --data-source "LongVideoReason" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/LongVideoBench/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/LongVideoBench/test.parquet" \
  --data-source "LongVideoBench" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/MVBench/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/MVBench/test.parquet" \
  --data-source "MVBench" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/Video-MME/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/Video-MME/test.parquet" \
  --data-source "Video-MME" \
  --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py \
  "$BENCH_DIR/VideoMME/test.json" \
  --media-dir "$MEDIA_ROOT/" \
  -o "$BENCH_DIR/VideoMME/test.parquet" \
  --data-source "VideoMME" \
  --num-proc 32
