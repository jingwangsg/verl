#!/bin/bash
DATASET_DIR="$1"

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py \
    "${DATASET_DIR}/Video-Holmes/test.json" \
    --num-frames 8 \
    --num-proc 16 \
    --media-dir "${DATASET_DIR}" \
    -o "${DATASET_DIR}/Video-Holmes/test.parquet" \
    --data-source TencentARC/Video-Holmes

python recipe/framethinker/data_preprocess/convert_to_rl_parquet.py \
    "${DATASET_DIR}/Video-Holmes/train.json" \
    --num-frames 8 \
    --num-proc 16 \
    --media-dir "${DATASET_DIR}" \
    -o "${DATASET_DIR}/Video-Holmes/train.parquet" \
    --data-source TencentARC/Video-Holmes