python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/LongVideoReason/train.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/LongVideoReason/train.parquet --data-source LongVideo-Reason --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/LongVideoReason/train_mcq_10k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/LongVideoReason/train_mcq_10k.parquet --data-source LongVideo-Reason --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/LongVideoReason/train_mcq_3k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/LongVideoReason/train_mcq_3k.parquet --data-source LongVideo-Reason --num-proc 32
