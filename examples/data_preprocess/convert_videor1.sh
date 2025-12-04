python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_video_mcq_10k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_video_mcq_10k.parquet --data-source Video-R1 --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_video_mcq_30k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_video_mcq_30k.parquet --data-source Video-R1 --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_video_nofree_106k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_video_nofree_106k.parquet --data-source Video-R1 --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_image_mcq_10k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_image_mcq_10k.parquet --data-source Video-R1 --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_image_mcq_30k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_image_mcq_30k.parquet --data-source Video-R1 --num-proc 32

python examples/data_preprocess/convert_to_rl_parquet.py /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_image_nofree_118k.json --media-dir /mnt/amlfs-02/shared/datasets/s3/video_reason/ -o /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-R1/train_image_nofree_118k.parquet --data-source Video-R1 --num-proc 32
