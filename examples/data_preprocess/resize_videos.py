#!/usr/bin/env python3
"""
Resize benchmark videos so that the short side <= max_short (default 512).

Features
- Reads video paths from benchmark test.json files
- Skips videos already within target size
- Multiprocessing with ffmpeg re-encode to H.264 mp4
- Replaces source files atomically (writes temp then rename)

Usage examples
  python examples/data_preprocess/resize_videos.py \
    --benchmarks LongVideoReason MMR-V Video-MME Video-MMMU \
    --media-root /mnt/amlfs-02/shared/datasets/s3/video_reason \
    --num-proc 16

  # dry-run just to see counts
  python examples/data_preprocess/resize_videos.py --dry-run
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Tuple
from collections import Counter

from torchcodec.decoders import VideoDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="Resize benchmark videos to a max short side")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["LongVideoReason", "MMR-V", "Video-MME", "Video-MMMU"],
        help="Benchmark names under BENCHMARK/ to process (default: high-res ones)",
    )
    parser.add_argument(
        "--media-root",
        type=str,
        default="/mnt/amlfs-02/shared/datasets/s3/video_reason",
        help="Root directory containing BENCHMARK/",
    )
    parser.add_argument(
        "--max-short",
        type=int,
        default=512,
        help="Target maximum for the short side (pixels)",
    )
    parser.add_argument("--num-proc", type=int, default=16, help="Multiprocessing workers")
    parser.add_argument("--dry-run", action="store_true", help="Only report counts, no writes")
    return parser.parse_args()


def load_video_paths(json_path: Path) -> Iterable[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        vp = item.get("video_path")
        if vp:
            yield vp


def needs_resize(path: Path, max_short: int) -> Tuple[bool, int, int]:
    """Return (need, width, height) for a video."""
    vr = VideoDecoder(str(path), num_ffmpeg_threads=0)
    w = int(vr.metadata.width)
    h = int(vr.metadata.height)
    short = min(w, h)
    return short > max_short, w, h


def ffmpeg_resize(src: Path, dst: Path, max_short: int) -> None:
    """Resize with ffmpeg so short side = max_short, keep aspect ratio, h264 mp4."""
    scale_filter = "scale='if(lt(iw,ih),{m},-2)':'if(lt(iw,ih),-2,{m})'".format(m=max_short)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(src),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def process_one(path_str: str, max_short: int, dry_run: bool) -> Tuple[str, str]:
    path = Path(path_str)
    try:
        need, w, h = needs_resize(path, max_short)
    except Exception as e:
        return path_str, f"meta_error:{e}"

    if not need:
        return path_str, "skip"

    if dry_run:
        return path_str, f"would_resize({w}x{h})"

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="resize_tmp_", dir=str(path.parent), delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        ffmpeg_resize(path, tmp_path, max_short)
        # replace original
        shutil.move(tmp_path, path)
        return path_str, f"resized({w}x{h})-><= {max_short}"
    except Exception as e:
        # cleanup temp
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return path_str, f"resize_error:{e}"


def main():
    args = parse_args()
    base = Path(args.media_root)
    bench_root = base / "BENCHMARK"

    tasks = []
    for name in args.benchmarks:
        json_path = bench_root / name / "test.json"
        if not json_path.exists():
            print(f"[skip] {name}: no test.json")
            continue
        paths = list(load_video_paths(json_path))
        unique_paths = list(dict.fromkeys(paths))  # dedupe, preserve order
        if len(unique_paths) > 50000:
            # avoid accidental blow-up
            unique_paths = unique_paths[:50000]
        abs_paths = [str(base / p) for p in unique_paths]
        tasks.extend(abs_paths)
        print(f"[queue] {name}: {len(abs_paths)} videos")

    print(f"Total queued videos: {len(tasks)}")
    if not tasks:
        return

    worker = partial(process_one, max_short=args.max_short, dry_run=args.dry_run)
    results = Counter()
    start = time.time()
    with Pool(processes=args.num_proc) as pool:
        for _, status in pool.imap_unordered(worker, tasks, chunksize=8):
            results[status] += 1
    elapsed = time.time() - start

    print("Done in {:.1f}s".format(elapsed))
    for k, v in results.most_common():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
