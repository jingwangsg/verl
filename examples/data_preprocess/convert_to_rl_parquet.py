#!/usr/bin/env python3
"""
Convert Video-Holmes JSON files to RL training parquet format.

This script creates complete RL-format parquet files from Video-Holmes JSON,
matching the official format from GitHub issue #4.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image
import numpy as np

# Increase decord EOF retry budget to better tolerate slow tails
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")
from decord import VideoReader, cpu

# Avoid EOF retry errors on some long/slow videos.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")


def build_prompt(question: str) -> List[Dict]:
    """
    Build prompt with system message and user message with frame placeholders.

    Args:
        question: Question text
        frame_indices: List of actual frame indices to display

    Returns:
        List of message dicts
    """
    return [{"role": "user", "content": question}]


def sanitize_json_file(json_path: str) -> str:
    """
    Strip or clear metadata to avoid pyarrow schema errors like
    `cannot mix list and non-list, non-null values`. Writes a sanitized copy
    to /tmp and returns the path; falls back to the original on failure.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON for sanitization: {e}")
        return json_path

    if not isinstance(data, list):
        # Unexpected structure; skip sanitization.
        return json_path

    changed = False
    for item in data:
        if item.get("metadata"):
            item["metadata"] = {}
            changed = True
        elif "metadata" not in item:
            item["metadata"] = {}
            changed = True

    if not changed:
        return json_path

    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{Path(json_path).stem}_nometa_", suffix=".json", dir="/tmp"
    )
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)

    print(f"Metadata stripped; using sanitized JSON at: {tmp_path}")
    return tmp_path


def process_single_sample(
    example: Dict[str, Any],
    idx: int,
    media_dir: str,
    data_source: str = "TencentARC/Video-Holmes",
    dataset_short: str | None = None,
) -> Dict[str, Any]:
    """
    Process a single sample for datasets.map().

    Args:
        example: Single sample from dataset
        idx: Sample index
        media_dir: Directory containing media files

    Returns:
        Processed sample in RL format, or None if processing failed
    """
    try:
        # Resolve media path (video or image)
        video_path = example.get("video_path")
        image_path = example.get("image_path")

        # Use only the explicitly provided field: prefer video_path when present, otherwise image_path.
        media_path = video_path if video_path else image_path
        if not media_path:
            print(f"\nWarning: Missing media path for sample {idx}")
            return None

        abs_media_path = os.path.join(media_dir, media_path)
        if not Path(abs_media_path).exists():
            print(f"\nWarning: Media not found: {abs_media_path}")
            return None

        # Branch: image vs video
        ext = Path(abs_media_path).suffix.lower()
        is_image = ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

        if is_image:
            with Image.open(abs_media_path) as img:
                width, height = img.size
            fps = 0
            total_frames = 1
        else:
            try:
                vr = VideoReader(abs_media_path, ctx=cpu(0))
                indices = np.linspace(0, len(vr) - 1, 8).astype(int).tolist()
                example_frames = vr.get_batch(indices)

                total_frames = len(vr)
                fps = vr.get_avg_fps()

                first_frame = vr[0].asnumpy()
                height, width = first_frame.shape[:2]

                # Fallbacks for edge cases where metadata is partial
                if not total_frames or total_frames <= 0:
                    total_frames = len(vr)
                if not fps or fps <= 0:
                    # Avoid div-by-zero; estimate from duration via timestamps if available
                    try:
                        if total_frames > 0:
                            ts = vr.get_frame_timestamp(total_frames - 1)
                            end_ts = ts[1] if len(ts) > 1 else None
                            duration = float(end_ts) if end_ts is not None else None
                        else:
                            duration = None
                        fps = total_frames / duration if duration and duration > 0 else 0
                    except Exception:
                        fps = 0
            except Exception as e:
                print(f"[decord] Failed to read video {abs_media_path}: {e}")
                raise

        # Build prompt with actual frame indices
        question = example["question"]
        prompt = build_prompt(question)

        # Get ground truth
        ground_truth = example["answer"]

        # Build extra_info
        extra_info = {
            # Use decoder metadata directly; some files miss ffprobe nb_frames
            "fps": float(fps),
            "height": int(height),
            "width": int(width),
            "total_frames": int(total_frames),
            ("video_path" if video_path else "image_path"): media_path,  # keep original key
            "answer": ground_truth,
            "question": question,
            "split": example.get("metadata", {}).get("split", "train"),
            "index": idx,
        }

        # tool metadata only makes sense for video samples
        if video_path:
            extra_info["tools_kwargs"] = {
                "video_think": {
                    "create_kwargs": {
                        "video_path": video_path,
                        "fps": fps,
                        "total_frames": total_frames,
                        "width": width,
                        "height": height,
                    }
                }
            }

        # Add optional fields if present
        if "thinking" in example:
            extra_info["thinking"] = example["thinking"]
        if "explanation" in example.get("metadata", {}):
            extra_info["explanation"] = example["metadata"]["explanation"]

        # Build metadata with unified question_id.
        # NOTE: downstream we concat Parquet shards across datasets. Allowing a
        # nested dict here makes Arrow infer a struct with fields that vary per
        # source (and even per row), which then explodes when concatenating.
        # To keep schema stable, serialize metadata to a JSON string.
        metadata = dict(example.get("metadata", {}))
        short_name = dataset_short or data_source
        metadata["question_id"] = f"{short_name}_{idx}"
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        # Build RL format sample
        sample = {
            "data_source": data_source,
            "prompt": prompt,
            # "images": frames,
            "ability": "vl_video_reasoning",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "ground_truth": ground_truth,
            "question_type": example.get("question_type", "mcq"),
            "metadata": metadata_json,
            "extra_info": extra_info,
        }

        media_key = "video_path" if video_path else "image_path"
        sample[media_key] = media_path

        return sample

    except Exception as e:
        print(f"\nError processing sample {idx}: {e}")
        return None


def get_empty_sample_schema():
    """
    Return empty schema matching successful samples for failed processing.

    This ensures schema consistency across all samples during multiprocessing,
    preventing KeyError when datasets.map() tries to merge results.
    """
    return {
        "_skip": True,
        "data_source": "",
        "prompt": [],
        "ability": "",
        "reward_model": {},
        "ground_truth": "",
        "question_type": "",
        "metadata": "",
        "extra_info": {},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Video-Holmes JSON to RL parquet format (using datasets with multiprocessing)"
    )
    parser.add_argument(
        "json_file", type=str, help="Path to input JSON file (train.json or test.json)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output parquet file (default: same name as JSON with .parquet extension)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="TencentARC/Video-Holmes",
        help="Data source name (default: TencentARC/Video-Holmes)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel processing (default: number of CPU cores)",
    )
    parser.add_argument(
        "--media-dir",
        type=str,
        default=None,
        help="Directory containing media files (default: auto: json_file/../../)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = Path(args.json_file).with_suffix(".parquet")
    else:
        output_path = Path(args.output)

    # Determine media dir
    media_dir = args.media_dir
    if media_dir is None:
        # Heuristic: json_file is typically .../BENCHMARK/<name>/split.json
        # So take grandparent to get media root (video_reason)
        json_parent = Path(args.json_file).resolve()
        try:
            media_dir = str(json_parent.parents[2])
        except IndexError:
            print("Cannot infer media-dir; please pass --media-dir explicitly.")
            exit(1)
    print(f"Media dir: {media_dir}")

    # Determine number of processes
    num_proc = args.num_proc if args.num_proc is not None else 1
    print(f"Using {num_proc} processes for processing")

    # Load dataset using datasets library
    print(f"Loading JSON from: {args.json_file}")
    sanitized_json = sanitize_json_file(args.json_file)
    dataset = load_dataset("json", data_files=sanitized_json, split="train")
    print(f"Found {len(dataset)} samples")

    # Process dataset using map with multiprocessing
    print(f"Processing samples with {num_proc} processes...")

    def process_wrapper(example, idx):
        """Wrapper function for datasets.map()"""
        result = process_single_sample(
            example=example,
            idx=idx,
            media_dir=media_dir,
            data_source=args.data_source,
            dataset_short=args.data_source,
        )
        # Return consistent schema if processing failed (will be filtered out)
        if result is None:
            return get_empty_sample_schema()
        # Add skip flag to successful results for consistency
        result["_skip"] = False
        return result

    processed_dataset = dataset.map(
        process_wrapper, with_indices=True, num_proc=num_proc, desc="Processing samples"
    )

    # Filter out failed samples
    if "_skip" in processed_dataset.column_names:
        original_len = len(processed_dataset)
        processed_dataset = processed_dataset.filter(
            lambda x: "_skip" not in x or not x["_skip"]
        )
        processed_dataset = processed_dataset.remove_columns(["_skip"])
        failed = original_len - len(processed_dataset)
        print(f"Filtered out {failed} failed samples")

    if len(processed_dataset) == 0:
        print("No samples processed successfully; aborting write.")
        return

    print(f"\nSuccessfully processed {len(processed_dataset)} samples")

    # Save to parquet
    print(f"Saving to: {output_path}")
    processed_dataset.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
