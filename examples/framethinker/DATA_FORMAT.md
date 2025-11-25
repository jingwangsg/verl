# FrameThinker Dataset Format

This document explains the required data format for training FrameThinker with VideoThinkTool.

## Overview

The FrameThinker training pipeline requires video metadata to be passed to the `VideoThinkTool` for proper video processing. This is done via the `tools_kwargs` field in the dataset.

## Parquet Format

Your training/validation parquet files must contain an `extra_info` column with the following structure:

```python
{
    "index": 0,  # Optional: sample index
    "tools_kwargs": {
        "video_think": {  # Tool name must match the tool schema name
            "create_kwargs": {
                "video_path": "/path/to/video.mp4",  # Absolute or relative to media_dir
                "fps": 30.0,                          # Video FPS (float or int)
                "total_frames": 9000,                 # Total number of frames
                "width": 1920,                        # Video width in pixels
                "height": 1080                        # Video height in pixels
            }
        }
    }
}
```

## Example Data Row

Here's a complete example of a single row in the parquet file:

```python
{
    "data_source": "video_holmes",
    "prompt": "Analyze the video and answer: What happened in the scene?",
    "answer": "A person walked across the street.",
    "extra_info": {
        "index": 0,
        "tools_kwargs": {
            "video_think": {
                "create_kwargs": {
                    "video_path": "videos/sample_001.mp4",
                    "fps": 30,
                    "total_frames": 1800,
                    "width": 1280,
                    "height": 720
                }
            }
        }
    }
}
```

## Field Descriptions

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `video_path` | str | Path to video file (relative to `data.media_dir` or absolute) |
| `fps` | float/int | Frames per second of the video |
| `total_frames` | int | Total number of frames in the video |
| `width` | int | Video width in pixels |
| `height` | int | Video height in pixels |

### Path Resolution

The `video_path` can be:

1. **Relative path**: Joined with `data.media_dir` config parameter
   ```python
   video_path = "videos/sample_001.mp4"
   media_dir = "/mnt/amlfs-02/shared/datasets/s3/video_reason/"
   # Final path: /mnt/amlfs-02/shared/datasets/s3/video_reason/videos/sample_001.mp4
   ```

2. **Absolute path**: Used as-is
   ```python
   video_path = "/absolute/path/to/video.mp4"
   # Final path: /absolute/path/to/video.mp4
   ```

## Creating Dataset from Existing Data

If you have an existing dataset without `tools_kwargs`, you can add it via preprocessing:

### Option 1: Python Script

```python
import pandas as pd
from torchcodec.decoders import VideoDecoder

def add_video_metadata(row, media_dir):
    """Add video metadata to extra_info"""
    video_path = row.get("video_path", "")
    if not video_path:
        return row

    # Load video to get metadata
    full_path = f"{media_dir}/{video_path}"
    decoder = VideoDecoder(full_path, num_ffmpeg_threads=0)
    metadata = decoder.metadata

    # Create tools_kwargs
    if "extra_info" not in row or not row["extra_info"]:
        row["extra_info"] = {}

    row["extra_info"]["tools_kwargs"] = {
        "video_think": {
            "create_kwargs": {
                "video_path": video_path,
                "fps": float(metadata.average_fps),
                "total_frames": int(metadata.num_frames),
                "width": int(metadata.width),
                "height": int(metadata.height)
            }
        }
    }

    return row

# Load existing dataset
df = pd.read_parquet("original_train.parquet")

# Add metadata
media_dir = "/mnt/amlfs-02/shared/datasets/s3/video_reason/"
df = df.apply(lambda row: add_video_metadata(row, media_dir), axis=1)

# Save updated dataset
df.to_parquet("train_with_metadata.parquet")
```

### Option 2: Use Existing Data Structure

If your dataset already has video metadata in `extra_info` like the old FrameThinker format:

```python
# Old format
extra_info = {
    "fps": 30,
    "video_path": "videos/sample_001.mp4",
    "total_frames": 1800,
    "height": 720,
    "width": 1280
}

# Convert to new format
def convert_extra_info(row):
    extra_info = row.get("extra_info", {})
    if "tools_kwargs" not in extra_info and "video_path" in extra_info:
        extra_info["tools_kwargs"] = {
            "video_think": {
                "create_kwargs": {
                    "video_path": extra_info["video_path"],
                    "fps": extra_info.get("fps", 30),
                    "total_frames": extra_info.get("total_frames", 0),
                    "width": extra_info.get("width", 0),
                    "height": extra_info.get("height", 0)
                }
            }
        }
    row["extra_info"] = extra_info
    return row

df = pd.read_parquet("old_format.parquet")
df = df.apply(convert_extra_info, axis=1)
df.to_parquet("new_format.parquet")
```

## Configuration

In your training script, ensure these settings:

```bash
# Enable tools_kwargs in dataset
data.return_raw_chat=True

# Set media directory (for relative video paths)
data.media_dir=/mnt/amlfs-02/shared/datasets/s3/video_reason/

# Optional: Enable tools_kwargs validation
+data.need_tools_kwargs=True  # Warns if tools_kwargs is empty
```

## Validation

To verify your dataset has the correct format:

```python
import pandas as pd

df = pd.read_parquet("train.parquet")

# Check for extra_info column
assert "extra_info" in df.columns, "Missing extra_info column"

# Check first row
sample = df.iloc[0]
extra_info = sample["extra_info"]

assert "tools_kwargs" in extra_info, "Missing tools_kwargs in extra_info"
assert "video_think" in extra_info["tools_kwargs"], "Missing video_think in tools_kwargs"
assert "create_kwargs" in extra_info["tools_kwargs"]["video_think"], "Missing create_kwargs"

create_kwargs = extra_info["tools_kwargs"]["video_think"]["create_kwargs"]
required_fields = ["video_path", "fps", "total_frames", "width", "height"]

for field in required_fields:
    assert field in create_kwargs, f"Missing required field: {field}"

print("✓ Dataset format is valid!")
print(f"Sample create_kwargs: {create_kwargs}")
```

## Troubleshooting

### Issue: "tools_kwargs is empty"

**Cause**: Dataset missing `extra_info.tools_kwargs` field

**Solution**: Add `tools_kwargs` to your dataset using the scripts above

### Issue: "video_path not found"

**Cause**: Incorrect `media_dir` or malformed video path

**Solution**:
- Check `data.media_dir` points to correct directory
- Verify video files exist at the specified paths
- Use absolute paths for debugging

### Issue: VideoDecoder errors

**Cause**: Invalid video metadata (fps, total_frames, etc.)

**Solution**:
- Extract actual metadata from video files using torchcodec
- Don't hardcode metadata - read from actual video files
- Ensure videos are not corrupted

## Migration from Old FrameThinker Dataset

If you have the old FrameThinker dataset format, run this migration:

```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl

# Create migration script
python3 << 'EOF'
import pandas as pd

def migrate_dataset(input_path, output_path):
    df = pd.read_parquet(input_path)

    def convert_row(row):
        extra_info = row.get("extra_info", {})

        # Check if already migrated
        if "tools_kwargs" in extra_info:
            return row

        # Extract old-format metadata
        video_path = extra_info.get("video_path", "")
        if not video_path:
            print(f"Warning: Row {extra_info.get('index', '?')} has no video_path")
            return row

        # Create new format
        extra_info["tools_kwargs"] = {
            "video_think": {
                "create_kwargs": {
                    "video_path": video_path,
                    "fps": extra_info.get("fps", 30),
                    "total_frames": extra_info.get("total_frames", 0),
                    "width": extra_info.get("width", 0),
                    "height": extra_info.get("height", 0)
                }
            }
        }

        row["extra_info"] = extra_info
        return row

    df = df.apply(convert_row, axis=1)
    df.to_parquet(output_path)
    print(f"✓ Migrated {len(df)} rows")
    print(f"✓ Saved to {output_path}")

# Migrate train and test sets
migrate_dataset(
    "/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train.parquet",
    "/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet"
)

migrate_dataset(
    "/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test.parquet",
    "/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet"
)
EOF
```

Then update your training script to use the migrated files:

```bash
TRAIN_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet
VAL_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet
```
