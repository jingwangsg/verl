#!/usr/bin/env python3
"""
Dataset migration script for FrameThinker

Converts old FrameThinker dataset format to new format with tools_kwargs.

Old format:
    extra_info = {
        "video_path": "...",
        "fps": 30,
        "total_frames": 1800,
        "width": 1280,
        "height": 720
    }

New format:
    extra_info = {
        "tools_kwargs": {
            "video_think": {
                "create_kwargs": {
                    "video_path": "...",
                    "fps": 30,
                    "total_frames": 1800,
                    "width": 1280,
                    "height": 720
                }
            }
        }
    }

Usage:
    python migrate_dataset.py --input train.parquet --output train_migrated.parquet
    python migrate_dataset.py --input train.parquet --output train_migrated.parquet --validate
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single row from old format to new format."""
    extra_info = row.get("extra_info", {})

    # Check if already migrated
    if "tools_kwargs" in extra_info:
        return row

    # Extract old-format metadata
    video_path = extra_info.get("video_path", "")
    if not video_path:
        index = extra_info.get("index", "?")
        print(f"Warning: Row {index} has no video_path - skipping migration")
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


def validate_row(row: Dict[str, Any], index: int) -> bool:
    """Validate that a row has correct format."""
    try:
        extra_info = row.get("extra_info", {})

        assert "tools_kwargs" in extra_info, "Missing tools_kwargs"
        assert "video_think" in extra_info["tools_kwargs"], "Missing video_think"
        assert "create_kwargs" in extra_info["tools_kwargs"]["video_think"], "Missing create_kwargs"

        create_kwargs = extra_info["tools_kwargs"]["video_think"]["create_kwargs"]
        required_fields = ["video_path", "fps", "total_frames", "width", "height"]

        for field in required_fields:
            assert field in create_kwargs, f"Missing required field: {field}"

        return True
    except AssertionError as e:
        print(f"✗ Validation failed for row {index}: {e}")
        return False


def migrate_dataset(input_path: str, output_path: str, validate: bool = False) -> None:
    """Migrate dataset from old format to new format."""
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"✓ Loaded {len(df)} rows")

    # Convert all rows
    print("Migrating rows...")
    df = df.apply(convert_row, axis=1)
    print(f"✓ Migrated {len(df)} rows")

    # Validate if requested
    if validate:
        print("\nValidating migrated dataset...")
        valid_count = 0
        for idx, row in df.iterrows():
            if validate_row(row, idx):
                valid_count += 1

        print(f"✓ Validated {valid_count}/{len(df)} rows")
        if valid_count < len(df):
            print(f"⚠ Warning: {len(df) - valid_count} rows failed validation")

    # Save output
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path)
    print(f"✓ Saved {len(df)} rows")

    # Show sample
    print("\nSample create_kwargs from first row:")
    first_row = df.iloc[0]
    create_kwargs = first_row["extra_info"]["tools_kwargs"]["video_think"]["create_kwargs"]
    for key, value in create_kwargs.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate FrameThinker dataset to new format with tools_kwargs"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate output dataset after migration"
    )

    args = parser.parse_args()

    # Check input exists
    if not Path(args.input).exists():
        print(f"✗ Error: Input file not found: {args.input}")
        return 1

    # Check output doesn't exist (safety)
    if Path(args.output).exists():
        response = input(f"⚠ Output file {args.output} exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted")
            return 1

    # Migrate
    try:
        migrate_dataset(args.input, args.output, args.validate)
        print("\n✓ Migration completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
