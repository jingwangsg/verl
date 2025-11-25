#!/usr/bin/env python3
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data format validation script for reward functions.

This script validates that parquet files contain the required columns and
data format for think_with_video_reward and answer_reward functions.

Usage:
    # Validate for video thinking tasks
    python validate_reward_data_format.py \\
        --data_path /path/to/data.parquet \\
        --reward_type video_think

    # Validate for answer generation tasks
    python validate_reward_data_format.py \\
        --data_path /path/to/data.parquet \\
        --reward_type answer

    # Validate with sample limit
    python validate_reward_data_format.py \\
        --data_path /path/to/data.parquet \\
        --reward_type video_think \\
        --sample_limit 100

    # Save validation report
    python validate_reward_data_format.py \\
        --data_path /path/to/data.parquet \\
        --reward_type answer \\
        --output_report validation_report.txt
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow is required. Install with: pip install pyarrow")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def validate_video_think_row(row: pd.Series, row_idx: int) -> Tuple[bool, List[str]]:
    """
    Validate a single row for video thinking reward.

    Args:
        row: DataFrame row
        row_idx: Row index for error reporting

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check predict_str
    if pd.isna(row.get('predict_str')):
        errors.append(f"Row {row_idx}: 'predict_str' is missing or NaN")
    elif not isinstance(row['predict_str'], str):
        errors.append(f"Row {row_idx}: 'predict_str' is not a string")
    else:
        # Check for think/action pairs
        predict_str = row['predict_str']
        think_count = predict_str.count('<think>')
        action_count = predict_str.count('<action>')
        if think_count == 0 or action_count == 0:
            errors.append(f"Row {row_idx}: 'predict_str' missing <think> or <action> tags")

    # Check ground_truth
    if pd.isna(row.get('ground_truth')):
        errors.append(f"Row {row_idx}: 'ground_truth' is missing or NaN")
    elif not isinstance(row['ground_truth'], str):
        errors.append(f"Row {row_idx}: 'ground_truth' is not a string")

    # Check extra_info
    if pd.isna(row.get('extra_info')):
        errors.append(f"Row {row_idx}: 'extra_info' is missing or NaN")
    else:
        extra_info = row['extra_info']

        # Try to parse if it's a string
        if isinstance(extra_info, str):
            try:
                extra_info = ast.literal_eval(extra_info)
            except:
                try:
                    extra_info = json.loads(extra_info)
                except:
                    errors.append(f"Row {row_idx}: 'extra_info' is not a valid dict/JSON string")
                    extra_info = None

        # Validate dict structure
        if isinstance(extra_info, dict):
            if 'question' not in extra_info:
                errors.append(f"Row {row_idx}: 'extra_info' missing required key 'question'")
            elif not isinstance(extra_info['question'], str):
                errors.append(f"Row {row_idx}: 'extra_info[question]' is not a string")

            if 'total_frames' not in extra_info:
                errors.append(f"Row {row_idx}: 'extra_info' missing required key 'total_frames'")
            elif not isinstance(extra_info['total_frames'], (int, float)):
                errors.append(f"Row {row_idx}: 'extra_info[total_frames]' is not a number")
            elif extra_info['total_frames'] <= 0:
                errors.append(f"Row {row_idx}: 'extra_info[total_frames]' must be positive")
        elif extra_info is not None:
            errors.append(f"Row {row_idx}: 'extra_info' is not a dictionary")

    return len(errors) == 0, errors


def validate_answer_row(row: pd.Series, row_idx: int) -> Tuple[bool, List[str]]:
    """
    Validate a single row for answer reward.

    Args:
        row: DataFrame row
        row_idx: Row index for error reporting

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check predict_str
    if pd.isna(row.get('predict_str')):
        errors.append(f"Row {row_idx}: 'predict_str' is missing or NaN")
    elif not isinstance(row['predict_str'], str):
        errors.append(f"Row {row_idx}: 'predict_str' is not a string")
    else:
        predict_str = row['predict_str']

        # Check for required tags (note: <think> will be prepended by function)
        if '<|begin_of_documents|>' not in predict_str:
            errors.append(f"Row {row_idx}: 'predict_str' missing <|begin_of_documents|> tag")
        if '<|end_of_documents|>' not in predict_str:
            errors.append(f"Row {row_idx}: 'predict_str' missing <|end_of_documents|> tag")
        if '<|begin_of_query|>' not in predict_str:
            errors.append(f"Row {row_idx}: 'predict_str' missing <|begin_of_query|> tag")
        if '<|end_of_query|>' not in predict_str:
            errors.append(f"Row {row_idx}: 'predict_str' missing <|end_of_query|> tag")
        if '</think>' not in predict_str:
            errors.append(f"Row {row_idx}: 'predict_str' missing </think> tag")
        if '<answer>' not in predict_str or '</answer>' not in predict_str:
            errors.append(f"Row {row_idx}: 'predict_str' missing <answer> tags")

        # Check if starts with <think> (should NOT for this function)
        if predict_str.strip().startswith('<think>'):
            print_warning(f"Row {row_idx}: 'predict_str' starts with <think> - this will be prepended automatically")

    # Check ground_truth
    if pd.isna(row.get('ground_truth')):
        errors.append(f"Row {row_idx}: 'ground_truth' is missing or NaN")
    elif not isinstance(row['ground_truth'], str):
        errors.append(f"Row {row_idx}: 'ground_truth' is not a string")

    # extra_info is optional for answer_reward, so just check type if present
    if 'extra_info' in row and not pd.isna(row['extra_info']):
        extra_info = row['extra_info']
        if isinstance(extra_info, str):
            try:
                ast.literal_eval(extra_info)
            except:
                try:
                    json.loads(extra_info)
                except:
                    errors.append(f"Row {row_idx}: 'extra_info' is not a valid dict/JSON string")

    return len(errors) == 0, errors


def validate_parquet_file(
    data_path: str,
    reward_type: str,
    sample_limit: Optional[int] = None
) -> Dict[str, any]:
    """
    Validate parquet file for reward function compatibility.

    Args:
        data_path: Path to parquet file
        reward_type: Type of reward ('video_think' or 'answer')
        sample_limit: Maximum number of rows to validate (None = all)

    Returns:
        Dictionary with validation results
    """
    print_header(f"Validating Data Format for {reward_type.upper()} Reward")

    # Check file exists
    data_path = Path(data_path)
    if not data_path.exists():
        print_error(f"File not found: {data_path}")
        return {"valid": False, "error": "File not found"}

    print_info(f"Loading data from: {data_path}")

    # Load parquet file
    try:
        df = pd.read_parquet(data_path)
        print_success(f"Loaded {len(df)} rows")
    except Exception as e:
        print_error(f"Failed to load parquet file: {e}")
        return {"valid": False, "error": str(e)}

    # Apply sample limit
    if sample_limit and sample_limit < len(df):
        print_info(f"Validating first {sample_limit} rows (sample limit)")
        df = df.head(sample_limit)
    else:
        print_info(f"Validating all {len(df)} rows")

    # Check required columns
    required_cols = ['predict_str', 'ground_truth']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print_error(f"Missing required columns: {missing_cols}")
        print_info(f"Available columns: {list(df.columns)}")
        return {"valid": False, "error": f"Missing columns: {missing_cols}"}

    print_success(f"All required columns present: {required_cols}")

    # Check extra_info column for video_think
    if reward_type == 'video_think':
        if 'extra_info' not in df.columns:
            print_error("Missing required column 'extra_info' for video_think reward")
            return {"valid": False, "error": "Missing extra_info column"}
        print_success("'extra_info' column present")
    else:
        if 'extra_info' in df.columns:
            print_info("'extra_info' column present (optional for answer reward)")

    # Validate rows
    print_info("\nValidating row data...")

    valid_rows = 0
    invalid_rows = 0
    all_errors = []

    validate_func = validate_video_think_row if reward_type == 'video_think' else validate_answer_row

    for idx, row in df.iterrows():
        is_valid, errors = validate_func(row, idx)
        if is_valid:
            valid_rows += 1
        else:
            invalid_rows += 1
            all_errors.extend(errors)

        # Print progress for large datasets
        if (idx + 1) % 1000 == 0:
            print_info(f"Validated {idx + 1} / {len(df)} rows...")

    # Print summary
    print_header("Validation Summary")

    total_rows = len(df)
    print_info(f"Total rows validated: {total_rows}")
    print_success(f"Valid rows: {valid_rows} ({valid_rows/total_rows*100:.1f}%)")

    if invalid_rows > 0:
        print_error(f"Invalid rows: {invalid_rows} ({invalid_rows/total_rows*100:.1f}%)")
        print_info("\nFirst 10 errors:")
        for error in all_errors[:10]:
            print(f"  • {error}")
        if len(all_errors) > 10:
            print_info(f"  ... and {len(all_errors) - 10} more errors")
    else:
        print_success("All rows are valid!")

    # Additional statistics
    print_header("Data Statistics")

    # predict_str length
    if 'predict_str' in df.columns:
        lengths = df['predict_str'].dropna().apply(len)
        print_info(f"predict_str length: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")

    # ground_truth length
    if 'ground_truth' in df.columns:
        gt_lengths = df['ground_truth'].dropna().apply(lambda x: len(str(x).split()))
        print_info(f"ground_truth words: min={gt_lengths.min()}, max={gt_lengths.max()}, mean={gt_lengths.mean():.1f}")

    # Video-specific stats
    if reward_type == 'video_think' and 'extra_info' in df.columns:
        try:
            # Parse extra_info
            parsed_info = []
            for item in df['extra_info'].dropna():
                if isinstance(item, str):
                    try:
                        parsed_info.append(ast.literal_eval(item))
                    except:
                        try:
                            parsed_info.append(json.loads(item))
                        except:
                            pass
                elif isinstance(item, dict):
                    parsed_info.append(item)

            if parsed_info:
                total_frames_list = [info.get('total_frames', 0) for info in parsed_info if isinstance(info, dict)]
                if total_frames_list:
                    print_info(f"total_frames: min={min(total_frames_list)}, max={max(total_frames_list)}, "
                             f"mean={sum(total_frames_list)/len(total_frames_list):.0f}")
        except Exception as e:
            print_warning(f"Could not compute extra_info statistics: {e}")

    result = {
        "valid": invalid_rows == 0,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "errors": all_errors
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate parquet data format for reward functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate for video thinking tasks
  python validate_reward_data_format.py --data_path data.parquet --reward_type video_think

  # Validate for answer generation tasks
  python validate_reward_data_format.py --data_path data.parquet --reward_type answer

  # Validate with sample limit
  python validate_reward_data_format.py --data_path data.parquet --reward_type video_think --sample_limit 100
        """
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to parquet file to validate'
    )

    parser.add_argument(
        '--reward_type',
        type=str,
        required=True,
        choices=['video_think', 'answer'],
        help='Type of reward function (video_think or answer)'
    )

    parser.add_argument(
        '--sample_limit',
        type=int,
        default=None,
        help='Maximum number of rows to validate (default: all rows)'
    )

    parser.add_argument(
        '--output_report',
        type=str,
        default=None,
        help='Path to save validation report (optional)'
    )

    args = parser.parse_args()

    # Run validation
    result = validate_parquet_file(
        data_path=args.data_path,
        reward_type=args.reward_type,
        sample_limit=args.sample_limit
    )

    # Save report if requested
    if args.output_report and result:
        print_info(f"\nSaving validation report to: {args.output_report}")
        with open(args.output_report, 'w') as f:
            f.write(f"Validation Report\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"Data Path: {args.data_path}\n")
            f.write(f"Reward Type: {args.reward_type}\n")
            f.write(f"Sample Limit: {args.sample_limit or 'None (all rows)'}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Rows: {result.get('total_rows', 0)}\n")
            f.write(f"  Valid Rows: {result.get('valid_rows', 0)}\n")
            f.write(f"  Invalid Rows: {result.get('invalid_rows', 0)}\n\n")

            if result.get('errors'):
                f.write(f"Errors:\n")
                for error in result['errors']:
                    f.write(f"  • {error}\n")
        print_success(f"Report saved successfully")

    # Exit with appropriate code
    if result.get('valid', False):
        print_success("\n✓ Validation passed! Data format is correct.")
        sys.exit(0)
    else:
        print_error("\n✗ Validation failed! Please fix the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
