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
Reward function for think_with_video interactive video reasoning tasks.

This reward function validates and scores agent trajectories that explore videos
through actions like frame selection, time queries, and zooming, before outputting
a final answer.

Migrated from FrameThinker-RL with modifications:
- Changed return format from tuple to dict for async compatibility
- Added comprehensive docstrings
- Maintained all validation logic
"""

import re
from typing import Dict, Any, Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute reward score for video thinking trajectory.

    This function validates the format of video exploration actions and rewards
    correct answers with bonuses for effective exploration strategies.

    Args:
        data_source: Data source identifier (unused, for compatibility with reward manager)
        solution_str: Complete model response containing all <think> and <action> blocks
        ground_truth: Correct answer (usually a single token like "A", "B", "C", "D")
        extra_info: Dictionary containing:
            - question (required): The question text
            - total_frames (required): Total number of frames in the video
            - video_path (optional): Path to video file
            - fps (optional): Frames per second
            - width (optional): Video width
            - height (optional): Video height
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Dictionary with keys:
            - score: Total score (acc_score + other_score)
            - acc: Accuracy score (1.0 if correct, 0.0 otherwise)
            - format: Format score (1.0 if valid format, 0.0 otherwise)
            - other: Bonus scores (only given if answer is correct)
    """
    # Initialize scores
    format_score = 0.0
    acc_score = 0.0
    other_score = 0.0
    total_score = 0.0

    # Validate extra_info
    if extra_info is None:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    question = extra_info.get("question", "")
    total_frames = extra_info.get("total_frames", 0)

    if not question or not total_frames:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # Extract think and action blocks
    try:
        think_contents = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
        answer_contents = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    except Exception:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # Validation: must have think and action blocks
    if (not think_contents or not answer_contents):
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    model_answer = answer_contents[-1].strip()
    if not model_answer:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # All format checks passed
    format_score = 1.0

    # Compute accuracy and bonus rewards
    if model_answer == ground_truth:
        acc_score = 1.0

    total_score = acc_score

    return {
        "score": total_score,
        "acc": acc_score,
        "format": format_score,
        "other": other_score
    }
