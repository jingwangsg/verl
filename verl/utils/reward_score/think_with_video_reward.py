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
    nframes: int = 8,
    lambda_gfn: float = 0.5,
    lambda_cf: float = 0.02,
    lambda_zoom: float = 0.1,
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
        nframes: Minimum frame range size for frame selection actions
        lambda_gfn: Bonus reward for using time queries on temporal questions
        lambda_cf: Bonus reward for frame selection (exploration)
        alpha_zoom: Bonus reward for using zoom feature
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Dictionary with keys:
            - score: Total score (acc_score + other_score)
            - acc: Accuracy score (1.0 if correct, 0.0 otherwise)
            - format: Format score (1.0 if valid format, 0.0 otherwise)
            - other: Bonus scores (only given if answer is correct)

    Validation Rules:
        1. Must have matching <think> and <action> pairs
        2. Last action must be "output answer: X"
        3. No duplicate frame selections
        4. No duplicate time queries
        5. Frame ranges must be large enough (>= nframes)
        6. Numbers in actions must appear in corresponding think blocks
        7. Time queries must be followed by using the returned frame
        8. Zoom frame indices must be valid (within [0, total_frames))

    Reward Structure:
        - format_score: 1.0 if all validations pass, 0.0 otherwise (binary)
        - acc_score: 1.0 if answer matches ground truth, 0.0 otherwise (binary)
        - other_score: Sum of bonuses (only if answer is correct):
            * +lambda_gfn (default 0.5) if used time query on temporal question
            * +lambda_cf (default 0.02) if selected frames (exploration)
            * +alpha_zoom (default 0.1) if used zoom
        - total_score: acc_score + other_score

    Example:
        >>> extra_info = {
        ...     "question": "What happens at 1:30?",
        ...     "total_frames": 9000,
        ...     "video_path": "video.mp4"
        ... }
        >>> result = compute_score(
        ...     data_source="video123",
        ...     solution_str="<think>I need to check frame 2700</think>"
        ...                  "<action>zoom in frame 2700</action>"
        ...                  "<think>The answer is B</think>"
        ...                  "<action>output answer: B</action>",
        ...     ground_truth="B",
        ...     extra_info=extra_info
        ... )
        >>> result
        {'score': 1.1, 'acc': 1.0, 'format': 1.0, 'other': 0.1}
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

    # Track exploration behaviors for bonus rewards
    time_reward = False
    zoom_used = False

    # Extract think and action blocks
    try:
        think_contents = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
        action_contents = re.findall(r"<action>(.*?)</action>", solution_str, re.DOTALL)
        system_responses = re.findall(r"</action>(.*?)<think>", solution_str, re.DOTALL)
    except Exception as e:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # Validation: must have think and action blocks
    if (
        not think_contents
        or not action_contents
        or len(think_contents) != len(action_contents)
    ):
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # Validation: last action must be "output answer: X"
    last_action = action_contents[-1].strip()
    answer_match = re.match(r"output answer:\s*(\S+)", last_action)
    if not answer_match:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # Track frame selection history (to prevent duplicates)
    action_frame_pairs = [(0, total_frames - 1)]
    requested_times = []
    expected_frame_in_next_action = None

    # Validate all intermediate actions
    for i, action in enumerate(action_contents[:-1]):
        action = action.strip()
        think_block = think_contents[i]

        # If we expect a frame from previous time query, validate it's used
        if expected_frame_in_next_action is not None:
            frame_match_for_check = re.match(
                r"choose frames between (\d+) and (\d+)", action
            )
            if frame_match_for_check:
                start_f = int(frame_match_for_check.group(1))
                end_f = int(frame_match_for_check.group(2))
                if not (start_f <= expected_frame_in_next_action <= end_f):
                    return {
                        "score": total_score,
                        "acc": acc_score,
                        "format": format_score,
                        "other": other_score
                    }
            expected_frame_in_next_action = None

        # Handle "choose frames between X and Y" action
        frame_match = re.match(r"choose frames between (\d+) and (\d+)", action)
        if frame_match:
            num1 = int(frame_match.group(1))
            num2 = int(frame_match.group(2))
            current_pair = (num1, num2)
            check_pair = (num1 + 1, num2 - 1)

            # No duplicate selections
            if current_pair in action_frame_pairs:
                return {
                    "score": total_score,
                    "acc": acc_score,
                    "format": format_score,
                    "other": other_score
                }
            action_frame_pairs.append(current_pair)
            action_frame_pairs.append(check_pair)

            # Range must be large enough
            if num1 >= num2 - nframes:
                return {
                    "score": total_score,
                    "acc": acc_score,
                    "format": format_score,
                    "other": other_score
                }

            # Numbers must appear in corresponding think block
            numbers_in_think = re.findall(r"(?<!:)\b(\d+)\b(?!:)", think_block)
            if len(numbers_in_think) >= 2:
                num_a, num_b = map(int, numbers_in_think[-2:])
                if (num_a, num_b) != current_pair:
                    return {
                        "score": total_score,
                        "acc": acc_score,
                        "format": format_score,
                        "other": other_score
                    }
            else:
                return {
                    "score": total_score,
                    "acc": acc_score,
                    "format": format_score,
                    "other": other_score
                }
            continue

        # Handle "get frame number at time MM:SS" action
        time_match = re.match(r"get frame number at time\s+(\d{1,3}:\d{2})", action)
        if time_match:
            time_str_action = time_match.group(1)

            # No duplicate time queries
            if time_str_action in requested_times:
                return {
                    "score": total_score,
                    "acc": acc_score,
                    "format": format_score,
                    "other": other_score
                }
            requested_times.append(time_str_action)

            # Must have valid system response
            system_response = system_responses[i].strip()
            response_match = re.search(r"is:\s*(\d+)", system_response)
            if response_match:
                expected_frame_in_next_action = int(response_match.group(1))
                if expected_frame_in_next_action > 0:
                    time_reward = True
            else:
                return {
                    "score": total_score,
                    "acc": acc_score,
                    "format": format_score,
                    "other": other_score
                }
            continue

        # Handle "zoom in frame X" action
        zoom_match = re.match(r"zoom in frame\s+(\d+)", action)
        if zoom_match:
            frame_idx = int(zoom_match.group(1))
            if frame_idx < 0 or frame_idx >= total_frames:
                return {
                    "score": total_score,
                    "acc": acc_score,
                    "format": format_score,
                    "other": other_score
                }
            zoom_used = True
            continue

        # Unknown action format = invalid
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # Final validation: time query must be followed by frame usage
    if expected_frame_in_next_action is not None:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score
        }

    # All format checks passed
    format_score = 1.0
    model_answer = answer_match.group(1)

    # Compute accuracy and bonus rewards
    if model_answer == ground_truth:
        acc_score = 1.0

        # Bonus: time query on temporal question
        if time_reward:
            time_pattern = re.compile(r"\d+:\d+(:\d+)?")
            if re.search(time_pattern, question):
                other_score += lambda_gfn

        # Bonus: frame selection (exploration)
        if len(action_frame_pairs) > 1:
            other_score += lambda_cf

        # Bonus: zoom usage
        if zoom_used:
            other_score += lambda_zoom

    total_score = acc_score + other_score

    return {
        "score": total_score,
        "acc": acc_score,
        "format": format_score,
        "other": other_score
    }
