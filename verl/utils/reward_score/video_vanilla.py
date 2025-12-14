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
Lightweight reward for vanilla multiple-choice video QA.

Unlike `think_with_video_reward.py`, this scorer is designed for plain MCQ
outputs (typically just "A"/"B"/"C"/"D" or "B. short explanation") without
tool calls or structured <action>/<think> blocks. It simply tries to extract
the chosen option robustly and compares it with the ground truth.
"""

import re
from typing import Dict, Any, Optional
from debug.snapshot import Snapshot
from uuid import uuid4


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
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
    _snap_id = str(uuid4())[:8]
    _snp = lambda x: Snapshot(
        f"reward_score/video_vanilla/compute_score/{_snap_id}/{x[0]}",
    ).snapshot(x[1])

    _snp(("extra_info", extra_info))
    _snp(("solution_str", solution_str))
    _snp(("ground_truth", ground_truth))

    def _extract_tokens(text: str) -> Dict[str, Any]:
        """
        Extract answer-bearing tokens for MCQ judging.

        Steps:
          1) If <answer>...</answer> exists, only use the inner content; else use full text.
          2) Split on non-alphanumeric chars; keep single-char uppercase letters as candidate choices.
        Returns {"raw": raw_text_used, "choices": [letters], "has_tag": bool}.
        """
        raw_source = text or ""
        tagged = re.findall(r"<answer>(.*?)</answer>", raw_source, re.IGNORECASE | re.DOTALL)
        has_tag = bool(tagged)
        if tagged:
            raw = tagged[-1].strip()
        else:
            raw = raw_source.strip()

        # Split on any non-alphanumeric
        parts = re.split(r"[^A-Za-z0-9]+", raw)
        choices = [p.upper() for p in parts if len(p) == 1 and p.isalpha()]
        return {"raw": raw, "choices": choices, "has_tag": has_tag}

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
            "other": other_score,
        }

    question = extra_info.get("question", "")
    total_frames = extra_info.get("total_frames", 0)

    if not question or not total_frames:
        return {
            "score": total_score,
            "acc": acc_score,
            "format": format_score,
            "other": other_score,
        }

    # Extract predicted and gold answers (robust to different formats)
    pred = _extract_tokens(solution_str)
    gold = _extract_tokens(ground_truth)

    model_raw = pred["raw"]
    gold_choices = gold["choices"] or [ground_truth.strip().upper()]

    # Format: we parsed at least one single-letter choice
    if pred["choices"]:
        format_score = 1.0

    def _norm(s: str) -> str:
        # Lowercase, strip surrounding punctuation and collapse spaces for text match
        s = s.strip().lower()
        s = re.sub(r"^[\\s\\.,;:!?]+|[\\s\\.,;:!?]+$", "", s)
        s = re.sub(r"\\s+", " ", s)
        return s

    # Compute accuracy: any predicted choice matching any gold choice
    if format_score == 1.0:
        for c in pred["choices"]:
            if c in gold_choices:
                acc_score = 1.0
                break

    total_score = acc_score  # no extra bonuses for vanilla MCQ scoring

    return {
        "score": total_score,
        "acc": acc_score,
        "format": format_score,
        "other": other_score,
    }
