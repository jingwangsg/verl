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

    def _parse_answer(text: str) -> Dict[str, str]:
        """
        Extract raw answer text and a leading choice token (first alnum) from free-form content.

        - If <answer>...</answer> exists, parse inside; otherwise use the full text.
        - Leading choice token is the first alphanumeric character (letter uppercased).
        Returns {"raw": raw_text, "choice": leading_alnum_or_empty}.
        """
        if not text:
            return {"raw": "", "choice": ""}

        # Highest priority: XML-style tags
        tagged = re.findall(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        if tagged:
            text = tagged[-1].strip()

        # Common textual patterns (capture first alphanumeric)
        patterns = [
            r"final answer\s*[:\-]\s*([A-Za-z0-9])",
            r"answer\s*[:\-]\s*([A-Za-z0-9])",
            r"^([A-Za-z0-9])[\).]",          # "B) ..." or "B. ..."
            r"^\s*([A-Za-z0-9])\s*$",        # single-letter/number line
            r"\boption\s*([A-Za-z0-9])\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                ch = m.group(1)
                choice = ch.upper() if ch.isalpha() else ch
                return {"raw": text.strip(), "choice": choice}

        # Fallback: first alphanumeric character anywhere
        m = re.search(r"[A-Za-z0-9]", text)
        if m:
            ch = m.group(0)
            choice = ch.upper() if ch.isalpha() else ch
            return {"raw": text.strip(), "choice": choice}

        return {"raw": text.strip(), "choice": ""}

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
    model_parsed = _parse_answer(solution_str)
    gold_parsed = _parse_answer(ground_truth)

    model_raw = model_parsed["raw"]
    gold_raw = gold_parsed["raw"] or ground_truth.strip()

    # Mark format as valid if we could read any non-empty answer text
    if model_raw:
        format_score = 1.0

    def _norm(s: str) -> str:
        # Lowercase, strip surrounding punctuation and collapse spaces for text match
        s = s.strip().lower()
        s = re.sub(r"^[\\s\\.,;:!?]+|[\\s\\.,;:!?]+$", "", s)
        s = re.sub(r"\\s+", " ", s)
        return s

    # Compute accuracy: prefer leading choice token if both have one; otherwise compare normalized text
    if model_raw and gold_raw:
        if model_parsed["choice"] and gold_parsed["choice"]:
            if model_parsed["choice"] == gold_parsed["choice"]:
                acc_score = 1.0
        else:
            if _norm(model_raw) == _norm(gold_raw):
                acc_score = 1.0

    total_score = acc_score  # no extra bonuses for vanilla MCQ scoring

    return {
        "score": total_score,
        "acc": acc_score,
        "format": format_score,
        "other": other_score,
    }
