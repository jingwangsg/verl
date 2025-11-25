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
Reward function for answer-based RAG (Retrieval-Augmented Generation) tasks.

This reward function validates and scores responses that use retrieval documents
and <think>/<answer> tags for structured reasoning.

Migrated from FrameThinker-RL with modifications:
- Changed return format from float to dict for detailed metrics
- Added comprehensive docstrings
- Maintained all validation logic
"""

import re
import string
from collections import Counter
from typing import Dict, Tuple, Optional, Any


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison.

    Applies the following transformations:
    1. Convert to lowercase
    2. Remove articles (a, an, the)
    3. Remove punctuation
    4. Replace underscores with spaces
    5. Fix whitespace

    Args:
        s: Input string to normalize

    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s: str) -> str:
    """
    Map boolean strings to yes/no format.

    Args:
        s: Input string

    Returns:
        "yes" if s=="True", "no" if s=="False", otherwise unchanged
    """
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate token-level F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer text
        ground_truth: Ground truth answer text

    Returns:
        Tuple of (f1, precision, recall). Returns (0, 0, 0) for no match.

    Example:
        >>> f1_score("the cat sat", "a cat sat")
        (0.8, 0.666..., 1.0)
    """
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    # Special handling for yes/no/noanswer responses
    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    # Tokenize and compute overlap
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)

    return f1, precision, recall


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute reward score for RAG-style answer generation.

    This function validates the format of retrieval-augmented responses and
    scores answers using token-level F1 score.

    Args:
        predict_str: Complete model response (will be prepended with '<think>')
        ground_truth: Correct answer text
        extra_info: Optional dictionary (not used in this function, for compatibility)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Dictionary with keys:
            - score: Total score (format + retrieval + acc)
            - format: Format reward (-1.0 if error, 0.0 if valid)
            - retrieval: Retrieval reward (1.0 if used, -1.0 if not)
            - acc: Accuracy reward (2.0 * F1 score)
            - f1: F1 score (0.0 to 1.0)
            - precision: Precision score (0.0 to 1.0)
            - recall: Recall score (0.0 to 1.0)

    Format Requirements:
        1. Must have matching document tags: <|begin_of_documents|> ... <|end_of_documents|>
        2. Must have matching query tags: <|begin_of_query|> ... <|end_of_query|>
        3. Must have matching <think> ... </think> tags
        4. Must have exactly one <answer> ... </answer> pair
        5. Answer must not contain special tags (begin_of_query, begin_of_documents)
        6. Answer must be ≤ 10 words
        7. Must not contain "Assistant:" or "assistant:"
        8. Must use retrieval (at least one document with index)

    Reward Structure:
        - format_reward: -1.0 if format error, 0.0 otherwise
        - retrieval_reward: +1.0 if used retrieval, -1.0 otherwise
        - acc_reward: 2.0 * F1_score (0.0 to 2.0 range)
        - total_score: format + retrieval + accuracy

    Example:
        >>> result = compute_score(
        ...     predict_str="<|begin_of_documents|>\\n(1) Some context\\n<|end_of_documents|>"
        ...                "<|begin_of_query|>What?<|end_of_query|>"
        ...                "<think>Let me think</think>"
        ...                "<answer>The answer</answer>",
        ...     ground_truth="The answer"
        ... )
        >>> result
        {'score': 3.0, 'format': 0.0, 'retrieval': 1.0, 'acc': 2.0, 'f1': 1.0, ...}
    """
    is_format_error = False

    # Prepend <think> if not present
    predict_str = '<think>' + predict_str

    # Validate document/query format
    count_1 = predict_str.count("<|begin_of_documents|>\n")
    count_2 = predict_str.count("<|end_of_documents|>\n")
    count_3 = predict_str.count("<|begin_of_query|>")
    count_4 = predict_str.count("<|end_of_query|>")
    count_5 = predict_str.count("<|begin_of_documents|>")
    count_6 = predict_str.count("<|end_of_documents|>")
    count_7 = predict_str.count("<|begin_of_documents|>\n(1)")  # Document with index

    if not (count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7):
        is_format_error = True

    # Check for forbidden "Assistant:" prefix
    count_assistant_1 = predict_str.count("Assistant:")
    count_assistant_2 = predict_str.count("assistant:")
    if count_assistant_1 != 0 or count_assistant_2 != 0:
        is_format_error = True

    # Validate think tags
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    # Validate answer tags (exactly one)
    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != 1 or count_answer_2 != 1:
        is_format_error = True

    # Extract answer text
    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # Answer must not contain special tags
    if "begin_of_query" in answer_text or "begin_of_documents" in answer_text:
        is_format_error = True

    # Answer must be short (≤10 words)
    answer_len = len(answer_text.split())
    if answer_len > 10:
        is_format_error = True

    # Compute rewards
    retrieval_reward = 1.0 if count_7 >= 1 else -1.0

    # F1 score for answer accuracy
    f1, precision, recall = f1_score(answer_text, ground_truth)
    acc_reward = 2.0 * f1

    # Format penalty
    format_reward = -1.0 if is_format_error else 0.0

    # Total score
    total_score = format_reward + retrieval_reward + acc_reward

    return {
        "score": total_score,
        "format": format_reward,
        "retrieval": retrieval_reward,
        "acc": acc_reward,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
