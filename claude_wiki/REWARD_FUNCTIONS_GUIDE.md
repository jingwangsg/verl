# Reward Functions Guide

## Overview

This guide documents the two reward functions migrated from FrameThinker-RL to the agent_rl/verl codebase:

1. **`think_with_video_reward`**: For interactive video exploration tasks
2. **`answer_reward`**: For RAG-style answer generation with retrieval documents

Both functions have been adapted for async compatibility while maintaining all original validation logic and reward calculations.

## Table of Contents

- [Quick Start](#quick-start)
- [think_with_video_reward](#think_with_video_reward)
- [answer_reward](#answer_reward)
- [Data Format Requirements](#data-format-requirements)
- [Integration Guide](#integration-guide)
- [Troubleshooting](#troubleshooting)
- [Migration Notes](#migration-notes)

---

## Quick Start

### Installation

The reward functions are located at:
```
verl/utils/reward_score/
├── think_with_video_reward.py
└── answer_reward.py
```

### Basic Usage

```python
# For video thinking tasks
from verl.utils.reward_score.think_with_video_reward import compute_score

result = compute_score(
    predict_str=model_output,
    ground_truth="B",
    extra_info={
        "question": "What happens at 1:30?",
        "total_frames": 9000,
        "video_path": "video.mp4"
    },
    nframes=8,
    lambda_gfn=0.5,
    lambda_cf=0.02,
    alpha_zoom=0.1
)
# Returns: {"score": 1.62, "acc": 1.0, "format": 1.0, "other": 0.62}

# For answer generation tasks
from verl.utils.reward_score.answer_reward import compute_score

result = compute_score(
    predict_str=model_output,
    ground_truth="Paris"
)
# Returns: {"score": 3.0, "format": 0.0, "retrieval": 1.0, "acc": 2.0, "f1": 1.0, ...}
```

---

## think_with_video_reward

### Purpose

Validates and scores agent trajectories that explore videos through interactive actions before outputting a final answer.

### Location

`verl/utils/reward_score/think_with_video_reward.py`

### Function Signature

```python
def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    nframes: int = 8,
    lambda_gfn: float = 0.5,
    lambda_cf: float = 0.02,
    alpha_zoom: float = 0.1,
    **kwargs
) -> Dict[str, float]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predict_str` | str | required | Complete model response with all `<think>` and `<action>` blocks |
| `ground_truth` | str | required | Correct answer (usually single token like "A", "B", "C", "D") |
| `extra_info` | Dict | None | Dictionary containing `question` (required) and `total_frames` (required) |
| `nframes` | int | 8 | Minimum frame range size for frame selection actions |
| `lambda_gfn` | float | 0.5 | Bonus reward for using time queries on temporal questions |
| `lambda_cf` | float | 0.02 | Bonus reward for frame selection (exploration) |
| `alpha_zoom` | float | 0.1 | Bonus reward for using zoom feature |

### Return Value

Returns a dictionary with the following keys:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `score` | float | [0.0, 1.62] | Total score (acc + other) |
| `acc` | float | {0.0, 1.0} | Accuracy score (binary) |
| `format` | float | {0.0, 1.0} | Format validation score (binary) |
| `other` | float | [0.0, 0.62] | Sum of bonus rewards |

### Reward Structure

#### Format Score (Binary)
- **1.0**: All validation rules pass
- **0.0**: Any validation rule fails

#### Accuracy Score (Binary)
- **1.0**: Model answer matches ground truth (exact match)
- **0.0**: Model answer does not match

#### Bonus Scores (Cumulative, only if answer is correct)
- **Time Query Bonus** (`lambda_gfn`): Awarded if:
  - Used "get frame number at time MM:SS" action
  - Question contains time pattern (e.g., "1:30", "02:45:10")
- **Frame Selection Bonus** (`lambda_cf`): Awarded if:
  - Used "choose frames between X and Y" action at least once
- **Zoom Bonus** (`alpha_zoom`): Awarded if:
  - Used "zoom in frame X" action at least once

#### Total Score Calculation
```python
total_score = acc_score + other_score
# where other_score is only added if acc_score == 1.0
```

### Validation Rules

The format score will be 0.0 if ANY of the following conditions fail:

1. ✅ Must have matching `<think>` and `<action>` pairs
2. ✅ Number of think blocks must equal number of action blocks
3. ✅ Last action must be `output answer: X` where X matches the pattern
4. ✅ No duplicate frame selections (same start/end pair)
5. ✅ No duplicate time queries (same timestamp)
6. ✅ Frame ranges must be large enough: `end - start >= nframes`
7. ✅ Numbers in actions must appear in corresponding think blocks
8. ✅ Time queries must be followed by using the returned frame
9. ✅ Zoom frame indices must be valid: `0 <= frame_idx < total_frames`
10. ✅ All actions must match one of the three supported formats

### Supported Action Formats

All actions are validated using regex patterns:

#### 1. Get Frame Number
```
Format: get frame number at time MM:SS
Regex: r"get frame number at time\s+(\d{1,3}:\d{2})"
Example: get frame number at time 01:30
```

#### 2. Zoom In Frame
```
Format: zoom in frame <FRAME_INDEX>
Regex: r"zoom in frame\s+(\d+)"
Example: zoom in frame 2700
```

#### 3. Choose Frame Range
```
Format: choose frames between <START> and <END>
Regex: r"choose frames between (\d+) and (\d+)"
Example: choose frames between 100 and 500
```

### Example Trajectories

#### Perfect Score (1.62)

```python
predict_str = """<think>I need to check what happens at 1:30 in the video.</think>
<action>get frame number at time 01:30</action>
The frame number at time 01:30 is: 2700
<think>Let me zoom in on frame 2700 to see details.</think>
<action>zoom in frame 2700</action>
<think>I can see the person is dancing. Let me explore nearby frames.</think>
<action>choose frames between 2650 and 2750</action>
<think>Based on the frames, the answer is B.</think>
<action>output answer: B</action>"""

ground_truth = "B"
extra_info = {
    "question": "What is the person doing at 1:30?",
    "total_frames": 9000
}

result = compute_score(predict_str, ground_truth, extra_info)
# Result: {
#     "score": 1.62,
#     "acc": 1.0,
#     "format": 1.0,
#     "other": 0.62  # 0.5 (time) + 0.02 (frame) + 0.1 (zoom)
# }
```

#### Correct Answer, Basic Exploration (1.02)

```python
predict_str = """<think>Let me analyze the video frames.</think>
<action>choose frames between 0 and 100</action>
<think>I can see a red car. The answer is C.</think>
<action>output answer: C</action>"""

ground_truth = "C"
extra_info = {
    "question": "What color is the car?",
    "total_frames": 300
}

result = compute_score(predict_str, ground_truth, extra_info)
# Result: {
#     "score": 1.02,
#     "acc": 1.0,
#     "format": 1.0,
#     "other": 0.02  # Only frame selection bonus
# }
```

#### Format Error (0.0)

```python
predict_str = """<think>Let me check a small range.</think>
<action>choose frames between 0 and 5</action>
<think>The answer is A.</think>
<action>output answer: A</action>"""

ground_truth = "A"
extra_info = {
    "question": "What is shown?",
    "total_frames": 100
}

result = compute_score(predict_str, ground_truth, extra_info, nframes=8)
# Result: {
#     "score": 0.0,
#     "acc": 0.0,
#     "format": 0.0,  # Frame range too small (5 - 0 = 5 < 8)
#     "other": 0.0
# }
```

#### Wrong Answer (0.0)

```python
predict_str = """<think>I think it's A.</think>
<action>choose frames between 0 and 100</action>
<think>Yes, definitely A.</think>
<action>output answer: A</action>"""

ground_truth = "B"
extra_info = {
    "question": "What is the answer?",
    "total_frames": 300
}

result = compute_score(predict_str, ground_truth, extra_info)
# Result: {
#     "score": 0.0,
#     "acc": 0.0,  # Wrong answer, no bonuses
#     "format": 1.0,
#     "other": 0.0
# }
```

### Configuration Example

```yaml
custom_reward_function:
  path: verl/utils/reward_score/think_with_video_reward.py
  name: compute_score
  reward_kwargs:
    nframes: 8
    lambda_gfn: 0.5
    lambda_cf: 0.02
    alpha_zoom: 0.1
```

See `verl/configs/reward/video_think_reward_example.yaml` for detailed configuration.

---

## answer_reward

### Purpose

Validates and scores RAG-style responses that use retrieval documents and `<think>`/`<answer>` tags for structured reasoning.

### Location

`verl/utils/reward_score/answer_reward.py`

### Function Signature

```python
def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, float]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predict_str` | str | required | Complete model response (will be prepended with `<think>`) |
| `ground_truth` | str | required | Correct answer text |
| `extra_info` | Dict | None | Optional dictionary (not used, for compatibility) |

**Important**: The function automatically prepends `<think>` to `predict_str`, so your input should NOT start with `<think>`.

### Return Value

Returns a dictionary with the following keys:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `score` | float | [-2.0, 3.0] | Total score (format + retrieval + acc) |
| `format` | float | {-1.0, 0.0} | Format validation penalty |
| `retrieval` | float | {-1.0, 1.0} | Retrieval usage reward/penalty |
| `acc` | float | [0.0, 2.0] | Accuracy reward (2.0 * F1) |
| `f1` | float | [0.0, 1.0] | F1 score |
| `precision` | float | [0.0, 1.0] | Precision score |
| `recall` | float | [0.0, 1.0] | Recall score |

### Reward Structure

#### Format Reward
- **0.0**: All validation rules pass
- **-1.0**: Any validation rule fails (harsh penalty)

#### Retrieval Reward
- **+1.0**: Documents contain at least one indexed item (e.g., "(1)")
- **-1.0**: No indexed documents found

#### Accuracy Reward
- **Calculation**: `2.0 * F1_score`
- **Range**: [0.0, 2.0]
- **Method**: Token-level F1 score between normalized prediction and ground truth

#### Total Score Calculation
```python
total_score = format_reward + retrieval_reward + acc_reward
# Range: [-2.0, 3.0]
# Best case: 0.0 + 1.0 + 2.0 = 3.0
# Worst case: -1.0 + (-1.0) + 0.0 = -2.0
```

### Format Requirements

All of the following must be satisfied for `format_reward = 0.0`:

1. ✅ Must have matching `<|begin_of_documents|>\n` ... `<|end_of_documents|>\n`
2. ✅ Must have matching `<|begin_of_query|>` ... `<|end_of_query|>`
3. ✅ Must have matching `<think>` ... `</think>` tags
4. ✅ Must have exactly one `<answer>` ... `</answer>` pair
5. ✅ Answer must not contain "begin_of_query" or "begin_of_documents"
6. ✅ Answer must be ≤ 10 words
7. ✅ Must not contain "Assistant:" or "assistant:"
8. ✅ Must use retrieval: at least one document with index like "(1)"

### F1 Score Calculation

#### Normalization Steps
Both prediction and ground truth undergo the following transformations:

1. **Boolean Mapping**: "True" → "yes", "False" → "no"
2. **Lowercase**: Convert to lowercase
3. **Remove Articles**: Remove "a", "an", "the"
4. **Remove Punctuation**: Remove all punctuation marks
5. **Replace Underscores**: Replace "_" with " "
6. **Fix Whitespace**: Normalize whitespace

#### Special Handling
For yes/no/noanswer responses:
- Must match exactly after normalization
- Returns (0, 0, 0) if mismatch

#### Token-Level Calculation
```python
# Tokenize by whitespace
prediction_tokens = normalized_prediction.split()
ground_truth_tokens = normalized_ground_truth.split()

# Count overlapping tokens (multiset intersection)
common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
num_same = sum(common.values())

# Calculate metrics
precision = num_same / len(prediction_tokens)
recall = num_same / len(ground_truth_tokens)
f1 = (2.0 * precision * recall) / (precision + recall)
```

### Example Trajectories

#### Perfect Score (3.0)

```python
predict_str = """<|begin_of_documents|>
(1) The capital of France is Paris.
(2) Paris is known for the Eiffel Tower.
<|end_of_documents|>
<|begin_of_query|>
What is the capital of France?
<|end_of_query|>
Based on document 1, the capital is Paris.
</think>
<answer>Paris</answer>"""

ground_truth = "Paris"

result = compute_score(predict_str, ground_truth)
# Result: {
#     "score": 3.0,
#     "format": 0.0,
#     "retrieval": 1.0,
#     "acc": 2.0,
#     "f1": 1.0,
#     "precision": 1.0,
#     "recall": 1.0
# }
```

#### Partial Match (2.33)

```python
predict_str = """<|begin_of_documents|>
(1) The Eiffel Tower is 330 meters tall.
<|end_of_documents|>
<|begin_of_query|>
How tall is the Eiffel Tower?
<|end_of_query|>
According to the document, it's 330 meters.
</think>
<answer>330 meters tall structure</answer>"""

ground_truth = "330 meters"

result = compute_score(predict_str, ground_truth)
# Result: {
#     "score": 2.33,  # 0.0 + 1.0 + 1.33
#     "format": 0.0,
#     "retrieval": 1.0,
#     "acc": 1.33,  # 2.0 * 0.67
#     "f1": 0.67,   # F1 calculation
#     "precision": 0.67,  # 2 of 3 tokens match
#     "recall": 1.0       # 2 of 2 tokens in ground truth match
# }
```

#### Format Error - Long Answer (-1.0)

```python
predict_str = """<|begin_of_documents|>
(1) Some context here.
<|end_of_documents|>
<|begin_of_query|>
What is the answer?
<|end_of_query|>
Let me think about this carefully.
</think>
<answer>This is a very long answer that exceeds the ten word limit clearly</answer>"""

ground_truth = "short answer"

result = compute_score(predict_str, ground_truth)
# Result: {
#     "score": 0.0,  # Capped due to format error
#     "format": -1.0,  # Answer too long (> 10 words)
#     "retrieval": 1.0,
#     "acc": 0.0,
#     "f1": 0.0,
#     "precision": 0.0,
#     "recall": 0.0
# }
```

#### No Retrieval Usage (1.0)

```python
predict_str = """<|begin_of_documents|>
Some text without index numbers.
<|end_of_documents|>
<|begin_of_query|>
What is X?
<|end_of_query|>
I think the answer is Y.
</think>
<answer>Y</answer>"""

ground_truth = "Y"

result = compute_score(predict_str, ground_truth)
# Result: {
#     "score": 1.0,  # 0.0 + (-1.0) + 2.0
#     "format": 0.0,
#     "retrieval": -1.0,  # No indexed documents
#     "acc": 2.0,
#     "f1": 1.0,
#     "precision": 1.0,
#     "recall": 1.0
# }
```

#### Yes/No Question - Correct (3.0)

```python
predict_str = """<|begin_of_documents|>
(1) The sky is blue.
<|end_of_documents|>
<|begin_of_query|>
Is the sky blue?
<|end_of_query|>
Yes, according to the document.
</think>
<answer>yes</answer>"""

ground_truth = "True"  # Will be mapped to "yes"

result = compute_score(predict_str, ground_truth)
# Result: {
#     "score": 3.0,
#     "format": 0.0,
#     "retrieval": 1.0,
#     "acc": 2.0,
#     "f1": 1.0,
#     "precision": 1.0,
#     "recall": 1.0
# }
```

#### Yes/No Question - Wrong (0.0)

```python
predict_str = """<|begin_of_documents|>
(1) Context.
<|end_of_documents|>
<|begin_of_query|>
Question?
<|end_of_query|>
I think no.
</think>
<answer>no</answer>"""

ground_truth = "yes"

result = compute_score(predict_str, ground_truth)
# Result: {
#     "score": 0.0,  # 0.0 + 1.0 + (-1.0)
#     "format": 0.0,
#     "retrieval": 1.0,
#     "acc": 0.0,  # Yes/no mismatch returns F1=0
#     "f1": 0.0,
#     "precision": 0.0,
#     "recall": 0.0
# }
```

### Configuration Example

```yaml
custom_reward_function:
  path: verl/utils/reward_score/answer_reward.py
  name: compute_score
  # No reward_kwargs needed for this function
```

See `verl/configs/reward/answer_reward_example.yaml` for detailed configuration.

---

## Data Format Requirements

### think_with_video_reward

Your parquet files should contain the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `predict_str` | str | Yes | Complete model response with all `<think>` and `<action>` blocks |
| `ground_truth` | str | Yes | Correct answer (usually single token) |
| `extra_info` | Dict | Yes | Dictionary with required keys below |

**Required keys in `extra_info`**:
- `question` (str): The question text
- `total_frames` (int): Total number of frames in the video

**Optional keys in `extra_info`**:
- `video_path` (str): Path to video file
- `fps` (int): Frames per second
- `width` (int): Video width
- `height` (int): Video height

**Example**:
```python
{
    "predict_str": "<think>...</think><action>...</action>...",
    "ground_truth": "B",
    "extra_info": {
        "question": "What happens at 1:30?",
        "total_frames": 9000,
        "video_path": "video.mp4",
        "fps": 30,
        "width": 1920,
        "height": 1080
    }
}
```

### answer_reward

Your parquet files should contain the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `predict_str` | str | Yes | Complete model response (without leading `<think>`) |
| `ground_truth` | str | Yes | Correct answer text |
| `extra_info` | Dict | No | Optional dictionary (not used by function) |

**Important**: Do NOT include the leading `<think>` tag in `predict_str` - the function adds it automatically.

**Example**:
```python
{
    "predict_str": "<|begin_of_documents|>\n(1) ...\n<|end_of_documents|>\n<|begin_of_query|>...<|end_of_query|>...\n</think>\n<answer>...</answer>",
    "ground_truth": "Paris",
    "extra_info": {}  # Optional, not used
}
```

---

## Integration Guide

### Step 1: Register Reward Function

Add to your training configuration YAML:

```yaml
# For video thinking tasks
custom_reward_function:
  path: verl/utils/reward_score/think_with_video_reward.py
  name: compute_score
  reward_kwargs:
    nframes: 8
    lambda_gfn: 0.5
    lambda_cf: 0.02
    alpha_zoom: 0.1

# OR for answer generation tasks
custom_reward_function:
  path: verl/utils/reward_score/answer_reward.py
  name: compute_score
```

### Step 2: Prepare Data

Ensure your parquet files contain the required columns (see [Data Format Requirements](#data-format-requirements)).

### Step 3: Validate Data Format

Use the validation script to check your data:

```bash
python verl/utils/reward_score/validate_reward_data_format.py \
    --data_path /path/to/data.parquet \
    --reward_type video_think  # or "answer"
```

### Step 4: Run Training

The reward manager will automatically load and use your custom reward function:

```bash
python your_training_script.py --config your_config.yaml
```

### Step 5: Monitor Metrics

Track the following metrics during training:

**For video thinking tasks**:
- `reward/score`: Total score
- `reward/acc`: Accuracy rate
- `reward/format`: Format validation rate
- `reward/other`: Average bonus rewards

**For answer generation tasks**:
- `reward/score`: Total score
- `reward/format`: Format validation rate
- `reward/retrieval`: Retrieval usage rate
- `reward/acc`: Average accuracy reward
- `reward/f1`: Average F1 score

---

## Troubleshooting

### think_with_video_reward

#### Low Format Scores

**Symptoms**: `format` score consistently 0.0

**Possible Causes**:
1. Model not generating matching `<think>`/`<action>` pairs
2. Missing or malformed final action (`output answer: X`)
3. Duplicate frame selections or time queries
4. Frame ranges too small (< nframes)
5. Numbers in actions not appearing in think blocks

**Solutions**:
1. Check model output format manually
2. Verify training data has correct format
3. Review validation error messages in logs
4. Consider adjusting `nframes` parameter if too strict

#### Low Bonus Scores

**Symptoms**: `other` score always 0.0 even with correct answers

**Possible Causes**:
1. Model not using advanced actions (time query, zoom)
2. Time queries used but question doesn't contain time patterns
3. Bonus weights too low

**Solutions**:
1. Check if model generates time query/zoom actions
2. Increase bonus weights (`lambda_gfn`, `lambda_cf`, `alpha_zoom`)
3. Verify questions contain time patterns for time query bonus
4. Review model's action distribution

#### High Variance in Scores

**Symptoms**: Scores fluctuate significantly between batches

**Possible Causes**:
1. Format errors causing binary 0/1 scores
2. Validation rules too strict for diverse data
3. Ground truth format inconsistencies

**Solutions**:
1. Monitor `format` score separately
2. Review and standardize ground truth format
3. Consider adjusting validation thresholds
4. Check for data quality issues

### answer_reward

#### Low Format Scores

**Symptoms**: `format` score consistently -1.0

**Possible Causes**:
1. Missing or mismatched document/query/think/answer tags
2. Documents not using indexed format "(1)", "(2)", etc.
3. Answers exceeding 10 words
4. Presence of "Assistant:" prefix

**Solutions**:
1. Verify all required tags are present and properly closed
2. Ensure documents use indexed format
3. Encourage shorter, more concise answers
4. Remove any "Assistant:" prefixes from training data

#### Low Retrieval Scores

**Symptoms**: `retrieval` score consistently -1.0

**Possible Causes**:
1. Documents missing index numbers like "(1)"
2. Document tags not ending with newline
3. Model not including retrieval documents in response

**Solutions**:
1. Check document format: must have "(1)", "(2)", etc.
2. Ensure tags end with newline: `<|begin_of_documents|>\n`
3. Verify model is trained to use retrieval format
4. Review training data for correct format

#### Low F1 Scores

**Symptoms**: `f1` score consistently low even for seemingly correct answers

**Possible Causes**:
1. Answers too verbose (extra words reducing precision)
2. Normalization differences between prediction and ground truth
3. Token mismatch due to formatting

**Solutions**:
1. Encourage shorter, more direct answers
2. Normalize ground truth consistently
3. Check for punctuation/case mismatches
4. Review F1 calculation logic for your use case

#### Negative Total Scores

**Symptoms**: `score` is negative

**Possible Causes**:
1. Format error (-1.0) combined with no retrieval (-1.0)
2. Low or zero F1 score

**Solutions**:
1. Fix format issues first (highest priority)
2. Ensure retrieval documents are indexed
3. Improve answer quality to boost F1 score

---

## Migration Notes

### Changes from Original Implementation

Both reward functions have been migrated from FrameThinker-RL with the following changes:

#### Return Format
- **Original**: Returned tuple `(score, acc, format, other)`
- **New**: Returns dict `{"score": ..., "acc": ..., "format": ..., "other": ...}`
- **Reason**: Better compatibility with async reward manager and provides clearer metric names

#### Async Compatibility
- **Original**: Synchronous functions
- **New**: Still synchronous but will be wrapped by async reward manager
- **Reason**: Reward calculations are CPU-bound and don't benefit from async

#### Documentation
- **Original**: Minimal docstrings
- **New**: Comprehensive docstrings with examples and type hints
- **Reason**: Improved maintainability and ease of use

### Validation Logic

All original validation logic has been **preserved exactly**:
- ✅ All format checks remain identical
- ✅ Reward calculations unchanged
- ✅ Bonus reward conditions unchanged
- ✅ F1 score calculation unchanged

### Backward Compatibility

To convert from old tuple format to new dict format in existing code:

```python
# Old code
score, acc, format_score, other = compute_score(...)

# New code
result = compute_score(...)
score = result["score"]
acc = result["acc"]
format_score = result["format"]
other = result["other"]
```

---

## Comparison: think_with_video_reward vs answer_reward

| Aspect | think_with_video_reward | answer_reward |
|--------|------------------------|---------------|
| **Use Case** | Video exploration with interactive actions | RAG-style document Q&A |
| **Reward Range** | [0.0, 1.62] | [-2.0, 3.0] |
| **Format Penalty** | 0.0 (no negative penalty) | -1.0 (harsh penalty) |
| **Accuracy Metric** | Binary exact match | Continuous F1 score |
| **Bonus Rewards** | Time query, frame selection, zoom | Retrieval usage |
| **Extra Info Required** | Yes (question, total_frames) | No (optional) |
| **Answer Length Limit** | None | 10 words |
| **Multi-turn** | Yes (multiple actions) | No (single answer) |

---

## Best Practices

### For think_with_video_reward

1. **Balance Exploration**: Adjust `lambda_cf` to encourage frame selection without over-rewarding
2. **Temporal Reasoning**: Use higher `lambda_gfn` for time-sensitive questions
3. **Format Enforcement**: Start with strict validation, relax if needed
4. **Monitor Bonuses**: Track which bonuses are being awarded to guide training
5. **Data Quality**: Ensure `extra_info` always includes required fields

### For answer_reward

1. **Short Answers**: Encourage concise answers (≤10 words)
2. **Use Retrieval**: Always include indexed documents "(1)", "(2)"
3. **Normalize Consistently**: Ensure ground truth follows same normalization as predictions
4. **Yes/No Questions**: Use "yes"/"no" or "True"/"False" consistently
5. **Monitor F1**: Track F1 separately from total score to identify answer quality issues

---

## Additional Resources

- **Configuration Examples**:
  - `verl/configs/reward/video_think_reward_example.yaml`
  - `verl/configs/reward/answer_reward_example.yaml`

- **Source Code**:
  - `verl/utils/reward_score/think_with_video_reward.py`
  - `verl/utils/reward_score/answer_reward.py`

- **Validation Script**:
  - `verl/utils/reward_score/validate_reward_data_format.py`

- **Related Documentation**:
  - `verl/IMPLEMENTATION_SUMMARY.md`: VideoThinkTool implementation
  - `verl/COMPATIBILITY_NOTES.md`: Tool compatibility notes

---

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review configuration examples in `verl/configs/reward/`
3. Run data validation script to check format
4. Check logs for detailed error messages
5. Compare your data format with examples in this guide

For questions or bug reports, please contact the maintainers or file an issue.

---

**Last Updated**: 2025-11-22
