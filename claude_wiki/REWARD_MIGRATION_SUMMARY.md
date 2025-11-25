# Reward Functions Migration Summary

## Overview

Successfully migrated two reward functions from FrameThinker-RL to agent_rl/verl codebase, adapting them for async compatibility while maintaining all original validation logic.

## Migration Date
2025-11-22

## Files Created

### Reward Function Implementations

1. **`verl/utils/reward_score/think_with_video_reward.py`** (341 lines)
   - Validates and scores video exploration trajectories
   - Supports three action types: frame selection, time queries, zoom
   - Binary format/accuracy scores + cumulative bonus rewards
   - Return format: `Dict[str, float]` with keys: score, acc, format, other

2. **`verl/utils/reward_score/answer_reward.py`** (259 lines)
   - Validates and scores RAG-style answer generation
   - Includes F1 score calculation with normalization
   - Format penalty + retrieval reward + accuracy reward
   - Return format: `Dict[str, float]` with keys: score, format, retrieval, acc, f1, precision, recall

### Configuration Examples

3. **`verl/configs/reward/video_think_reward_example.yaml`**
   - Complete configuration with all parameters documented
   - Includes reward structure explanation
   - Example trajectories with expected scores
   - Tips and troubleshooting guidance

4. **`verl/configs/reward/answer_reward_example.yaml`**
   - Configuration for answer reward function
   - Format requirements documentation
   - F1 score calculation details
   - Comparison with video_think reward

### Documentation

5. **`verl/REWARD_FUNCTIONS_GUIDE.md`** (800+ lines)
   - Comprehensive guide for both reward functions
   - Detailed reward structure explanations
   - Data format requirements
   - Integration guide with step-by-step instructions
   - Troubleshooting section for common issues
   - Multiple example trajectories with expected outputs

### Tools

6. **`verl/utils/reward_score/validate_reward_data_format.py`** (450+ lines)
   - Validates parquet files for both reward types
   - Checks required columns and data types
   - Validates extra_info structure for video tasks
   - Provides detailed error messages and statistics
   - Supports sample limits and report generation

## Key Changes from Original

### Return Format
- **Before**: Tuple `(score, acc, format, other)` or `(score, acc, format, ...)`
- **After**: Dictionary with named keys for clarity and async compatibility

### Documentation
- **Before**: Minimal docstrings
- **After**: Comprehensive docstrings with examples, type hints, and detailed explanations

### Compatibility
- **Maintained**: All validation logic, reward calculations, and bonus conditions
- **Changed**: Only return format for async compatibility

## Reward Function Details

### think_with_video_reward

**Purpose**: Score interactive video exploration trajectories

**Reward Range**: [0.0, 1.62]

**Components**:
- Format score: 1.0 (valid) or 0.0 (invalid)
- Accuracy score: 1.0 (correct) or 0.0 (wrong)
- Bonus rewards (if correct):
  - Time query: +0.5 (default)
  - Frame selection: +0.02 (default)
  - Zoom usage: +0.1 (default)

**Required Data**:
- `predict_str`: Complete trajectory with `<think>` and `<action>` blocks
- `ground_truth`: Correct answer (usually single token)
- `extra_info`: Dict with `question` and `total_frames` (required)

**Validation Rules**: 10 checks including format, duplicate detection, range validation

### answer_reward

**Purpose**: Score RAG-style answer generation with retrieval

**Reward Range**: [-2.0, 3.0]

**Components**:
- Format score: 0.0 (valid) or -1.0 (invalid)
- Retrieval score: +1.0 (used) or -1.0 (not used)
- Accuracy score: 2.0 * F1 score (continuous, 0.0 to 2.0)

**Required Data**:
- `predict_str`: Response with documents, query, think, and answer tags
- `ground_truth`: Correct answer text
- `extra_info`: Optional (not used)

**Validation Rules**: 8 checks including tag matching, answer length, retrieval usage

## Integration Instructions

### Step 1: Choose Reward Function

For video exploration tasks:
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

For answer generation tasks:
```yaml
custom_reward_function:
  path: verl/utils/reward_score/answer_reward.py
  name: compute_score
```

### Step 2: Validate Your Data

```bash
# For video thinking
python verl/utils/reward_score/validate_reward_data_format.py \
    --data_path /path/to/data.parquet \
    --reward_type video_think

# For answer generation
python verl/utils/reward_score/validate_reward_data_format.py \
    --data_path /path/to/data.parquet \
    --reward_type answer
```

### Step 3: Run Training

The async reward manager will automatically load your custom reward function.

## File Locations

```
verl/
├── configs/
│   └── reward/
│       ├── video_think_reward_example.yaml
│       └── answer_reward_example.yaml
├── utils/
│   └── reward_score/
│       ├── think_with_video_reward.py
│       ├── answer_reward.py
│       └── validate_reward_data_format.py
├── REWARD_FUNCTIONS_GUIDE.md
└── REWARD_MIGRATION_SUMMARY.md (this file)
```

## Testing Recommendations

### For think_with_video_reward

1. **Format Validation**:
   - Test with missing/mismatched tags
   - Test with duplicate frame selections
   - Test with invalid frame ranges

2. **Accuracy Testing**:
   - Test correct vs incorrect answers
   - Verify bonus rewards are awarded correctly
   - Check time query bonus requires time pattern in question

3. **Edge Cases**:
   - Empty extra_info
   - Missing required fields
   - Very long trajectories with many actions

### For answer_reward

1. **Format Validation**:
   - Test with missing tags
   - Test with answers >10 words
   - Test with unindexed documents

2. **F1 Score Testing**:
   - Test exact matches (F1 = 1.0)
   - Test partial matches
   - Test yes/no questions
   - Test with punctuation/case variations

3. **Edge Cases**:
   - Empty ground truth
   - Multiple answer tags
   - Missing retrieval documents

## Usage Examples

### Basic Usage - Video Thinking

```python
from verl.utils.reward_score.think_with_video_reward import compute_score

result = compute_score(
    predict_str="""<think>Let me check frame 100.</think>
<action>zoom in frame 100</action>
<think>I see a car. Answer is B.</think>
<action>output answer: B</action>""",
    ground_truth="B",
    extra_info={
        "question": "What is shown in the video?",
        "total_frames": 1000
    }
)

print(result)
# {'score': 1.1, 'acc': 1.0, 'format': 1.0, 'other': 0.1}
```

### Basic Usage - Answer

```python
from verl.utils.reward_score.answer_reward import compute_score

result = compute_score(
    predict_str="""<|begin_of_documents|>
(1) Paris is the capital of France.
<|end_of_documents|>
<|begin_of_query|>
What is the capital of France?
<|end_of_query|>
Based on document 1, the answer is Paris.
</think>
<answer>Paris</answer>""",
    ground_truth="Paris"
)

print(result)
# {'score': 3.0, 'format': 0.0, 'retrieval': 1.0, 'acc': 2.0,
#  'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
```

## Metrics to Monitor

### For think_with_video_reward

- `reward/score`: Total reward score
- `reward/acc`: Accuracy rate (% of correct answers)
- `reward/format`: Format validation rate (% passing format checks)
- `reward/other`: Average bonus rewards
- Distribution of bonus types (time query, frame selection, zoom)

### For answer_reward

- `reward/score`: Total reward score
- `reward/format`: Format validation rate
- `reward/retrieval`: Retrieval usage rate
- `reward/acc`: Average accuracy reward
- `reward/f1`: Average F1 score
- `reward/precision`: Average precision
- `reward/recall`: Average recall

## Troubleshooting Quick Reference

| Issue | Reward Type | Solution |
|-------|-------------|----------|
| Low format scores | Both | Check tag matching, run validation script |
| Low bonus scores | Video | Increase bonus weights, verify action usage |
| Low F1 scores | Answer | Encourage shorter answers, normalize ground truth |
| Negative scores | Answer | Fix format errors first, then check retrieval |
| High variance | Both | Monitor format scores, check data quality |

## Migration Checklist

- [x] Migrate think_with_video_reward.py
- [x] Migrate answer_reward.py
- [x] Create configuration examples
- [x] Write comprehensive documentation
- [x] Create validation script
- [ ] Test with real data (user to complete)
- [ ] Integration testing with training pipeline (user to complete)
- [ ] Benchmark performance (user to complete)

## Next Steps

1. **Validate Your Data**: Run validation script on your parquet files
2. **Test Integration**: Try both reward functions with small data samples
3. **Monitor Metrics**: Set up logging for all reward components
4. **Tune Parameters**: Adjust bonus weights based on your task requirements
5. **Production Testing**: Run full training with validated configurations

## References

- **Original Implementation**: `/mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/framethinker/FrameThinker-RL/verl/workers/reward_manager/`
- **New Implementation**: `/mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl/verl/utils/reward_score/`
- **Documentation**: `verl/REWARD_FUNCTIONS_GUIDE.md`
- **Validation Tool**: `verl/utils/reward_score/validate_reward_data_format.py`

## Support

For questions or issues:
1. Check `REWARD_FUNCTIONS_GUIDE.md` for detailed documentation
2. Run validation script to verify data format
3. Review configuration examples in `verl/configs/reward/`
4. Check troubleshooting sections in documentation

---

**Status**: ✓ Migration Complete - Ready for Testing

**Last Updated**: 2025-11-22
