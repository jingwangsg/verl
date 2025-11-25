# FrameThinker Training with ToolAgentLoop

This directory contains the migrated FrameThinker training scripts that use the new `ToolAgentLoop` and `VideoThinkTool` architecture.

## Files

- **`train_frame_thinker_vhonly.sh`** - Main training script (direct parameter overrides)
- **`train_frame_thinker_vhonly_hydra.sh`** - Alternative script using Hydra config
- **`config/framethinker_grpo.yaml`** - Hydra configuration file
- **`../../recipe/framethinker/video_think_tool_config.yaml`** - VideoThinkTool configuration

## Migration Summary

### Key Changes from Original Script

| Component | Old Implementation | New Implementation |
|-----------|-------------------|-------------------|
| Agent Loop | `agent.activate_agent=True` | `multi_turn.enable=True` |
| Tool Config | Inline parameters | Separate YAML config file |
| Max Turns | `agent.max_turns=5` | `multi_turn.max_assistant_turns=5` |
| Tool Parser | Implicit | `multi_turn.format=think_with_video` |
| Tool Response | N/A | `multi_turn.tool_response_role=user` |

### Architecture

```
FrameThinker Training Pipeline
│
├── Data Loading (RL Dataset)
│   └── Video-Holmes parquet files
│
├── ToolAgentLoop
│   ├── VideoThinkTool
│   │   ├── get frame number at time
│   │   ├── zoom in frame
│   │   └── choose frames between
│   └── Multi-turn state machine
│
├── Rollout (vLLM)
│   └── Qwen2.5-VL-7B-Instruct
│
└── GRPO Training
    └── Reward: think_with_video_reward
```

## Usage

### Option 1: Direct Parameter Script (Recommended for Testing)

```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl

# Run with default settings
bash examples/framethinker/train_frame_thinker_vhonly.sh

# Override experiment name
EXP_NAME=my_experiment bash examples/framethinker/train_frame_thinker_vhonly.sh

# Override model path
MODEL_PATH=path/to/model bash examples/framethinker/train_frame_thinker_vhonly.sh
```

### Option 2: Hydra Config Script (Recommended for Production)

```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl

# Run with Hydra config
bash examples/framethinker/train_frame_thinker_vhonly_hydra.sh

# Override specific parameters
bash examples/framethinker/train_frame_thinker_vhonly_hydra.sh \
    data.train_batch_size=64 \
    trainer.total_epochs=20
```

## Configuration Details

### VideoThinkTool Configuration

Located at `recipe/framethinker/video_think_tool_config.yaml`:

```yaml
tools:
  - class_name: "verl.tools.video_think_tool.VideoThinkTool"
    config:
      max_workers: 4
      timeout: 30
    tool_schema:
      # OpenAI function schema
```

### Multi-Turn Parameters

```python
actor_rollout_ref.rollout.multi_turn:
  enable: True                           # Enable multi-turn agent loop
  tool_config_path: <path>              # Path to tool config YAML
  max_assistant_turns: 5                # Max model responses
  max_user_turns: 20                    # Max tool/user responses
  max_parallel_calls: 1                 # Max concurrent tool calls
  format: think_with_video              # Tool parser format
  max_tool_response_length: 4096        # Truncate tool responses
  tool_response_truncate_side: middle   # Where to truncate
  tool_response_role: user              # Role for tool responses
```

### Reward Function

Uses `verl/utils/reward_score/think_with_video_reward.py`:

- **Format correctness**: +1.0 for valid think/action tags
- **Answer correctness**: +1.0 for correct answer
- **Coarse-to-fine navigation**: +0.2 (controlled by `lambda_gfn`)
- **Time-based navigation**: +0.2 if used
- **Zoom usage**: +0.1 if used

## Important Notes

### 1. Video Metadata Passing

**CRITICAL**: The current implementation requires passing video metadata to the tool. This needs to be handled in the dataset or data collator:

```python
# In your data processing:
tools_kwargs = {
    "video_think": {
        "create_kwargs": {
            "video_path": "/path/to/video.mp4",
            "fps": 30,
            "total_frames": 9000,
            "width": 1920,
            "height": 1080
        }
    }
}
```

**TODO**: Verify how `tools_kwargs` is populated from the dataset. The dataset needs to provide:
- `video_path` or video file identifier
- Video metadata (fps, total_frames, width, height)

### 2. Tool Response Role

Set `tool_response_role: user` for compatibility with the original FrameThinker format where tool responses appeared as user messages.

### 3. Memory Requirements

- **2 nodes × 8 GPUs** required (same as original)
- **Offloading enabled**: Both param and optimizer offloading for actor
- **GPU memory**: 0.6 utilization for vLLM rollout
- **Tensor parallelism**: TP=2 for Qwen2.5-VL-7B

### 4. Data Paths

Update these paths if your data is located elsewhere:

```bash
TRAIN_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train.parquet
VAL_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test.parquet
data.media_dir=/mnt/amlfs-02/shared/datasets/s3/video_reason/
```

## Debugging

### Test Multi-Turn Setup

Before full training, test the multi-turn setup:

```bash
# Reduce scale for testing
bash examples/framethinker/train_frame_thinker_vhonly.sh \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.train_batch_size=2 \
    trainer.total_epochs=1 \
    trainer.test_freq=1
```

### Common Issues

1. **"tools_kwargs not found"**
   - Check dataset implementation provides `tools_kwargs` with video metadata
   - See `verl/utils/dataset/rl_dataset.py` for reference

2. **"Tool config not found"**
   - Verify `TOOL_CONFIG_PATH` points to correct location
   - Use absolute path or verify relative path resolution

3. **"Too many open files"**
   - Run `ulimit -n 65535` before training
   - Or add to script: `ulimit -n 65535`

4. **vLLM image limit errors**
   - Adjust `max_vllm_images` if needed (was 128 in original)
   - May need to set via vLLM engine kwargs

## Monitoring

### WandB

Logs will be sent to WandB project `video_holmes_rl`:

```python
trainer.logger='["console","wandb"]'
trainer.project_name=video_holmes_rl
trainer.experiment_name=<your_exp_name>
```

### TensorBoard

Logs saved to:

```
${SAVE_CHECKPOINT_DIR}/logs/tensorboard
${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board
```

## Next Steps

1. **Verify dataset integration**:
   - Check if `rl_dataset.py` properly passes video metadata
   - May need to modify data collator

2. **Small-scale test**:
   - Run 1-GPU test to verify agent loop works
   - Check tool calls in logs

3. **Full training**:
   - Launch 2-node, 16-GPU training
   - Monitor WandB for rewards and tool usage

## Differences from Original

- **No `agent.tool_name_key`**: Tool identification now via config
- **No `agent.concurrent_workers`**: Replaced by `max_parallel_calls`
- **No `agent.show_tqdm`**: Removed in new implementation
- **New parameter `format`**: Specifies tool parser (`think_with_video`, `gpt-oss`, etc.)
- **New parameter `tool_response_role`**: Controls message role for tool outputs

## Migration Checklist

- [x] Create `video_think_tool_config.yaml`
- [x] Migrate training script parameters
- [x] Create Hydra config alternative
- [ ] **Test dataset provides `tools_kwargs`** ⚠️ CRITICAL
- [ ] Run small-scale test (1 GPU)
- [ ] Verify agent loop executes correctly
- [ ] Run full-scale training (2 nodes)
- [ ] Compare results with original FrameThinker-RL

## Support

For issues or questions:
- Check `verl/claude_wiki/VIDEO_THINK_TOOL_USAGE.md`
- Review `verl/experimental/agent_loop/tool_agent_loop.py`
- See example: `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh`
