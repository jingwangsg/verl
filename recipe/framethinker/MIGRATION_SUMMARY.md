# FrameThinker Migration Summary

## Overview

Successfully migrated the FrameThinker training script from the old verl version to the new architecture using `ToolAgentLoop` and `VideoThinkTool`.

**Migration Date**: 2025-11-22
**Status**: ✅ Configuration Complete - Ready for Testing

---

## What Was Done

### ✅ Completed Tasks

1. **Created VideoThinkTool configuration** (`recipe/framethinker/video_think_tool_config.yaml`)
   - Defines the `video_think` tool schema
   - Configures async workers and timeouts

2. **Created training scripts**
   - `examples/framethinker/train_frame_thinker_vhonly.sh` - Direct parameter overrides
   - `examples/framethinker/train_frame_thinker_vhonly_hydra.sh` - Hydra-based config

3. **Created Hydra configuration** (`examples/framethinker/config/framethinker_grpo.yaml`)
   - Modular configuration structure
   - Easier to maintain and customize

4. **Created documentation**
   - `README.md` - Usage guide and troubleshooting
   - `DATA_FORMAT.md` - Dataset format specification
   - `MIGRATION_SUMMARY.md` - This file

5. **Created migration tools**
   - `migrate_dataset.py` - Script to convert old dataset format to new

6. **Validated implementation**
   - All YAML files validated
   - Scripts made executable
   - Dataset support verified

---

## Files Created

```
verl/
├── recipe/framethinker/
│   └── video_think_tool_config.yaml          # Tool configuration
│
├── examples/framethinker/
│   ├── train_frame_thinker_vhonly.sh         # Main training script
│   ├── train_frame_thinker_vhonly_hydra.sh   # Hydra variant
│   ├── migrate_dataset.py                     # Dataset migration script
│   ├── README.md                              # Usage documentation
│   ├── DATA_FORMAT.md                         # Dataset format spec
│   ├── MIGRATION_SUMMARY.md                   # This file
│   └── config/
│       └── framethinker_grpo.yaml            # Hydra config
│
└── (Existing files - no changes needed)
    ├── verl/tools/video_think_tool.py        # Already implemented
    ├── verl/experimental/agent_loop/tool_agent_loop.py  # Already implemented
    ├── verl/utils/reward_score/think_with_video_reward.py  # Already migrated
    └── verl/utils/dataset/templates/framethinker_default.py  # Already migrated
```

---

## Key Architecture Changes

### Old → New Parameter Mapping

| Old Parameter | New Parameter | Notes |
|--------------|---------------|-------|
| `agent.activate_agent=True` | `multi_turn.enable=True` | Enable agent loop |
| `agent.max_turns=5` | `multi_turn.max_assistant_turns=5` | Max model turns |
| `agent.tool_name_key=env_name` | (removed) | Tool config now via YAML |
| `agent.concurrent_workers=1` | `multi_turn.max_parallel_calls=1` | Renamed |
| `agent.show_tqdm=True` | (removed) | Not in new implementation |
| `agent.max_vllm_images=128` | (TBD) | May need vLLM config |
| (none) | `multi_turn.tool_config_path=...` | **New**: Path to tool YAML |
| (none) | `multi_turn.format=think_with_video` | **New**: Tool parser format |
| (none) | `multi_turn.tool_response_role=user` | **New**: Tool message role |

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Loading (RLDataset)                                    │
│     └─ Video-Holmes parquet files                               │
│        └─ extra_info.tools_kwargs.video_think.create_kwargs     │
│           ├─ video_path                                         │
│           ├─ fps                                                 │
│           ├─ total_frames                                       │
│           ├─ width                                               │
│           └─ height                                              │
│                                                                  │
│  2. Multi-Turn Rollout (ToolAgentLoop)                          │
│     ├─ State Machine (PENDING → GENERATING → PROCESSING_TOOLS) │
│     ├─ VideoThinkTool                                           │
│     │  ├─ get frame number at time MM:SS                        │
│     │  ├─ zoom in frame <INDEX>                                 │
│     │  └─ choose frames between <START> and <END>               │
│     └─ vLLM Inference Engine                                    │
│        └─ Qwen2.5-VL-7B-Instruct (TP=2)                        │
│                                                                  │
│  3. Reward Computation                                          │
│     └─ think_with_video_reward.compute_score()                 │
│        ├─ Format correctness: +1.0                              │
│        ├─ Answer correctness: +1.0                              │
│        ├─ Coarse-to-fine: +0.2 (λ_gfn=0.2)                     │
│        ├─ Time-based nav: +0.2 (if used)                        │
│        └─ Zoom usage: +0.1 (if used)                            │
│                                                                  │
│  4. GRPO Training                                               │
│     └─ Update Qwen2.5-VL-7B-Instruct policy                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Next Steps (CRITICAL)

### 🔴 Step 1: Migrate Dataset (REQUIRED)

Your existing Video-Holmes dataset needs to be migrated to include `tools_kwargs`:

```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl

# Migrate training set
python3 examples/framethinker/migrate_dataset.py \
    --input /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train.parquet \
    --output /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet \
    --validate

# Migrate validation set
python3 examples/framethinker/migrate_dataset.py \
    --input /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test.parquet \
    --output /mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet \
    --validate
```

**Then update training script paths**:
```bash
TRAIN_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet
VAL_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet
```

### ⚠️ Step 2: Small-Scale Test (RECOMMENDED)

Before full 2-node training, test with minimal resources:

```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl

# Test with 1 GPU, small batch
bash examples/framethinker/train_frame_thinker_vhonly.sh \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.train_batch_size=2 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    TRAIN_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet \
    VAL_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet
```

**What to check:**
- ✅ Dataset loads without errors
- ✅ `tools_kwargs` is passed correctly
- ✅ VideoThinkTool creates instances
- ✅ Agent loop executes tool calls
- ✅ Reward function computes scores
- ✅ Training step completes

**Expected logs:**
```
Initialized tools: {'video_think': <VideoThinkTool object>}
Created VideoThinkTool instance ...: video=..., fps=30, frames=1800
Extracted 8 frames from range [100, 500)
```

### 🚀 Step 3: Full Training

Once testing passes, launch full-scale training:

```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl

# Option 1: Direct parameter script
bash examples/framethinker/train_frame_thinker_vhonly.sh \
    TRAIN_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet \
    VAL_FILES=/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet

# Option 2: Hydra config script
bash examples/framethinker/train_frame_thinker_vhonly_hydra.sh \
    "data.train_files=[/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/train_migrated.parquet]" \
    "data.val_files=[/mnt/amlfs-02/shared/datasets/s3/video_reason/Video-Holmes/test_migrated.parquet]"
```

**Monitor:**
- WandB: Project `video_holmes_rl`
- TensorBoard: `${SAVE_CHECKPOINT_DIR}/logs/tensorboard`
- Checkpoints: `${SAVE_CHECKPOINT_DIR}/video_holmes_rl/${EXP_NAME}`

### 📊 Step 4: Validation

Compare new results with original FrameThinker-RL:

**Metrics to check:**
- Average episode reward
- Tool usage frequency (zoom, time-based nav, coarse-to-fine)
- Answer accuracy
- Training convergence speed

---

## Potential Issues & Solutions

### Issue: "tools_kwargs is empty"

**Symptoms:**
```
WARNING: tools_kwargs is empty for index 0, data source: video_holmes
```

**Cause:** Dataset not migrated

**Solution:** Run `migrate_dataset.py` on your parquet files

---

### Issue: "Video file not found"

**Symptoms:**
```
Error: Failed to open video for low-res: [Errno 2] No such file or directory
```

**Cause:** Incorrect `data.media_dir` or wrong video paths

**Solution:**
1. Check `data.media_dir=/mnt/amlfs-02/shared/datasets/s3/video_reason/`
2. Verify videos exist: `ls /mnt/amlfs-02/shared/datasets/s3/video_reason/videos/`
3. Check first parquet row: video_path should be relative or absolute

---

### Issue: "Tool config not found"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '...video_think_tool_config.yaml'
```

**Cause:** Wrong path resolution

**Solution:**
```bash
# Use absolute path
TOOL_CONFIG_PATH="/mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl/recipe/framethinker/video_think_tool_config.yaml"

# Or verify relative path from working directory
pwd  # Should be /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/verl
ls -la recipe/framethinker/video_think_tool_config.yaml
```

---

### Issue: Out of memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size**: `data.train_batch_size=16`
2. **Enable offloading** (already enabled):
   - `actor.fsdp_config.param_offload=True`
   - `actor.fsdp_config.optimizer_offload=True`
3. **Reduce GPU memory for vLLM**: `rollout.gpu_memory_utilization=0.5`
4. **Use gradient checkpointing**: `model.enable_gradient_checkpointing=True`

---

### Issue: "Too many open files"

**Symptoms:**
```
OSError: [Errno 24] Too many open files
```

**Solution:**
```bash
# Add to script or run before training
ulimit -n 65535
```

Already added to `train_frame_thinker_vhonly_hydra.sh`.

---

## Comparison with Original

### What Changed

| Aspect | Old (FrameThinker-RL) | New (Current verl) |
|--------|----------------------|-------------------|
| Agent Loop | Built-in `agent.*` params | `ToolAgentLoop` with tool_config |
| Tool Definition | Inline params | Separate YAML config |
| Tool Metadata | Direct in `extra_info` | Nested in `tools_kwargs` |
| Config Style | Flat parameters | Hydra defaults system |
| Tool Response | Implicit user role | Configurable via `tool_response_role` |

### What Stayed the Same

- ✅ Reward function: `think_with_video_reward.py`
- ✅ Message template: `framethinker_default.py`
- ✅ GRPO algorithm with KL=0
- ✅ Model: Qwen2.5-VL-7B-Instruct
- ✅ Dataset: Video-Holmes
- ✅ Hardware: 2 nodes × 8 GPUs
- ✅ Memory offloading strategy

---

## Contact & Support

**Documentation:**
- `examples/framethinker/README.md` - Usage guide
- `examples/framethinker/DATA_FORMAT.md` - Dataset format
- `claude_wiki/VIDEO_THINK_TOOL_USAGE.md` - Tool usage

**Reference Examples:**
- Tool agent: `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh`
- VLM GRPO: `examples/grpo_trainer/run_qwen2_5_vl-7b.sh`

**Key Code Locations:**
- Tool: `verl/tools/video_think_tool.py:55`
- Agent Loop: `verl/experimental/agent_loop/tool_agent_loop.py:81`
- Reward: `verl/utils/reward_score/think_with_video_reward.py`

---

## Summary Checklist

Before running training, ensure:

- [ ] Dataset migrated to include `tools_kwargs`
- [ ] Video files accessible at paths specified in dataset
- [ ] `data.media_dir` points to correct directory
- [ ] `tool_config_path` points to `video_think_tool_config.yaml`
- [ ] Small-scale test completed successfully
- [ ] WandB credentials configured (if using)
- [ ] Sufficient disk space for checkpoints
- [ ] `ulimit -n 65535` set for file handles

---

**Status**: ✅ Migration Complete - Ready for Testing

**Next Action**: Migrate dataset using `migrate_dataset.py`, then run small-scale test

Good luck! 🚀
