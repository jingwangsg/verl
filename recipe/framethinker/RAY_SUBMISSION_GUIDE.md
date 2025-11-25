# Ray Job Submission Guide for FrameThinker Training

This guide explains how to submit FrameThinker training jobs using Ray job submission.

## Quick Start

### Debug/Testing (Minimal Resources)
```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl
bash verl/examples/framethinker/submit_train_debug.sh
```

### Full Training
```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl
bash verl/examples/framethinker/submit_train_frame_thinker.sh
```

### Custom Experiment Name
```bash
EXP_NAME=my_experiment bash verl/examples/framethinker/submit_train_debug.sh
```

### Custom Model Path
```bash
MODEL_PATH=/path/to/model EXP_NAME=custom_exp bash verl/examples/framethinker/submit_train_frame_thinker.sh
```

## How It Works

### 1. Runtime Environment (`runtime_env.yaml`)
Located at: `/mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl/runtime_env.yaml`

Key configurations:
- **py_executable**: Points to the virtual environment Python (with all dependencies like tensordict)
- **PATH**: Prepends venv bin directory to ensure correct Python packages are used
- **working_dir**: Sets working directory to `verl/` where scripts run from
- **env_vars**: Configures Ray logging, HuggingFace cache, and debugging options

This ensures Ray workers have access to all installed dependencies (tensordict, vllm, etc.)

### 2. Submission Scripts

#### Debug Script (`submit_train_debug.sh`)
- Uses minimal resources (1 node, 2 GPUs)
- Small batch sizes (batch_size=2, n=2)
- Reduced sequence lengths (4096 tokens)
- 1 epoch for quick validation
- Console logging only

#### Full Training Script (`submit_train_frame_thinker.sh`)
- Full resources (2 nodes, 8 GPUs per node)
- Production batch sizes (batch_size=32, n=8)
- Full sequence lengths (8192 tokens)
- 10 epochs
- WandB + console logging

### 3. Training Scripts

Both submission scripts call the corresponding training script:
- `train_frame_thinker_debug.sh` - Debug configuration
- `train_frame_thinker_vhonly.sh` - Full training configuration

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAY_ADDRESS` | Ray cluster address | `http://127.0.0.1:8300` |
| `EXP_NAME` | Experiment name | `framethinker_debug` or `framethinker_baseline_v2` |
| `MODEL_PATH` | Model checkpoint path | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `TRAIN_FILES` | Training data path | `Video-Holmes/train_migrated.parquet` |
| `VAL_FILES` | Validation data path | `Video-Holmes/test_migrated.parquet` |

## Ray Job Commands

### Check Job Status
```bash
RAY_ADDRESS="http://127.0.0.1:8300" ray job status <submission-id>
```

### View Job Logs
```bash
RAY_ADDRESS="http://127.0.0.1:8300" ray job logs <submission-id>
```

### Stop a Job
```bash
RAY_ADDRESS="http://127.0.0.1:8300" ray job stop <submission-id>
```

### List All Jobs
```bash
RAY_ADDRESS="http://127.0.0.1:8300" ray job list
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensordict'"
**Solution**: The runtime_env.yaml ensures Ray workers use the virtual environment. If you still see this error:
1. Verify the venv has tensordict installed: `pip list | grep tensordict`
2. Check runtime_env.yaml paths are correct
3. Ensure Ray cluster can access the working directory

### Issue: "Runtime environment file not found"
**Solution**: Make sure you run the submission script from the project root directory:
```bash
cd /mnt/amlfs-03/shared/jingwang/PROJECTS/video_reason/agent_rl
bash verl/examples/framethinker/submit_train_debug.sh
```

### Issue: "Tool config not found"
**Solution**: Verify the tool configuration exists:
```bash
ls -la verl/recipe/framethinker/video_think_tool_config.yaml
```

### Issue: Ray cluster not running
**Solution**: Check Ray status and start if needed:
```bash
ray status
# If not running:
ray start --head --port=8300
```

## File Structure

```
agent_rl/
├── runtime_env.yaml                              # Ray runtime environment config
└── verl/
    ├── examples/framethinker/
    │   ├── submit_train_debug.sh                 # Ray submission for debug
    │   ├── submit_train_frame_thinker.sh         # Ray submission for full training
    │   ├── train_frame_thinker_debug.sh          # Debug training script
    │   ├── train_frame_thinker_vhonly.sh         # Full training script
    │   ├── train_frame_thinker_vhonly_hydra.sh   # Hydra-based alternative
    │   ├── migrate_dataset.py                    # Dataset migration utility
    │   ├── README.md                             # General usage guide
    │   ├── DATA_FORMAT.md                        # Dataset format docs
    │   ├── MIGRATION_SUMMARY.md                  # Migration details
    │   └── RAY_SUBMISSION_GUIDE.md               # This file
    └── recipe/framethinker/
        └── video_think_tool_config.yaml          # VideoThinkTool configuration
```

## Key Differences from Direct Execution

### Before (Direct Execution)
```bash
bash examples/framethinker/train_frame_thinker_vhonly.sh
# Problem: Ray workers don't inherit environment, missing dependencies
```

### After (Ray Job Submission)
```bash
bash examples/framethinker/submit_train_frame_thinker.sh
# Solution: runtime_env.yaml ensures Ray workers have correct environment
```

## Next Steps

1. **Test with debug script first**:
   ```bash
   bash verl/examples/framethinker/submit_train_debug.sh
   ```

2. **Monitor the job**:
   ```bash
   # Get submission ID from output, then:
   RAY_ADDRESS="http://127.0.0.1:8300" ray job logs <submission-id> --follow
   ```

3. **Once debug passes, run full training**:
   ```bash
   EXP_NAME=framethinker_exp1 bash verl/examples/framethinker/submit_train_frame_thinker.sh
   ```

## Additional Resources

- **FrameThinker README**: `verl/examples/framethinker/README.md`
- **Migration Summary**: `verl/examples/framethinker/MIGRATION_SUMMARY.md`
- **Dataset Format**: `verl/examples/framethinker/DATA_FORMAT.md`
- **Ray Documentation**: https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
