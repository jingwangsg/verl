# VERL Essential Commands

## Development Setup

### Quick Python-Only Installation
```bash
# For vLLM backend
pip install -e .[test,vllm]

# For SGLang backend  
pip install -e .[test,sglang]
```

### Full GPU Setup
```bash
pip install -e .[test,vllm,gpu]
```

## Pre-commit Hooks (CRITICAL - Always Run Before Committing)
```bash
# Install hooks
pre-commit install

# Run all checks
pre-commit run --all-files

# CRITICAL: After modifying Hydra configs, regenerate trainer configs
bash scripts/generate_trainer_config.sh
```

## Testing

### CPU-Only Tests
```bash
pytest -s tests/*_on_cpu.py
```

### GPU Tests (Standard)
```bash
# Exclude special categories and CPU tests
pytest -s --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' tests/
```

### Specific Test Suites
```bash
pytest tests/trainer/
pytest tests/workers/
pytest tests/utils/
```

### Distributed Tests (Requires 2+ GPUs)
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py
```

## Linting & Formatting

### Ruff (Linting and Formatting)
```bash
# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .
```

### Type Checking
```bash
# Only strict on specific modules (see pyproject.toml)
mypy verl/
```

## Running Examples

### GRPO Training
```bash
# Example: Qwen3-8B
bash examples/grpo_trainer/run_qwen3-8b.sh
```

### PPO Training
```bash
# Example with sequence packing
bash examples/ppo_trainer/run_qwen2-7b_seq_balance.sh

# Example with sequence parallelism
bash examples/ppo_trainer/run_deepseek7b_llm_sp2.sh
```

### SFT (Supervised Fine-tuning)
```bash
# With LoRA
bash examples/sft/gsm8k/run_qwen_05_peft.sh

# With Liger-kernel
bash examples/sft/gsm8k/run_qwen_05_sp2_liger.sh
```

## System Utilities (macOS/Darwin)

### File Operations
```bash
ls -la          # List files with details
find . -name    # Find files by name
grep -r         # Search in files recursively
```

### Git Operations
```bash
git status
git add .
git commit -m "message"
git push
git pull
git diff
```

### Process Management
```bash
ps aux | grep python    # Find Python processes
top                     # Monitor system resources
kill -9 <pid>           # Force kill process
```

## Documentation
```bash
# Documentation is available at:
# https://verl.readthedocs.io/en/latest/
```

## Important Notes
- Always run pre-commit hooks before committing
- After modifying Hydra configs, MUST run: `bash scripts/generate_trainer_config.sh`
- Use `pytest -s` for verbose test output
- GPU tests require appropriate GPU resources
