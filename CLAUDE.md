# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About verl

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training framework for large language models. It's the open-source version of the HybridFlow framework presented at EuroSys 2025. The framework uses a hybrid-controller programming model that enables flexible representation and efficient execution of complex post-training dataflows.

**Key capabilities:**
- Multiple RL algorithms: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, and more
- Training backends: FSDP, FSDP2, Megatron-LM
- Inference engines: vLLM (v0.8.2+), SGLang, HF Transformers
- Model support: Qwen, Llama, Gemma2, DeepSeek, and other HuggingFace models
- Scales to 671B models with expert parallelism
- Multi-turn and tool-calling support for agentic workflows

## Development Commands

### Installation

```bash
# Install with vLLM backend
pip install -e .[test,vllm]

# Install with SGLang backend
pip install -e .[test,sglang]

# For GPU-specific features (flash-attn, liger-kernel)
pip install -e .[test,vllm,gpu]

# For math verification
pip install -e .[test,vllm,math]
```

See the [installation docs](https://verl.readthedocs.io/en/latest/start/install.html) for GPU/NPU setup and dependency details.

### Running Examples

Training scripts use Hydra configs and are located in `examples/`:

```bash
# Run PPO training
bash examples/ppo_trainer/run_qwen2-7b_rm.sh

# Run GRPO training
bash examples/grpo_trainer/run_qwen2-7b_math.sh

# Run multi-turn tool use
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

The main entry point is:
```bash
python3 -m verl.trainer.main_ppo [hydra_config_overrides...]
```

### Linting and Formatting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook (ruff is the main linter)
pre-commit run --all-files --show-diff-on-failure ruff
```

The codebase uses `ruff` for linting/formatting with a 120 character line length (configured in `pyproject.toml`).

### Testing

```bash
# Run pytest on all tests
pytest tests/

# Run specific test
pytest tests/path/to/test_file.py

# CI test categories:
# - tests/trainer/ - trainer functionality
# - tests/models/ - model implementations
# - tests/workers/ - worker implementations
# - tests/special_distributed/ - multi-GPU tests
# - tests/special_e2e/ - end-to-end tests
# - tests/*_on_cpu.py - CPU-only tests
```

See `.github/workflows/` for full CI test configurations.

### Building Documentation

```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html
python -m http.server -d _build/html/
```

Then open http://localhost:8000

## Architecture Overview

### Core Components

**verl/trainer/** - Training orchestration and main entry points
- `main_ppo.py` - Main PPO/GRPO training entry point
- `ppo/` - PPO-specific algorithms and trainers
- `config/` - Hydra configuration system (actor, critic, rollout, reward model configs)

**verl/workers/** - Distributed worker implementations
- `actor/` - Policy network (actor) training workers
- `critic/` - Value network (critic) training workers
- `rollout/` - Inference workers for trajectory generation
- `reward_model/` - Reward model workers
- `fsdp_workers.py`, `megatron_workers.py` - Backend-specific worker implementations

**verl/models/** - Model implementations and adapters for various architectures

**verl/utils/** - Shared utilities and helper functions

**recipe/** - Algorithm implementations and research recipes
- Contains implementations of various RL algorithms (DAPO, PRIME, SPPO, etc.)
- Each recipe typically includes config files and custom training logic

### Hybrid-Controller Architecture

verl uses a hybrid programming model that separates:
1. **Single-Controller** - Sequential execution within a worker (gradient computation, model updates)
2. **Multi-Controller** - Distributed execution across workers via Ray (rollout, training, reward computation)

This design enables:
- Flexible device mapping (different model placements on GPU sets)
- Efficient actor model resharding between training/generation (3D-HybridEngine)
- Integration with existing frameworks (FSDP, Megatron, vLLM, SGLang)

### Configuration System

The framework uses **Hydra** for hierarchical configuration management. Configs are composed from:
- `verl/trainer/config/ppo_trainer.yaml` (or `ppo_megatron_trainer.yaml`) - Main config
- `actor/`, `critic/`, `rollout/`, `reward_model/` - Component configs
- Command-line overrides using dot notation (e.g., `actor_rollout_ref.actor.optim.lr=1e-6`)

Key config groups:
- `actor_rollout_ref` - Combines actor (training), rollout (generation), and reference model
- `critic` - Value function for advantage estimation
- `reward_model` - External reward model (optional, can use function-based rewards)
- `data` - Dataset configuration
- `algorithm` - RL algorithm settings (PPO, GRPO, etc.)

### Training Backends

**FSDP/FSDP2:** Default backend for most models
- Strategy: `fsdp` or `fsdp2`
- Supports gradient checkpointing, parameter/optimizer offloading
- FSDP2 is recommended for better throughput and composability

**Megatron-LM:** For very large models (235B+, MoE models)
- Supports tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP)
- Use `*_megatron_trainer.yaml` configs

**Inference Engines:**
- vLLM: Fast inference with PagedAttention, supports TP
- SGLang: Multi-turn, tool-calling, RadixAttention for prefix caching
- HF Transformers: Fallback option

## Common Patterns

### Adding a New Model

**For FSDP backend:**
1. Add model class to `verl/models/` (or use existing HF model)
2. Implement tokenizer handling if needed
3. Update `verl/workers/fsdp_workers.py` if custom initialization required

**For Megatron backend:**
1. Add model to Megatron-LM fork or use existing architecture
2. Configure TP/PP/EP settings in config
3. See [Megatron extension guide](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)

### Adding a New RL Algorithm

1. Create config in `verl/trainer/config/algorithm/`
2. Implement core logic in `verl/trainer/ppo/core_algos.py` or as a separate module
3. Add trainer class if needed (see `recipe/` for examples)
4. Hook into `main_ppo.py` via algorithm config

### Multi-turn and Tool Use

Multi-turn workflows use SGLang's constrained generation:
1. Define tools in `tool_config.yaml` (format: Hermes, Llama3 JSON, etc.)
2. Configure `actor_rollout_ref.rollout.multi_turn.enable=True`
3. Set `actor_rollout_ref.rollout.name=sglang`
4. Specify `tool_config_path` and optional `interaction_config_path`

See `examples/sglang_multiturn/` for working examples.

### Reward Functions

Two types of rewards:
1. **Model-based:** External reward model (e.g., `reward_model.model.path=...`)
2. **Function-based:** Python function for verifiable tasks (math, code execution)

Function rewards are defined in dataset processing scripts (see `examples/data_preprocess/`).

## Important Notes

- **Data Preparation:** Always run data preprocessing scripts in `examples/data_preprocess/` before training
- **vLLM Version:** Use v0.8.2+, avoid v0.7.x (contains critical bugs)
- **Memory Tuning:** Key parameters are `gpu_memory_utilization`, `max_num_batched_tokens`, `ppo_micro_batch_size_per_gpu`
- **Performance:** See the [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for optimization tips
- **Breaking Changes:** List of breaking changes since v0.4 at https://github.com/volcengine/verl/discussions/2270

## Debugging

- Set `trainer.logger='["console","wandb"]'` for detailed logging
- Use `trainer.val_before_train=True` to validate setup before training
- Enable `actor_rollout_ref.rollout.calculate_log_probs=True` for debugging probability issues
- Profile with `nsys`: Set profiler configs in actor/rollout sections

## File Naming Conventions

- `*_on_cpu.py` in tests/ - CPU-only tests
- `special_*` in tests/ - Special test categories (distributed, e2e, npu, sanity)
- `run_*.sh` in examples/ - Executable training scripts
- `*_megatron_*` - Megatron-LM backend specific files
- `legacy_*` - Deprecated or backwards compatibility code
