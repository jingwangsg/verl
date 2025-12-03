# VERL Codebase Structure

## Top-Level Directories

### Source Code
- **verl/** - Main source package
- **examples/** - Example scripts and training recipes
- **recipe/** - Advanced algorithm recipes (DAPO, GSPO, PRIME, SPPO, etc.)
- **tests/** - Test suite (unit, integration, E2E)
- **scripts/** - Utility scripts (config generation, etc.)
- **docs/** - Documentation source

### Configuration & Build
- **.github/** - CI/CD workflows
- **.vscode/** - VS Code settings
- **docker/** - Docker configurations
- **pyproject.toml** - Build and tool configuration
- **setup.py**, **setup.sh** - Installation scripts
- **.pre-commit-config.yaml** - Pre-commit hooks configuration
- **requirements*.txt** - Dependency specifications

### Documentation
- **README.md** - Project overview
- **CLAUDE.md** - Claude Code instructions (comprehensive guide)
- **CONTRIBUTING.md** - Contribution guidelines
- **LICENSE**, **Notice.txt** - Legal files

## Main Source Package (verl/)

### Core Modules
```
verl/
├── __init__.py
├── protocol.py              # DataProto - universal data container
├── base_config.py           # Configuration base class
└── py.typed                 # Type checking marker
```

### Trainer Module
```
verl/trainer/
├── ppo/
│   ├── ray_trainer.py       # Main PPO training loop (2000+ lines)
│   ├── core_algos.py        # RL algorithms (GAE, GRPO, CPO, etc.)
│   └── reward.py            # Reward computation utilities
└── config/
    ├── ppo_trainer.yaml     # FSDP-based config template
    ├── ppo_megatron_trainer.yaml  # Megatron-based config
    ├── _generated_*.yaml    # Auto-generated flattened configs
    └── algorithm/           # Algorithm-specific configs
```

### Workers Module
```
verl/workers/
├── fsdp_workers.py          # FSDP-based ActorRolloutRefWorker
├── megatron_workers.py      # Megatron-based workers
├── actor/                   # Actor worker implementations
├── rollout/                 # Rollout engines
│   ├── vllm_rollout.py     # vLLM integration
│   ├── sglang_rollout.py   # SGLang integration
│   └── hf_rollout.py       # HuggingFace fallback
├── sharding_manager/        # Weight resharding
│   ├── fsdp_vllm.py        # FSDP ↔ vLLM weight conversion
│   └── fsdp_ulysses.py     # FSDP with Ulysses sequence parallel
└── reward_manager/          # Reward computation management
```

### Single Controller Module
```
verl/single_controller/
├── ray/
│   └── base.py              # RayWorkerGroup, ResourcePool
└── base/
    └── decorator.py         # Dispatch/collect decorators
```

### Other Modules
```
verl/
├── models/                  # Model implementations and adapters
├── utils/                   # Utility functions
├── configs/                 # Additional configurations
├── tools/                   # Development tools
├── interactions/            # Interaction utilities
├── experimental/            # Experimental features
│   └── agent_loop/         # Agent loop implementation
├── third_party/             # Third-party integrations
├── version/                 # Version management
│   └── version             # Version file (read by setup.py)
└── model_merger/            # Model merging utilities
```

## Test Suite (tests/)

### Test Categories
```
tests/
├── trainer/                 # Trainer module tests
├── workers/                 # Worker module tests
│   ├── actor/              # Actor worker tests
│   └── rollout/            # Rollout engine tests
├── utils/                   # Utility tests
├── special_sanity/          # Sanity checks
│   ├── check_docstrings.py # Docstring coverage
│   ├── check_license.py    # License header check
│   └── ...
├── special_distributed/     # Multi-GPU unit tests
├── special_e2e/            # End-to-end training tests
└── *_on_cpu.py             # CPU-only tests
```

### Test Naming Conventions
- `test_*.py` - Standard unit tests
- `test_special_*.py` - Special test categories
- `*_on_cpu.py` - CPU-only tests (no GPU required)

## Examples Directory

### Training Examples
```
examples/
├── ppo_trainer/             # PPO training scripts
│   ├── run_qwen2-7b_seq_balance.sh
│   └── run_deepseek7b_llm_sp2.sh
├── grpo_trainer/            # GRPO training scripts
│   ├── run_qwen3-8b.sh
│   └── run_qwen2_5_vl-7b.sh  # Multi-modal
├── rloo_trainer/            # RLOO training
├── remax_trainer/           # ReMax training
├── sft/                     # Supervised fine-tuning
│   └── gsm8k/              # GSM8K dataset examples
└── sglang_multiturn/        # Multi-turn examples
```

## Recipe Directory

### Advanced Algorithm Implementations
```
recipe/
├── dapo/                    # DAPO (50 points AIME 2024)
├── gspo/                    # GSPO
├── prime/                   # PRIME
├── sppo/                    # Self-play preference optimization
├── drgrpo/                  # DrGRPO
└── entropy/                 # KL_Cov & Clip_Cov
```

## Scripts Directory

### Utility Scripts
```
scripts/
├── generate_trainer_config.sh  # CRITICAL: Regenerate configs
└── ...
```

## CI/CD Workflows (.github/workflows/)

### Always-Run Workflows
- `sanity.yml` - Sanity checks
- `pre-commit.yml` - Pre-commit validation
- `doc.yml` - Documentation build

### Path-Filtered Workflows
- GPU/CPU unit tests
- E2E tests
- Backend-specific tests (model.yml, vllm.yml, sgl.yml)

## Key Files to Know

### Configuration
- `pyproject.toml` - Build config, ruff, mypy settings
- `.pre-commit-config.yaml` - Pre-commit hooks
- `verl/trainer/config/ppo_trainer.yaml` - Main training config

### Documentation
- `CLAUDE.md` - Comprehensive development guide
- `README.md` - Project overview
- `CONTRIBUTING.md` - Contribution guidelines

### Critical Source Files
- `verl/protocol.py` - Data protocol (1241 lines)
- `verl/trainer/ppo/ray_trainer.py` - Main trainer (2000+ lines)
- `verl/trainer/ppo/core_algos.py` - Algorithms (1800+ lines)

## Navigation Tips

1. **Start with**: `CLAUDE.md` for comprehensive overview
2. **Training flow**: `verl/trainer/ppo/ray_trainer.py`
3. **Algorithms**: `verl/trainer/ppo/core_algos.py`
4. **Workers**: `verl/workers/fsdp_workers.py` or `megatron_workers.py`
5. **Data handling**: `verl/protocol.py`
6. **Configuration**: `verl/trainer/config/`
7. **Examples**: `examples/<algorithm>_trainer/`
8. **Recipes**: `recipe/<algorithm>/`

## File Count Estimates
- **Total source files**: Hundreds of Python files
- **Test files**: 100+ test files
- **Config files**: Dozens of YAML configs
- **Example scripts**: 20+ shell scripts
