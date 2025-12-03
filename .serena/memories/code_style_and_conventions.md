# VERL Code Style and Conventions

## Linting and Formatting

### Ruff Configuration
- **Line length**: 120 characters (not a hard limit for formatter)
- **Tool**: Ruff (replaces flake8, isort, pyupgrade, etc.)

### Enabled Checks
- `E`: pycodestyle errors
- `F`: Pyflakes
- `UP`: pyupgrade
- `B`: flake8-bugbear
- `I`: isort (import sorting)
- `G`: logging format

### Ignored Rules
- `F405`, `F403`: Star imports (allowed)
- `E731`: Lambda expression assignment (allowed)
- `B007`: Loop control variable not used (allowed)
- `UP032`: f-string format (old style allowed)
- `G004`: f-string in .log() (allowed)
- `UP045`: `X | None` for type annotations (old style allowed)
- `UP035`: Deprecated imports (allowed)

### Import Sorting
- **First-party package**: `verl`
- Imports sorted according to isort rules

## Type Checking (mypy)

### Strict Typing Modules (Type checking enforced)
The following modules have strict type checking enabled:
- `verl.trainer.config.algorithm`
- `verl.trainer.ppo.core_algos`
- `verl.trainer.ppo.reward`
- `verl.workers.reward_manager`
- `verl.workers.reward_manager.*`

### General Behavior
- **Pretty output**: Enabled
- **Ignore missing imports**: True (for most modules)
- **Explicit package bases**: True
- **Follow imports**: Skip
- **Blanket ignore**: Enabled for modules not in strict list

## Docstring Requirements

Public functions and classes in these files **MUST** have docstrings:
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/reward.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/workers/sharding_manager/fsdp_vllm.py`
- `verl/workers/sharding_manager/fsdp_ulysses.py`

## License Headers

### Required
All Python files must include Apache 2.0 license headers

### Checked Directories
- `examples`
- `recipe`
- `scripts`
- `tests`
- `verl`
- `setup.py`

## Naming Conventions

### Files and Modules
- Use lowercase with underscores: `my_module.py`
- Test files: `test_*.py` or `*_test.py`
- Special tests: `test_special_*.py`
- CPU-only tests: `*_on_cpu.py`

### Classes
- PascalCase: `MyClass`, `DataProto`, `RayWorkerGroup`

### Functions and Methods
- snake_case: `compute_advantages`, `get_symbols_overview`

### Constants
- UPPERCASE with underscores: `ADV_ESTIMATOR_REGISTRY`, `DISPATCH_MODE`

## Configuration Files

### Hydra YAML Configs
- Located in `verl/trainer/config/`
- After modification, **MUST** run: `bash scripts/generate_trainer_config.sh`
- This generates `_generated_*.yaml` files

### Config Dataclasses
- All Hydra configs convert to frozen dataclasses
- Support dict-like interface
- Defined using `@dataclass(frozen=True)` from `verl.base_config`

## Code Quality Pre-commit Hooks

### Enforced Checks (in order)
1. **ruff**: Linting with auto-fix
2. **ruff-format**: Code formatting
3. **mypy**: Type checking (strict modules only)
4. **autogen-trainer-cfg**: Config generation validation
5. **check-docstrings**: Docstring coverage verification
6. **check-license**: Apache 2.0 license header check
7. **compileall**: Python syntax validation

## Best Practices

### Code Organization
- Use the registry pattern for extensibility
- Leverage decorators for dispatch/collect patterns
- Follow the hybrid controller model for worker management

### Testing
- Separate CPU and GPU tests
- Use `pytest -s` for verbose output
- Organize tests by module: `tests/<module>/test_*.py`

### Documentation
- Keep docstrings up-to-date for required files
- Follow existing documentation patterns
- Include type hints for strict modules

### Configuration
- Use Hydra for hierarchical configs
- Keep config changes in sync with generated files
- Test config changes thoroughly

## Exclusions
- `tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`
- `scripts/legacy_model_merger.py`
