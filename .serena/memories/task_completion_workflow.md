# VERL Task Completion Workflow

## Standard Development Workflow

When completing any task, follow these steps in order:

### 1. Make Code Changes
- Implement the requested feature, fix, or modification
- Follow code style and conventions
- Add/update docstrings for required files
- Ensure type hints are present for strict modules

### 2. Run Pre-commit Checks (CRITICAL)
```bash
pre-commit run --all-files
```

This runs (in order):
1. Ruff linting with auto-fix
2. Ruff formatting
3. Mypy type checking (strict modules only)
4. Config generation validation
5. Docstring coverage check
6. License header check
7. Python syntax validation

**IMPORTANT**: All checks must pass before committing.

### 3. Regenerate Configs (If Modified)
If you modified any Hydra configuration files in `verl/trainer/config/`:
```bash
bash scripts/generate_trainer_config.sh
```

This is **CRITICAL** and will be validated by the pre-commit hook.

### 4. Run Relevant Tests Locally

#### For Code Changes
```bash
# If CPU-compatible
pytest -s tests/*_on_cpu.py

# If GPU-required
pytest -s --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' tests/

# For specific modules
pytest tests/trainer/
pytest tests/workers/
pytest tests/utils/
```

#### For Distributed Changes (2+ GPUs required)
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py
```

### 5. Commit Changes
```bash
git add .
git commit -m "descriptive message"
```

The pre-commit hooks will run automatically. If they fail:
- Fix the issues
- Re-stage the files: `git add .`
- Commit again

### 6. CI Automatic Validation
After push, CI will run the full test suite:
- Sanity checks (always)
- Pre-commit validation (always)
- Documentation build (always)
- GPU/CPU unit tests (path-filtered)
- E2E tests (path-filtered)
- Backend-specific tests (vllm, sglang, model tests)

## Specific Task Workflows

### Adding New RL Algorithms

1. **Register Advantage Estimator** (if needed)
   - Add to `verl/trainer/ppo/core_algos.py`
   - Use `@register_adv_est("name")` decorator
   
2. **Register Policy Loss** (if needed)
   - Add to `verl/trainer/ppo/core_algos.py`
   - Use `@register_policy_loss("name")` decorator

3. **Update Configuration**
   - Modify `verl/trainer/config/ppo_trainer.yaml` or similar
   - Run `bash scripts/generate_trainer_config.sh`

4. **Add Tests**
   - Create test in appropriate `tests/` subdirectory
   - Include both CPU and GPU tests if applicable

5. **Follow standard workflow** (steps 1-6 above)

### Adding New Worker Types

1. **Implement Worker Class**
   - In `verl/workers/` directory
   - Inherit from appropriate base class
   - Use dispatch/collect decorators

2. **Add Configuration Support**
   - Update Hydra configs
   - Regenerate configs

3. **Add Tests**
   - Unit tests for worker functionality
   - Integration tests with Ray

4. **Follow standard workflow**

### Modifying Core Components

For changes to:
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/protocol.py`
- Worker implementations

1. **Ensure backward compatibility** (if possible)
2. **Update docstrings** (required for these files)
3. **Add type hints** (if in strict modules)
4. **Update tests** to cover changes
5. **Run full test suite**
6. **Follow standard workflow**

## Pre-commit Failure Resolution

### Ruff Linting Failures
```bash
ruff check --fix .
git add .
```

### Ruff Formatting Failures
```bash
ruff format .
git add .
```

### Mypy Type Checking Failures
- Fix type hints in strict modules
- Ensure imports are correct
- Add `# type: ignore` only if absolutely necessary

### Config Generation Failures
```bash
bash scripts/generate_trainer_config.sh
git add verl/trainer/config/_generated_*.yaml
```

### Docstring Coverage Failures
- Add docstrings to public functions/classes in required files
- Follow existing docstring format

### License Header Failures
- Add Apache 2.0 license header to new files
- Check `tests/special_sanity/check_license.py` for format

## Quick Commands Reference

```bash
# Complete pre-commit check
pre-commit run --all-files

# Fix linting + formatting
ruff check --fix . && ruff format .

# Regenerate configs
bash scripts/generate_trainer_config.sh

# Run CPU tests
pytest -s tests/*_on_cpu.py

# Run GPU tests
pytest -s --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' tests/
```

## Important Reminders

- ✅ **ALWAYS** run pre-commit before committing
- ✅ **ALWAYS** regenerate configs after Hydra changes
- ✅ **ALWAYS** add docstrings to required files
- ✅ **ALWAYS** run relevant tests locally
- ❌ **NEVER** skip pre-commit hooks
- ❌ **NEVER** commit without testing
- ❌ **NEVER** modify configs without regenerating
