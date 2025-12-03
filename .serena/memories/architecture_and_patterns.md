# VERL Architecture and Key Patterns

## Core Design Pattern: Hybrid Controller Model

The fundamental architectural pattern enabling VERL's efficiency:

```
Ray Cluster
    ↓
RayPPOTrainer (orchestrator)
    ↓
WorkerGroups (distributed workers)
├── ActorRolloutRefWorker (hybrid: training + inference)
├── CriticWorker (value function)
└── RewardModelWorker (scoring)
    ↓
DataProto (unified data protocol)
```

**Key Insight**: Workers switch between `trainer_mode()` and `rollout_mode()` without reloading weights, enabling efficient memory utilization on the same GPU resources.

## Critical Abstractions

### 1. DataProto (`verl/protocol.py`)
Universal data container for inter-component communication.

**Structure**:
- `batch`: TensorDict for tensors with same batch size
- `non_tensor_batch`: Dict for numpy arrays
- `meta_info`: Metadata and metrics

**Operations**:
- Slicing, chunking, padding
- Serialization/deserialization
- Fold/unfold batch dimensions
- Union (combining DataProto instances)

**Usage Pattern**:
```python
# Chunking for distribution
chunks = data.chunk(n_workers)

# Padding for balanced batches
padded = pad_dataproto_to_divisor(data, divisor=8)

# Adding new fields
data = data.union(new_data)
```

### 2. Worker Groups (`verl/single_controller/`)
Ray-based distributed worker management.

**Components**:
- **ResourcePool**: Tracks GPU allocations across nodes
- **RayWorkerGroup**: Manages Ray actors with dispatch/collect pattern

**Dispatch Modes**:
- `ONE_TO_ALL`: Broadcast to all workers
- `GATHER`: Scatter batch, gather results
- `DP_COMPUTE`: Data parallel computation
- `MEGATRON_COMPUTE`: Megatron-style parallelism

**Usage Pattern**:
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    """Executed on all workers"""

@register(dispatch_mode=Dispatch.GATHER)
def compute_log_prob(self, data: DataProto):
    """Scatter batch, gather results"""
```

### 3. Registry Pattern
Extensible algorithm registration for easy customization.

**Registries**:
- `ADV_ESTIMATOR_REGISTRY`: GAE, GRPO, REINFORCE++, RLOO, CPO, etc.
- `POLICY_LOSS_REGISTRY`: Vanilla PPO, GSPO, GPG, KL coverage, etc.

**Usage Pattern**:
```python
@register_adv_est("my_estimator")
def my_advantage_estimator(returns, values, gamma=1.0):
    advantages = compute_my_advantages(returns, values)
    return advantages, {"metric/key": value}

@register_policy_loss("my_loss")
def my_loss(old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode, config):
    loss = compute_my_loss(...)
    return loss, {"actor/my_loss": loss.mean()}
```

### 4. Mode Switching (Hybrid Engine)
Efficient switching between training and inference modes.

**Pattern**:
```python
async with worker.rollout_mode():
    data = await rollout.generate_sequences(prompts)

async with worker.trainer_mode():
    metrics = update_policy(data)
```

## Training Flow

```
1. Rollout Phase
   └─ Switch workers to rollout_mode()
   └─ Generate sequences → DataProto

2. Reward Computation
   └─ Compute rewards and add to DataProto
   └─ Apply KL penalty (if enabled)

3. Advantage Estimation
   └─ Get estimator from registry (GAE/GRPO/CPO/etc.)
   └─ Compute advantages and returns

4. Policy Update
   └─ Switch workers to trainer_mode()
   └─ Actor: Compute policy loss, optimizer step
   └─ Critic: Compute value loss, optimizer step

5. Checkpoint & Metrics
   └─ Save model states
   └─ Log to tensorboard/wandb
```

## Training Backends

### FSDP (`verl/workers/fsdp_workers.py`)
- Fully Sharded Data Parallel with FSDP2 support
- Ulysses Sequence Parallel integration
- CPU offloading for params/optimizer
- LoRA support

### Megatron (`verl/workers/megatron_workers.py`)
- Tensor Parallel (TP), Pipeline Parallel (PP), Data Parallel (DP)
- Expert Model Parallel for MoE models
- Virtual Pipeline Parallel

## Inference Backends

### Rollout Engines (`verl/workers/rollout/`)
- **vLLM**: Sync/async modes
- **SGLang**: Sync/async with server adapter
- **HF Transformers**: Fallback implementation

### Weight Resharding (`verl/workers/sharding_manager/`)
Manages weight transformation between training (FSDP/Megatron) and inference (vLLM/SGLang) parallelism strategies.

## Configuration System

### Hydra-based Hierarchical Configs

**Main Templates**:
- `verl/trainer/config/ppo_trainer.yaml`: FSDP-based training
- `verl/trainer/config/ppo_megatron_trainer.yaml`: Megatron-based training

**Structure**:
```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  - critic@critic: dp_critic

algorithm:
  adv_estimator: gae  # or grpo, cpo, etc.
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
```

**Config Dataclasses**:
- All Hydra configs convert to frozen dataclasses
- Dict-like interface via `verl.base_config`
- Type-safe configuration

**Critical Hook**: 
`autogen-trainer-cfg` generates flattened YAML references from Hydra configs. Must run after config changes:
```bash
bash scripts/generate_trainer_config.sh
```

## Directory Structure

```
verl/
├── trainer/              # Training orchestration
│   ├── ppo/             # PPO trainer and core algorithms
│   └── config/          # Hydra configurations
├── workers/             # Distributed worker implementations
│   ├── fsdp_workers.py  # FSDP-based workers
│   ├── megatron_workers.py  # Megatron-based workers
│   ├── rollout/         # Rollout engines (vLLM, SGLang)
│   └── sharding_manager/  # Weight resharding
├── single_controller/   # Ray worker group management
├── protocol.py          # DataProto implementation
├── base_config.py       # Configuration base class
├── models/              # Model implementations
├── utils/               # Utility functions
└── experimental/        # Experimental features

tests/
├── trainer/             # Trainer tests
├── workers/             # Worker tests
├── special_sanity/      # Sanity checks
├── special_distributed/ # Multi-GPU tests
└── special_e2e/         # End-to-end tests

examples/
├── ppo_trainer/         # PPO examples
├── grpo_trainer/        # GRPO examples
├── sft/                 # Supervised fine-tuning
└── sglang_multiturn/    # Multi-turn examples

recipe/
├── dapo/                # DAPO recipe
├── gspo/                # GSPO recipe
├── prime/               # PRIME recipe
└── sppo/                # SPPO recipe
```

## Key Extension Points

### Adding New RL Algorithms
1. Register advantage estimator in `verl/trainer/ppo/core_algos.py`
2. Register policy loss (if custom)
3. Update configuration
4. Add tests

### Custom Reward Functions
Create reward function in separate file:
```python
def compute_score(data: DataProto) -> torch.Tensor:
    responses = data.batch["responses"]
    return your_scoring_logic(responses)
```

Reference in config:
```yaml
custom_reward_function:
  path: /path/to/reward.py
  name: compute_score
```

## Important Files Reference

**Core Orchestration**:
- `verl/trainer/ppo/ray_trainer.py`: Main training loop (2000+ lines)
- `verl/trainer/ppo/core_algos.py`: Algorithm implementations (1800+ lines)

**Worker Implementations**:
- `verl/workers/fsdp_workers.py`: FSDP ActorRolloutRefWorker
- `verl/workers/megatron_workers.py`: Megatron ActorRolloutRefWorker

**Data & Protocol**:
- `verl/protocol.py`: DataProto implementation (1241 lines)
- `verl/base_config.py`: Configuration base class

**Ray Integration**:
- `verl/single_controller/ray/base.py`: RayWorkerGroup, ResourcePool
- `verl/single_controller/base/decorator.py`: Dispatch/collect decorators

## Design Principles

1. **Hybrid Controller Model**: Enable efficient mode switching
2. **Modularity**: Decouple computation and data dependencies
3. **Extensibility**: Registry pattern for algorithms
4. **Flexibility**: Support multiple backends and parallelism strategies
5. **Efficiency**: Minimize communication overhead and memory usage
