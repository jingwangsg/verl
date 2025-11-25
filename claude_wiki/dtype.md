# dtype (数据类型) 配置指南

## 概述

`dtype` (数据类型) 参数是 veRL 中的关键配置,用于控制模型训练和推理过程中使用的数值精度。本指南详细解释 dtype 参数的作用、在配置系统中的位置、为什么 dtype 对齐至关重要,以及如何针对不同场景正确配置 dtype。

**核心概念:**
- **dtype**: 混合精度训练使用的精度(梯度、激活值)
- **model_dtype**: 模型参数存储和初始化使用的精度
- **对齐 (Alignment)**: 确保训练组件(Actor/Critic)和推理组件(Rollout)使用一致的 dtype

---

## 目录

- [快速开始](#快速开始)
- [理解 dtype 参数](#理解-dtype-参数)
- [generation.yaml vs rollout.yaml 的区别](#generationyaml-vs-rolloutyaml-的区别)
- [dtype 出现的位置](#dtype-出现的位置)
- [关键对齐规则](#关键对齐规则)
- [支持的数据类型](#支持的数据类型)
- [GPU 特定注意事项](#gpu-特定注意事项)
- [配置示例](#配置示例)
- [命令行配置](#命令行配置)
- [常见陷阱](#常见陷阱)
- [故障排查](#故障排查)
- [最佳实践](#最佳实践)
- [相关文件](#相关文件)

---

## 快速开始

### 推荐的默认配置 (bfloat16)

适用于现代 GPU (H100, A100, A800):

```yaml
# FSDP 后端 (默认)
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16        # 训练精度
      model_dtype: fp32      # 参数初始化精度

  ref:
    fsdp_config:
      dtype: bfloat16
      model_dtype: fp32

  rollout:
    dtype: bfloat16          # 推理精度 - 必须与 actor dtype 匹配!

critic:
  model:
    fsdp_config:
      dtype: bfloat16
      model_dtype: fp32
```

### 旧款 GPU 的 FP16 配置

适用于 V100 或不支持 bfloat16 的 GPU:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: float16
      model_dtype: fp32

  rollout:
    dtype: float16           # 必须与 actor dtype 匹配!
```

---

## 理解 dtype 参数

### dtype vs model_dtype

veRL 在 FSDP 配置中使用**两个独立的 dtype 参数**:

| 参数 | 用途 | 默认值 | 内存影响 | 设置时机 |
|------|------|--------|----------|---------|
| `dtype` | 混合精度训练: 梯度、激活值、前向/反向传播 | `bfloat16` | 中等 | 训练期间 |
| `model_dtype` | 模型参数存储和初始化 | `fp32` | 高 | 模型加载时 |

**工作流程示例:**

1. **模型初始化** (`model_dtype=fp32`):
   - 模型参数以 FP32 加载/初始化
   - 在初始化过程中提供数值稳定性

2. **训练** (`dtype=bfloat16`):
   - 参数被转换为 bfloat16 用于前向/反向传播
   - 梯度以 bfloat16 计算
   - 优化器状态可能在内部使用更高精度

3. **推理** (`rollout.dtype=bfloat16`):
   - 模型权重从 Actor 同步到 Rollout 引擎
   - 推理引擎 (vLLM/SGLang) 以相同的 dtype 运行
   - **关键**: 必须与训练 dtype 匹配以避免转换开销

### 为什么需要两个 dtype 参数?

**关注点分离:**
- `model_dtype`: 确保稳定的初始化,特别是对大型模型
- `dtype`: 启用混合精度训练以提高速度和内存效率

**常见配置:**

| 使用场景 | model_dtype | dtype | 说明 |
|---------|-------------|-------|------|
| 最大稳定性 | fp32 | bfloat16 | 默认,推荐 |
| 最大内存节省 | bfloat16 | bfloat16 | 减少约 25% 内存 |
| 调试 | fp32 | fp32 | 全精度,速度慢 |
| 旧款 GPU | fp32 | float16 | V100 兼容性 |

---

## generation.yaml vs rollout.yaml 的区别

### 概述

veRL 中存在**两个独立的配置文件**都包含 rollout.dtype 参数,但它们用于**不同的场景**,容易产生混淆。

### 配置文件对比

| 方面 | generation.yaml | rollout.yaml |
|------|----------------|--------------|
| **文件位置** | `verl/trainer/config/generation.yaml` | `verl/trainer/config/rollout/rollout.yaml` |
| **使用入口** | `main_generation.py` (纯推理) | `main_ppo.py` (训练) |
| **用途** | 独立推理配置 | 训练中的 Rollout 组件配置 |
| **何时加载** | 运行纯推理任务时 | 作为 `ppo_trainer.yaml` 的一部分导入 |
| **包含内容** | 完整应用配置 (trainer, data, model, rollout, actor, ray) | 仅 rollout 特定设置 |
| **独立性** | 是,完全独立 | 否,是训练配置层次的一部分 |

### generation.yaml 的作用

**文件:** `verl/verl/trainer/config/generation.yaml`

**用途:** 专门用于**纯推理模式** (inference-only),不用于训练。

**使用方式:**

```python
# verl/trainer/main_generation.py
@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)
```

**应用场景:**
- 生成合成数据
- 在测试集上进行推理
- 批量文本生成任务

**示例脚本:**
```bash
python3 -m verl.trainer.main_generation \
    model.path=/path/to/model \
    rollout.dtype=bfloat16 \
    data.input_path=input.parquet \
    data.output_path=output.parquet
```

### rollout.yaml 的作用

**文件:** `verl/verl/trainer/config/rollout/rollout.yaml`

**用途:** 训练过程中 Rollout 组件的配置,通过 `ppo_trainer.yaml` 的 defaults 导入。

**配置层次:**

```yaml
# ppo_trainer.yaml
defaults:
  - rollout@actor_rollout_ref.rollout: rollout  # 加载 rollout/rollout.yaml
```

**应用场景:**
- PPO/GRPO/其他 RL 训练
- 训练中的轨迹采样
- 与训练紧密集成

### generation.yaml 中的 dtype 需要与训练对齐吗?

**答案: 取决于使用场景**

#### 场景 A: 独立推理 (与训练无关)

如果使用 `main_generation.py` 进行**独立推理** (例如生成合成数据、测试集推理):

- ❌ **不需要**与任何训练配置匹配
- 根据**checkpoint 的原生 dtype** 设置
- 如果加载 bfloat16 训练的 checkpoint,使用 `rollout.dtype: bfloat16`
- 如果加载 float16 训练的 checkpoint,使用 `rollout.dtype: float16`

#### 场景 B: 为训练生成数据

如果使用 `main_generation.py` 生成稍后将用于训练的数据:

- ❌ **仍然不严格要求** dtype 对齐
- 生成的文本输出以字符串形式保存 (在 parquet 中)
- 加载用于训练时,将以训练 dtype 重新分词和处理

**但为了一致性和可重现性:**
- 建议使用与训练设置相同的 dtype
- 这确保生成行为与训练 rollout 期间的行为一致

### 何时需要修改 generation.yaml 的 dtype?

#### ✅ 需要修改的情况:

1. **GPU 不支持 bfloat16** (如 V100):
   ```bash
   python3 -m verl.trainer.main_generation \
       rollout.dtype=float16
   ```

2. **加载特定 dtype 的 checkpoint**:
   - 匹配 checkpoint 的原生 dtype
   - float16 训练的模型 → `rollout.dtype=float16`
   - bfloat16 训练的模型 → `rollout.dtype=bfloat16`

3. **为了与训练保持一致** (可选):
   - 希望推理行为完全匹配训练 rollout
   - 不是必需的,但对调试有帮助

#### ❌ 不需要修改的情况:

- 修改训练 dtype (在 `ppo_trainer.yaml` 中) 时
- 使用 dtype 覆盖运行 `main_ppo.py` 时
- 不要假设 generation.yaml 会影响训练

### 关于 "should align with FSDP" 注释的说明

`generation.yaml` 中的注释:
```yaml
dtype: bfloat16 # should align with FSDP
```

这个注释在纯推理场景下有些**误导性**。它可能:

1. **从 rollout.yaml 复制而来** (在训练中确实需要与 FSDP 对齐)
2. **指代同一文件中的 actor FSDP 配置** (但在推理模式下 actor 不用于训练)
3. **更准确的说法应该是:** "should match the checkpoint's native dtype"

在 `rollout/rollout.yaml` (用于训练) 中,这个注释是**正确且关键的**:
```yaml
# Rollout 模型参数类型。与 actor 模型的 FSDP/Megatron 类型对齐。
dtype: bfloat16
```

### 配置关系图

```
训练流程 (main_ppo.py)
├── ppo_trainer.yaml (主配置)
│   └── rollout/rollout.yaml (导入)
│       └── dtype: bfloat16  ← 必须与 actor.dtype 对齐!
│
推理流程 (main_generation.py)
└── generation.yaml (独立配置)
    └── rollout:
        └── dtype: bfloat16  ← 独立设置,基于 GPU 能力或 checkpoint
```

### 实际建议

| 使用场景 | 关注的配置文件 | dtype 设置原则 |
|---------|---------------|---------------|
| **训练** (`main_ppo.py`) | `ppo_trainer.yaml` + `rollout/rollout.yaml` | 确保 `actor.dtype` = `rollout.dtype` |
| **纯推理** (`main_generation.py`) | `generation.yaml` | 根据 GPU 能力或 checkpoint dtype |
| **两者都用** | 分别维护 | 训练和推理配置独立 |

### 总结

- `generation.yaml` 是**完全独立**的推理配置
- 它**从不**在训练 (`main_ppo.py`) 中使用
- 修改训练 dtype 时**不需要**修改 `generation.yaml`
- 两个使用场景 (纯推理 vs 训练中的 rollout) 有**独立的配置文件和入口点**

---

## dtype 出现的位置

### 配置层次结构

dtype 参数在 veRL 配置系统中出现在**三个主要位置**:

```
ppo_trainer.yaml (根配置)
│
├── actor_rollout_ref/
│   ├── actor/                    # 策略网络 (训练)
│   │   ├── fsdp_config/
│   │   │   ├── dtype: bfloat16              [1] 训练精度
│   │   │   └── model_dtype: fp32            [2] 初始化精度
│   │   └── megatron/
│   │       └── dtype: bfloat16              [1] 训练精度 (Megatron)
│   │
│   ├── ref/                      # 参考模型 (KL 散度)
│   │   ├── fsdp_config/
│   │   │   ├── dtype: bfloat16              [3] Ref 训练精度
│   │   │   └── model_dtype: fp32            [4] Ref 初始化精度
│   │   └── megatron/
│   │       └── dtype: bfloat16              [3] Ref 精度 (Megatron)
│   │
│   └── rollout/                  # 推理引擎 (生成)
│       └── dtype: bfloat16                  [5] 推理精度 ⚠️ 关键
│
└── critic/                       # 价值网络 (仅 PPO)
    ├── model/
    │   └── fsdp_config/
    │       ├── dtype: bfloat16              [6] Critic 训练精度
    │       └── model_dtype: fp32            [7] Critic 初始化精度
    └── megatron/
        └── dtype: bfloat16                  [6] Critic 精度 (Megatron)
```

### 参数位置表

| 组件 | FSDP 路径 | Megatron 路径 | 默认值 | 必需 |
|------|----------|--------------|--------|------|
| **Actor 训练** | `actor_rollout_ref.actor.fsdp_config.dtype` | `actor_rollout_ref.actor.megatron.dtype` | `bfloat16` | ✅ 是 |
| **Actor 初始化** | `actor_rollout_ref.actor.fsdp_config.model_dtype` | N/A | `fp32` | 可选 |
| **参考模型** | `actor_rollout_ref.ref.fsdp_config.dtype` | `actor_rollout_ref.ref.megatron.dtype` | `bfloat16` | ✅ 是 (如果使用 KL) |
| **Rollout 推理** | `actor_rollout_ref.rollout.dtype` | `actor_rollout_ref.rollout.dtype` | `bfloat16` | ✅ **关键** |
| **Critic 训练** | `critic.model.fsdp_config.dtype` | `critic.megatron.dtype` | `bfloat16` | ✅ 是 (仅 PPO) |
| **Critic 初始化** | `critic.model.fsdp_config.model_dtype` | N/A | `fp32` | 可选 |

### 文件位置

**配置文件:**
- FSDP 引擎: `verl/trainer/config/engine/fsdp.yaml` (第 56, 57 行)
- Megatron 引擎: `verl/trainer/config/engine/megatron.yaml` (第 81 行)
- Rollout 配置: `verl/trainer/config/rollout/rollout.yaml` (第 29 行)
- Generation 配置: `verl/trainer/config/generation.yaml` (第 26 行)

**源代码:**
- FSDP Workers: `verl/workers/fsdp_workers.py` (第 461-471 行: dtype 设置, 第 704-721 行: 权重同步)
- vLLM Rollout: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` (第 236, 594 行)
- SGLang Rollout: `verl/workers/rollout/sglang_rollout/sglang_rollout.py` (第 447 行)
- 配置数据类: `verl/workers/config/rollout.py` (第 132 行), `verl/workers/config/engine.py` (第 49, 75, 103, 122 行)

---

## 关键对齐规则

### ⚠️ **黄金法则**: Actor dtype 必须等于 Rollout dtype

```python
# 这个条件必须为真:
actor_rollout_ref.actor.fsdp_config.dtype == actor_rollout_ref.rollout.dtype
```

### 为什么对齐至关重要

在 veRL 的 **Hybrid Engine 架构**中,训练和推理共享同一组 workers。工作流程是:

```
训练阶段                       推理阶段
┌─────────────────┐           ┌─────────────────┐
│  Actor (FSDP)   │           │ Rollout (vLLM)  │
│  dtype=bfloat16 │ ────────► │ dtype=bfloat16  │
│                 │  权重同步  │                 │
└─────────────────┘           └─────────────────┘
```

**权重同步过程中发生的事情:**

1. **FSDP 存储参数** 以 `param_dtype` (例如 bfloat16)
2. **权重被传输** 到推理引擎,不进行 dtype 转换
3. **推理引擎加载权重** 期望相同的 dtype

**代码参考** (`verl/workers/fsdp_workers.py:704-721`):

```python
# 从 FSDP 提取参数 (保留原始 dtype)
device = get_device_id()
per_tensor_param = (
    (name, param.to(device, non_blocking=True).full_tensor()
     if isinstance(param, DTensor) else param)
    for name, param in params.items()
)

# 更新 rollout 引擎参数 (无 dtype 转换)
await self.rollout.update_weights(
    per_tensor_param,
    peft_config=peft_config,
    base_sync_done=self.base_sync_done
)
```

### 不对齐的后果

| 场景 | Actor dtype | Rollout dtype | 结果 |
|------|-------------|---------------|------|
| ✅ **对齐** | `bfloat16` | `bfloat16` | 高效,无转换 |
| ❌ **不对齐** | `bfloat16` | `float16` | **运行时错误**: dtype 不匹配 |
| ❌ **不对齐** | `bfloat16` | `fp32` | **运行时错误**: dtype 不匹配 |
| ⚠️ **隐式转换** | `float16` | `bfloat16` | 错误或静默精度损失 |

**错误示例:**

```
RuntimeError: Expected all tensors to be on the same device and dtype,
but found at least two dtypes: BFloat16 and Float!
```

### 对齐检查清单

运行训练前,验证:

- [ ] `actor_rollout_ref.actor.fsdp_config.dtype` = `actor_rollout_ref.rollout.dtype`
- [ ] `actor_rollout_ref.ref.fsdp_config.dtype` = `actor_rollout_ref.rollout.dtype` (如果使用 KL)
- [ ] `critic.model.fsdp_config.dtype` 与 actor dtype 匹配 (仅 PPO)
- [ ] 对于 Megatron: 所有 `.megatron.dtype` 值匹配 `rollout.dtype`

---

## 支持的数据类型

### dtype 对比表

| 数据类型 | 精度 | 范围 | 内存 (每个参数) | GPU 支持 | 推荐用途 |
|---------|------|------|----------------|---------|---------|
| **bfloat16** | 7位尾数, 8位指数 | ~10⁻³⁸ 到 10³⁸ | 2 字节 | Ampere+ (A100, H100) | ✅ **现代 GPU 的默认选择** |
| **float16** | 10位尾数, 5位指数 | ~10⁻⁸ 到 65504 | 2 字节 | 大多数 GPU (V100+) | 旧款 GPU 兼容性 |
| **fp32** | 23位尾数, 8位指数 | ~10⁻³⁸ 到 10³⁸ | 4 字节 | 所有 GPU | 调试, model_dtype |

### 详细特性

#### bfloat16 (Brain Floating Point 16)

**优点:**
- ✅ 与 fp32 相同的指数范围 (8位) → 数值稳定性好
- ✅ 现代 GPU 上的硬件加速 (Tensor Cores)
- ✅ 在 LLM 训练中相比 fp32 性能下降很小
- ✅ 不需要 loss scaling 或特殊处理
- ✅ 相比 fp32 内存减少 50%

**缺点:**
- ❌ 精度较低 (7位尾数 vs fp32 的 23位)
- ❌ 需要 Ampere 或更新的 GPU 架构
- ❌ V100 或更早的 GPU 不支持

**何时使用:**
- H100, H800, A100, A800 GPU
- 训练 ≥7B 参数的模型
- 生产 RLHF 工作负载

#### float16 (半精度)

**优点:**
- ✅ 广泛的 GPU 支持 (V100, T4 等)
- ✅ 相比 fp32 内存减少 50%
- ✅ 大多数现代 GPU 上的硬件加速

**缺点:**
- ❌ 指数范围有限 (5位) → 容易发生下溢/溢出
- ❌ 通常需要 loss scaling 才能稳定训练
- ❌ 在深度网络中可能导致数值不稳定
- ❌ 不太适合超大型模型

**何时使用:**
- V100 或其他 Ampere 之前的 GPU
- 较小的模型 (<7B 参数)
- 当 bfloat16 不可用时

#### fp32 (全精度)

**优点:**
- ✅ 最高的数值精度和稳定性
- ✅ 通用 GPU 支持
- ✅ 在正常场景下无下溢/溢出风险

**缺点:**
- ❌ 相比 bfloat16/float16 内存使用 2倍
- ❌ 训练速度慢 (无 Tensor Core 加速)
- ❌ 在消费级 GPU 上对 >70B 模型不可行

**何时使用:**
- 用于稳定初始化的 `model_dtype`
- 调试数值问题
- GPU 内存充足的小型模型

### 性能对比

A100 80GB 上 7B 模型的大致指标:

| dtype | 内存 (训练) | 吞吐量 | 数值稳定性 |
|-------|------------|--------|-----------|
| fp32 | ~56 GB | 1x (基线) | 优秀 |
| bfloat16 | ~28 GB | ~3x | 非常好 |
| float16 | ~28 GB | ~3x | 好 (需要 loss scaling) |

---

## GPU 特定注意事项

### H100 / H800 (Hopper 架构)

**推荐配置:**

```yaml
actor_rollout_ref.actor.fsdp_config.dtype: bfloat16
actor_rollout_ref.actor.fsdp_config.model_dtype: bfloat16  # 也可以使用 fp32
actor_rollout_ref.rollout.dtype: bfloat16
```

**注意事项:**
- ✅ 完全的 bfloat16 硬件加速 (第4代 Tensor Cores)
- ✅ 可以使用 `model_dtype: bfloat16` 实现最大内存节省
- ✅ 支持 Transformer Engine 的自动 FP8 混合精度 (实验性)
- ⭐ 最佳选择: 对大型模型 (70B+) 使用 `dtype=bfloat16`, `model_dtype=bfloat16`

### A100 / A800 (Ampere 架构)

**推荐配置:**

```yaml
actor_rollout_ref.actor.fsdp_config.dtype: bfloat16
actor_rollout_ref.actor.fsdp_config.model_dtype: fp32
actor_rollout_ref.rollout.dtype: bfloat16
```

**注意事项:**
- ✅ 完全支持 bfloat16 (第3代 Tensor Cores)
- ✅ 默认配置效果很好
- ⚠️ 对于 40GB 版本,大型模型考虑使用 `model_dtype: bfloat16`

### V100 及更旧的 GPU

**推荐配置:**

```yaml
actor_rollout_ref.actor.fsdp_config.dtype: float16
actor_rollout_ref.actor.fsdp_config.model_dtype: fp32
actor_rollout_ref.rollout.dtype: float16
```

**注意事项:**
- ❌ 无 bfloat16 硬件支持
- ✅ 支持 float16 (FP16) 和 Tensor Cores
- ⚠️ 可能需要 loss scaling 才能稳定训练
- ⚠️ 注意梯度溢出/下溢问题
- 考虑梯度裁剪: `actor_rollout_ref.actor.optim.clip_grad=1.0`

### NPU / Ascend (华为)

**推荐配置:**

```yaml
actor_rollout_ref.actor.fsdp_config.dtype: bfloat16  # 或 float16,取决于 NPU 型号
actor_rollout_ref.rollout.dtype: bfloat16
```

**注意事项:**
- 查看 NPU 文档了解支持的 dtype
- Ascend 910B 支持 bfloat16
- 某些算子可能回退到 fp32

### 多 GPU / 多节点注意事项

**网络通信:**
- 梯度 all-reduce 使用 `reduce_dtype` (默认: fp32),无论 `dtype` 是什么
- 这由 FSDP 的 `MixedPrecision.reduce_dtype` 控制
- 确保分布式训练中的数值稳定性

**内存压力:**
- 在异构 GPU 集群上,使用最低公分母 dtype
- 例如: 混合 A100 + V100 → 使用 float16

---

## 配置示例

### 示例 1: 完整 bfloat16 配置 (FSDP)

现代 GPU 的最大内存节省:

```yaml
# verl/trainer/config/ppo_trainer.yaml 覆盖
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16
      model_dtype: bfloat16        # ← 从默认的 fp32 更改

  ref:
    fsdp_config:
      dtype: bfloat16
      model_dtype: bfloat16

  rollout:
    dtype: bfloat16
    name: vllm

critic:
  model:
    fsdp_config:
      dtype: bfloat16
      model_dtype: bfloat16
```

**使用场景:** H100/A100 上的大型模型 (70B+),内存预算紧张

### 示例 2: V100 GPU 的 FP16

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: float16
      model_dtype: fp32             # 保持 fp32 初始化以确保稳定性
    optim:
      clip_grad: 1.0                # 推荐梯度裁剪

  ref:
    fsdp_config:
      dtype: float16
      model_dtype: fp32

  rollout:
    dtype: float16
    name: vllm

critic:
  model:
    fsdp_config:
      dtype: float16
      model_dtype: fp32
```

**使用场景:** V100 GPU 或不支持 bfloat16 的 GPU

### 示例 3: Megatron 后端与 bfloat16

```yaml
# 使用 ppo_megatron_trainer.yaml 配置
actor_rollout_ref:
  actor:
    megatron:
      dtype: bfloat16

  ref:
    megatron:
      dtype: bfloat16

  rollout:
    dtype: bfloat16
    name: vllm

critic:
  megatron:
    dtype: bfloat16
```

**使用场景:** 超大型模型 (235B+), 带有专家并行的 MoE 模型

### 示例 4: GRPO (无 Critic)

```yaml
algorithm:
  adv_estimator: grpo               # GRPO 不使用 critic

actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16
      model_dtype: fp32

  ref:
    fsdp_config:
      dtype: bfloat16
      model_dtype: fp32

  rollout:
    dtype: bfloat16
    n: 8                            # GRPO 采样多个响应

# 不需要 critic 配置
```

**使用场景:** GRPO 算法,更简单的配置

### 示例 5: 混合精度 (稳定初始化 + 快速训练)

```yaml
# 默认推荐配置
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16               # 快速训练
      model_dtype: fp32             # 稳定初始化
      mixed_precision:
        param_dtype: bfloat16       # 显式混合精度
        reduce_dtype: fp32          # 高精度梯度归约
        buffer_dtype: fp32          # 高精度缓冲区

  rollout:
    dtype: bfloat16
```

**使用场景:** 生产工作负载,最大稳定性 + 性能平衡

### 示例 6: SGLang 与多轮对话

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16

  rollout:
    dtype: bfloat16
    name: sglang                    # SGLang 用于多轮对话
    multi_turn:
      enable: true
      tool_config_path: /path/to/tool_config.yaml

# dtype 仍然必须对齐以进行权重同步
```

**使用场景:** 使用 SGLang 的多轮工具调用工作流

---

## 命令行配置

### 基本 Hydra 覆盖语法

veRL 使用 **Hydra** 进行配置。使用点号语法覆盖任何嵌套参数:

```bash
python3 -m verl.trainer.main_ppo \
    config.subconfig.param=value \
    other.nested.param=value
```

### 通过命令行设置 dtype

#### FSDP 后端 (默认)

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.ref.fsdp_config.dtype=float16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=float16 \
    critic.model.fsdp_config.dtype=float16 \
    critic.model.fsdp_config.model_dtype=float16 \
    # ... 其他参数
```

#### Megatron 后端

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    actor_rollout_ref.actor.megatron.dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.ref.megatron.dtype=float16 \
    critic.megatron.dtype=float16 \
    # ... 其他参数
```

### Shell 脚本示例

#### 示例脚本: FSDP 与 FP16

创建文件 `run_my_training_fp16.sh`:

```bash
#!/bin/bash
set -x

# 定义 dtype 作为变量以确保一致性
DTYPE=float16
MODEL_DTYPE=fp32

# 数据路径
TRAIN_FILES="['$HOME/data/train.parquet']"
TEST_FILES="['$HOME/data/test.parquet']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    \
    `# Actor 配置` \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    \
    `# Rollout 配置 - 必须与 actor dtype 匹配` \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    \
    `# 参考模型配置` \
    actor_rollout_ref.ref.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    \
    `# Trainer 配置` \
    trainer.total_epochs=15 \
    trainer.n_gpus_per_node=8 \
    trainer.logger='["console","wandb"]' \
    $@  # 允许额外参数
```

使其可执行:

```bash
chmod +x run_my_training_fp16.sh
./run_my_training_fp16.sh
```

#### 示例脚本: 动态 dtype 选择

```bash
#!/bin/bash

# 自动检测 GPU 并设置 dtype
if nvidia-smi | grep -q "H100\|A100"; then
    DTYPE=bfloat16
    echo "检测到 H100/A100, 使用 bfloat16"
elif nvidia-smi | grep -q "V100"; then
    DTYPE=float16
    echo "检测到 V100, 使用 float16"
else
    DTYPE=bfloat16
    echo "未知 GPU, 默认使用 bfloat16"
fi

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    # ... 其他参数
```

### veRL 仓库中的真实示例

**文件:** `recipe/flowrl/run_flowrl_qwen2.5_7b_fp16.sh`

```bash
python -m verl.trainer.main_ppo \
    --config-name flowrl_trainer \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16 \
    # ... (其他配置)
```

这个脚本展示了从 bfloat16 切换到 float16 所需的**最小覆盖集**。

### 使用环境变量

```bash
#!/bin/bash

# 通过环境变量设置 dtype
export VERL_DTYPE=${VERL_DTYPE:-bfloat16}  # 默认为 bfloat16

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=$VERL_DTYPE \
    actor_rollout_ref.rollout.dtype=$VERL_DTYPE \
    actor_rollout_ref.ref.fsdp_config.dtype=$VERL_DTYPE
```

使用自定义 dtype 运行:

```bash
VERL_DTYPE=float16 ./train.sh
```

### Hydra 提示

**使用 `+` 前缀添加新参数:**

```bash
# 如果参数在默认配置中不存在
+actor_rollout_ref.actor.fsdp_config.custom_param=value
```

**使用 `~` 前缀删除参数:**

```bash
~actor_rollout_ref.actor.fsdp_config.some_param
```

**使用 YAML 多行配置:**

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    actor_rollout_ref.actor.fsdp_config="$(cat <<EOF
dtype: float16
model_dtype: fp32
param_offload: false
EOF
)"
```

---

## 常见陷阱

### 陷阱 1: Actor 和 Rollout dtype 不对齐

**问题:**

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16        # ← 训练使用 bfloat16

  rollout:
    dtype: float16           # ← 错误: Rollout 使用 float16!
```

**症状:**

```
RuntimeError: Expected all tensors to be on the same device and dtype,
but found at least two dtypes: BFloat16 and Float!
```

或训练和生成之间的静默数值不一致。

**解决方案:**

确保 `actor.fsdp_config.dtype` == `rollout.dtype`:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16
  rollout:
    dtype: bfloat16          # ✅ 与 actor 匹配
```

**如何验证:**

在 Hydra 处理后检查合并的配置:

```bash
python3 -m verl.trainer.main_ppo --cfg job 2>&1 | grep -A5 "dtype"
```

### 陷阱 2: 在不支持的 GPU 上使用 bfloat16

**问题:**

```yaml
# 在 V100 GPU 上
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16        # ← 错误: V100 不支持 bfloat16
```

**症状:**

```
RuntimeError: "addmm_impl_cpu_" not implemented for 'BFloat16'
```

或训练非常慢(回退到 CPU 模拟)。

**解决方案:**

在选择 dtype 之前检查 GPU 架构:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

**GPU dtype 支持矩阵:**

| GPU | bfloat16 | float16 | fp32 |
|-----|----------|---------|------|
| H100 | ✅ 原生 | ✅ 原生 | ✅ |
| A100 | ✅ 原生 | ✅ 原生 | ✅ |
| V100 | ❌ 否 | ✅ 原生 | ✅ |
| T4 | ❌ 否 | ✅ 原生 | ✅ |

对于 V100: 使用 `dtype: float16`

### 陷阱 3: 混淆 model_dtype 和 dtype

**问题:**

```bash
# 用户只设置 model_dtype,认为它控制训练精度
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float16 \
    # ← 忘记设置 dtype!
```

**结果:**

- 训练仍然使用默认的 `dtype=bfloat16`
- 模型以 float16 初始化,然后转换为 bfloat16
- 不必要的 dtype 转换

**解决方案:**

**同时**显式设置 dtype 和 model_dtype:

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float16
```

**经验法则:**
- 最大内存节省: `dtype` = `model_dtype` = `bfloat16`/`float16`
- 最大稳定性: `dtype` = `bfloat16`/`float16`, `model_dtype` = `fp32`

### 陷阱 4: 忘记参考模型 dtype

**问题:**

```bash
# 用户设置了 actor 和 rollout dtype,但忘记了 ref
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16
    # ← 忘记 ref.fsdp_config.dtype!
```

**结果:**

- 参考模型仍然使用默认的 `dtype=bfloat16`
- KL 散度计算涉及混合 dtype
- 潜在的数值不一致

**解决方案:**

使用 KL 散度时始终设置 ref dtype:

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.ref.fsdp_config.dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16
```

### 陷阱 5: Critic dtype 不一致 (PPO)

**问题:**

```yaml
# PPO 训练
actor_rollout_ref.actor.fsdp_config.dtype: bfloat16
critic.model.fsdp_config.dtype: float16      # ← 与 actor 不同!
```

**症状:**

- 由于 dtype 转换导致训练速度变慢
- 优势计算中可能存在数值精度不匹配

**解决方案:**

保持 critic dtype 与 actor 一致:

```yaml
actor_rollout_ref.actor.fsdp_config.dtype: bfloat16
critic.model.fsdp_config.dtype: bfloat16      # ✅ 与 actor 匹配
```

**注意:** 这不如 actor-rollout 对齐那么关键,但建议保持一致。

### 陷阱 6: Megatron vs FSDP 配置混淆

**问题:**

```bash
# 使用 Megatron 后端但设置 FSDP dtype
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    actor_rollout_ref.actor.fsdp_config.dtype=float16  # ← 错误! 使用的是 Megatron
```

**结果:**

- 配置无效 (Megatron 不使用 fsdp_config)
- 训练使用默认的 Megatron dtype

**解决方案:**

为您的后端使用正确的配置路径:

```bash
# 对于 Megatron
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    actor_rollout_ref.actor.megatron.dtype=float16     # ✅ 正确

# 对于 FSDP
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    actor_rollout_ref.actor.fsdp_config.dtype=float16  # ✅ 正确
```

---

## 故障排查

### 问题 1: CUDA dtype 不匹配错误

**症状:**

```
RuntimeError: Expected all tensors to be on the same device and dtype,
but found at least two dtypes: BFloat16 and Float!
```

或:

```
RuntimeError: "addmm_cuda" not implemented for 'BFloat16'
```

**诊断:**

1. **检查 dtype 对齐:**

```bash
# 查看有效配置
python3 -m verl.trainer.main_ppo --cfg job 2>&1 | grep "dtype"
```

查找以下之间的不匹配:
- `actor_rollout_ref.actor.fsdp_config.dtype`
- `actor_rollout_ref.rollout.dtype`

2. **检查 GPU 支持:**

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

bfloat16 需要计算能力 ≥ 8.0 (Ampere)。

**解决方案:**

**对于 dtype 不匹配:**

```bash
# 确保所有 dtype 匹配
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16
```

**对于 GPU 不兼容:**

```bash
# 对于 V100 切换到 float16
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.rollout.dtype=float16
```

### 问题 2: 内存溢出 (OOM)

**症状:**

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB (GPU 0; 79.20 GiB total capacity; ...)
```

**诊断:**

检查是否使用 fp32:

```bash
python3 -m verl.trainer.main_ppo --cfg job | grep -E "dtype|model_dtype"
```

如果大型模型使用 `dtype: fp32` 或 `model_dtype: fp32`,内存使用过高。

**解决方案:**

**选项 1: 使用混合精度 (推荐):**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32  # fp32 初始化, bf16 训练
```

**选项 2: 完全低精度 (最大节省):**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16  # 都用 bf16
```

**选项 3: 启用卸载:**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
```

### 问题 3: 数值不稳定 / NaN Loss

**症状:**

```
Loss became NaN at step 150
```

或梯度爆炸/消失。

**诊断:**

1. **检查是否使用 float16:**

```bash
python3 -m verl.trainer.main_ppo --cfg job | grep "dtype: float16"
```

float16 范围有限,可能导致不稳定。

2. **检查梯度值:**

```bash
# 添加到训练脚本
wandb log → 检查梯度范数图
```

**解决方案:**

**选项 1: 切换到 bfloat16 (如果 GPU 支持):**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16  # 比 float16 更稳定
```

**选项 2: 启用梯度裁剪:**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.actor.optim.clip_grad=1.0  # 裁剪梯度
```

**选项 3: 降低学习率:**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.optim.lr=5e-7  # 更低的学习率以提高稳定性
```

**选项 4: 使用 fp32 进行调试:**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=fp32  # 全精度 (慢)
```

### 问题 4: 训练性能慢

**症状:**

- 训练比预期慢得多
- GPU 利用率低
- CPU 活动高

**诊断:**

检查 dtype 是否为 fp32:

```bash
python3 -m verl.trainer.main_ppo --cfg job | grep "dtype: fp32"
```

或检查 GPU 是否不支持当前 dtype (回退到 CPU)。

**解决方案:**

```bash
# 使用适当的低精度 dtype
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16  # 或 float16
```

确保 Tensor Core 利用率:

```bash
# 使用 nvidia-smi dmon 检查
nvidia-smi dmon -s u
# 应该显示高 "tensor" 核心使用率
```

### 问题 5: Rollout 生成错误

**症状:**

```
Error during rollout: tensor dtype mismatch in vLLM
```

**诊断:**

Rollout dtype 与同步的权重不匹配:

```python
# 检查日志中的权重同步消息
# 查找 dtype 转换警告
```

**解决方案:**

确保 rollout dtype 与 actor 训练 dtype 匹配:

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16  # ← 必须匹配!
```

### 问题 6: 混合 FSDP/Megatron 配置

**症状:**

```
Config override has no effect
```

**诊断:**

使用了错误的后端配置路径:

```bash
# 检查您使用的配置
python3 -m verl.trainer.main_ppo --config-name=ppo_megatron_trainer --cfg job | head -1
```

**解决方案:**

**对于 Megatron 后端:**

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    actor_rollout_ref.actor.megatron.dtype=bfloat16  # 不是 fsdp_config!
```

**对于 FSDP 后端:**

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16  # 不是 megatron!
```

---

## 最佳实践

### 1. ⭐ 始终在所有组件间对齐 dtype

**黄金法则:**

```python
actor.dtype == ref.dtype == rollout.dtype == critic.dtype
```

**实现:**

```bash
# 一次定义 dtype
DTYPE=bfloat16

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.ref.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    critic.model.fsdp_config.dtype=$DTYPE
```

**原因:** 防止 dtype 转换开销并确保数值一致性。

### 2. ⭐ 在现代 GPU 上默认使用 bfloat16

**推荐配置:**

```yaml
# 对于 H100, A100, A800
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16
      model_dtype: fp32      # 或大型模型用 bfloat16
```

**原因:**
- ✅ 低精度格式中的最佳数值稳定性
- ✅ Ampere+ GPU 上的硬件加速
- ✅ 相比 fp32 精度损失最小
- ✅ 不需要 loss scaling

### 3. ⭐ 先在小批量上测试 dtype 配置

**在完整训练之前:**

```bash
# 使用小批量测试运行
python3 -m verl.trainer.main_ppo \
    data.train_batch_size=64 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16
```

**验证:**
- ✅ 无 dtype 不匹配错误
- ✅ Loss 值合理 (不是 NaN)
- ✅ GPU 内存使用可接受
- ✅ Rollout 生成正常工作

**原因:** 在昂贵的完整训练运行之前及早发现配置错误。

### 4. ⭐ 考虑模型初始化的混合精度

**推荐模式:**

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16        # 快速训练
      model_dtype: fp32      # 稳定初始化
```

**何时使用:**
- 默认配置 (适用于大多数情况)
- 大型模型 (70B+)
- 从头初始化时
- 从 checkpoint 微调时

**何时避免:**
- 内存限制紧张 (改用 `model_dtype: bfloat16`)
- 小型模型 (<7B),内存不是问题

**原因:** 在初始化期间的数值稳定性与训练效率之间取得平衡。

### 5. ⭐ 在实验配置中记录 dtype 选择

**示例:**

```yaml
# experiment_config.yaml

# dtype 配置
# -------------------
# GPU: H100 80GB
# 理由: 在 Hopper 架构上使用 bfloat16 以获得最大性能
# 权衡: 相比 fp32 的 2倍内存节省 vs 最小精度损失
# 测试: 在 1000 步测试运行中未观察到数值不稳定

actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16
      model_dtype: bfloat16  # 完全 bf16 以在 80GB 中容纳 70B 模型
```

**原因:**
- 帮助协作者理解配置选择
- 实现可重现性
- 记录 GPU 要求
- 跟踪权衡和测试结果

### 6. 训练期间监控数值健康状况

**添加日志记录:**

```yaml
trainer:
  logger: '["console", "wandb"]'

# 记录这些指标:
# - 梯度范数 (注意爆炸)
# - Loss 值 (注意 NaN)
# - KL 散度 (应该稳定)
# - 奖励统计
```

**警告信号:**
- 🔴 Loss 变成 NaN
- 🔴 梯度范数 > 10.0 (持续)
- 🔴 KL 散度剧烈震荡

**行动:** 如果出现问题,考虑切换到更高精度或启用梯度裁剪。

### 7. 使用 Shell 变量确保 dtype 一致性

**模式:**

```bash
#!/bin/bash

# 集中的 dtype 配置
export TRAIN_DTYPE=bfloat16
export MODEL_DTYPE=fp32

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=$TRAIN_DTYPE \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.dtype=$TRAIN_DTYPE \
    actor_rollout_ref.ref.fsdp_config.dtype=$TRAIN_DTYPE \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    critic.model.fsdp_config.dtype=$TRAIN_DTYPE \
    critic.model.fsdp_config.model_dtype=$MODEL_DTYPE
```

**原因:** 单一真实来源,更容易为所有组件更改 dtype。

### 8. GPU 特定优化

**创建 GPU 特定配置:**

```bash
# run_h100.sh
DTYPE=bfloat16
MODEL_DTYPE=bfloat16  # 激进的内存节省

# run_v100.sh
DTYPE=float16
MODEL_DTYPE=fp32      # 保守以确保稳定性
```

**自动化 GPU 检测:**

```bash
if nvidia-smi | grep -qE "H100|A100"; then
    DTYPE=bfloat16
else
    DTYPE=float16
    echo "警告: 使用 float16 (不支持 bfloat16)"
fi
```

### 9. 版本控制 dtype 配置

**在 Git 中跟踪:**

```bash
# configs/
#   ├── h100_bfloat16.yaml
#   ├── a100_bfloat16.yaml
#   └── v100_float16.yaml
```

**示例:**

```yaml
# configs/h100_bfloat16.yaml
defaults:
  - _self_
  - override /actor_rollout_ref/actor/fsdp_config: {dtype: bfloat16, model_dtype: bfloat16}
  - override /actor_rollout_ref/rollout: {dtype: bfloat16}
```

**使用:**

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=configs \
    --config-name=h100_bfloat16
```

### 10. Hydra 组合后验证配置

**始终检查最终配置:**

```bash
python3 -m verl.trainer.main_ppo \
    [your overrides] \
    --cfg job 2>&1 | grep -A2 "dtype"
```

**预期输出:**

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      dtype: bfloat16       # ✅ 验证这与您的意图匹配
  rollout:
    dtype: bfloat16         # ✅ 验证对齐
```

**原因:** Hydra 的组合可能很复杂;始终验证最终合并的配置。

---

## 相关文件

### 配置文件

**主训练配置:**
- `verl/verl/trainer/config/ppo_trainer.yaml` - FSDP PPO trainer
- `verl/verl/trainer/config/ppo_megatron_trainer.yaml` - Megatron PPO trainer
- `verl/verl/trainer/config/_generated_ppo_trainer.yaml` - 自动生成的配置

**引擎配置:**
- `verl/verl/trainer/config/engine/fsdp.yaml` - FSDP 引擎 (第 56-57 行)
- `verl/verl/trainer/config/engine/megatron.yaml` - Megatron 引擎 (第 81 行)

**组件配置:**
- `verl/verl/trainer/config/actor/actor.yaml` - Actor 配置
- `verl/verl/trainer/config/critic/critic.yaml` - Critic 配置
- `verl/verl/trainer/config/ref/ref.yaml` - 参考模型配置
- `verl/verl/trainer/config/rollout/rollout.yaml` - Rollout 配置 (第 29 行)

**特殊配置:**
- `verl/verl/trainer/config/generation.yaml` - 纯推理配置 (第 26 行)
- `verl/examples/split_placement/config/ppo_trainer_split.yaml` - 分离放置示例

### 源代码

**Worker 实现:**
- `verl/verl/workers/fsdp_workers.py` - FSDP workers (第 461-471 行: dtype 设置, 第 704-721 行: 权重同步)
- `verl/verl/workers/megatron_workers.py` - Megatron workers

**Rollout 引擎:**
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` - vLLM rollout (第 236, 594 行)
- `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py` - SGLang rollout (第 447 行)

**配置数据类:**
- `verl/verl/workers/config/engine.py` - 引擎配置 (第 49, 75, 103, 122 行)
- `verl/verl/workers/config/rollout.py` - Rollout 配置 (第 132 行)

### 示例脚本

**带 dtype 覆盖的 FSDP 示例:**
- `verl/recipe/flowrl/run_flowrl_qwen2.5_7b_fp16.sh` - FP16 配置
- `verl/examples/framethinker/train_frame_thinker_vhonly.sh` - 视觉模型与 FP16
- `verl/examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh` - 多轮与 FP16

**标准示例:**
- `verl/examples/ppo_trainer/run_qwen2-7b_rm.sh` - 带奖励模型的 PPO
- `verl/examples/grpo_trainer/run_qwen2-7b_math.sh` - GRPO 训练
- `verl/examples/grpo_trainer/run_gptoss_20b.sh` - 大型模型与 bfloat16

### 文档

**官方文档:**
- `verl/docs/examples/config.rst` - 配置文档 (第 186, 349-350 行)
- 在线: https://verl.readthedocs.io/en/latest/examples/config.html

**claude_wiki 中的相关指南:**
- `COMPATIBILITY_NOTES.md` - 工具兼容性
- `REWARD_FUNCTIONS_GUIDE.md` - 奖励函数配置

### Pull Requests & Issues

**FP16 支持:**
- PR #4036 - 添加了全面的 FP16 支持
- PR #3068 - 混合精度配置改进

**相关讨论:**
- GitHub Discussions: 搜索 "dtype" 查看社区故障排查
- Slack: #verl-support 频道

---

## 总结

**关键要点:**

1. ⭐ **dtype 控制精度** 用于训练 (梯度/激活值) 和推理
2. ⭐ **model_dtype 控制初始化** 精度 (通常为 fp32)
3. ⭐ **对齐至关重要**: `actor.dtype` 必须等于 `rollout.dtype`
4. ⭐ **在现代 GPU 上默认使用 bfloat16** (H100, A100)
5. ⭐ **对 V100 使用 float16** 或不支持 bfloat16 的旧款 GPU
6. ⭐ **先测试配置** 在完整训练之前在小批量上
7. ⭐ **监控数值健康** (梯度, loss) 在训练期间

**快速参考卡:**

| GPU 类型 | 推荐 dtype | model_dtype | 说明 |
|---------|-----------|-------------|------|
| H100/H800 | `bfloat16` | `bfloat16` | 最大内存节省 |
| A100/A800 | `bfloat16` | `fp32` | 默认, 平衡 |
| V100 | `float16` | `fp32` | 添加梯度裁剪 |
| NPU/Ascend | `bfloat16` (或 `float16`) | `fp32` | 查看 NPU 文档 |

**基本命令:**

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    # 确保这两个始终对齐!
```

---

**状态:** ✓ 文档完整

**最后更新:** 2025-11-25

**维护者:** veRL 社区

**有问题?** 在 [GitHub](https://github.com/volcengine/verl/issues) 上开 issue 或在 [Slack](https://join.slack.com/t/verl-project/shared_invite/zt-3c6mc2khw-v0lo6NfDPuFP6OnkrZwfqw) 中提问
