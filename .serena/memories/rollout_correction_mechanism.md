# Rollout Correction 机制详解

## 概述

Rollout Correction 是 veRL 中用于解决 **off-policy 问题** 的核心机制。它通过 Importance Sampling (IS) 和 Rejection Sampling (RS) 来纠正 rollout policy 和 training policy 之间的分布偏差。

### 适用场景

1. **Policy 实现差异**：rollout 使用 vLLM BFloat16，training 使用 FSDP FP32
2. **模型更新滞后**：训练时使用的是旧 checkpoint 生成的轨迹
3. **一般分布偏移**：数据收集和训练之间的任何分布差异

---

## 两种运行模式

### 1. Decoupled Mode (默认，bypass_mode=False)

**特点：三策略系统 (π_rollout, π_old, π_θ)**

- **π_rollout**：用于生成轨迹的策略（如 vLLM inference）
- **π_old**：作为 proximal anchor 的稳定参考策略（每个 data batch 开始时计算一次）
- **π_θ**：当前正在训练的策略（在 mini-batch 更新中不断演化）

**关键配置：**
```yaml
algorithm:
  rollout_correction:
    bypass_mode: false  # 默认值
    rollout_is: "token"  # 或 "sequence"，None 禁用
    rollout_is_threshold: 2.0
    rollout_rs: "token"  # 或 "sequence", "geometric", None 禁用
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5
    rollout_token_veto_threshold: 0.1  # 可选，灾难性异常值过滤
    rollout_is_batch_normalize: false
```

**执行流程：**

1. **Trainer 层面**（`ray_trainer.py:1363-1386`）：
   - 重新计算 `old_log_probs = π_old(a|s)`
   - 调用 `compute_rollout_correction_and_add_to_batch()`

2. **计算 IS weights 和 RS**（`ray_trainer.py:1446-1462`）：
   ```python
   if (rollout_corr_config is not None
       and "rollout_log_probs" in batch.batch
       and not bypass_recomputing_logprobs):  # Decoupled mode 条件
       
       batch, is_metrics = compute_rollout_correction_and_add_to_batch(
           batch, rollout_corr_config
       )
       metrics.update(is_metrics)
   ```

3. **添加到 batch**：
   - `batch["rollout_is_weights"]`: IS weights tensor
   - `batch["response_mask"]`: 修改后的 mask（已应用 RS 过滤）
   - `metrics`: 统计信息（KL、PPL、ESS 等）

4. **Actor 层面**（`dp_actor.py:477-494`）：
   - 从 batch 中提取预计算的 `rollout_is_weights`
   - 应用到 loss 计算：`pg_losses *= rollout_is_weights`
   - 反向传播 + 优化器更新

---

### 2. Bypass Mode (bypass_mode=True)

**特点：两策略系统 (π_rollout, π_θ)**

- **π_rollout = π_old**：直接复用 rollout_log_probs 作为 old_log_probs（零成本）
- **π_θ**：当前正在训练的策略

**适用场景**：rollout policy 和 training policy 差异很小（如同一模型只更新了几步）

#### 子模式 1：Bypass + PPO Loss（默认）

**配置：**
```yaml
algorithm:
  rollout_correction:
    bypass_mode: true
    use_policy_gradient: false  # 默认值
```

**特点：**
- ✅ 省略昂贵的 `compute_log_prob()` 调用
- ❌ **不计算 IS weights 和 RS**
- 使用标准 PPO loss：`ratio = π_θ / π_rollout`，PPO clipping

**代码路径**（`rollout_corr_helper.py:908-948`）：
```python
def apply_rollout_correction(batch, rollout_corr_config, policy_loss_config):
    # Use rollout log probs as old log probs (zero-cost substitution)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
    
    # Bypass + PPO loss (默认)
    # PPO clips ratio π_θ/π_rollout instead of π_θ/π_old
```

#### 子模式 2：Bypass + Policy Gradient Loss

**配置：**
```yaml
algorithm:
  rollout_correction:
    bypass_mode: true
    use_policy_gradient: true  # 启用 policy gradient 模式
    rollout_is: "token"
    rollout_is_threshold: 2.0
    rollout_rs: "token"
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5
```

**特点：**
- ✅ 省略 trainer 层面的 `compute_log_prob()` 调用
- ✅ **在 Actor 内部实时计算 IS weights 和 RS**
- 使用 policy gradient (REINFORCE-style)，**无 PPO clipping**

**代码路径**（`rollout_corr_helper.py:945-948`）：
```python
if use_policy_gradient:
    # Policy gradient mode: Configure actor to use rollout_correction loss function
    # This will use compute_policy_loss_with_rollout_correction (no PPO clipping)
    policy_loss_config["loss_mode"] = "rollout_correction"
```

**Actor 内部计算**（`core_algos.py:1632-1646`）：
```python
# 在 compute_policy_loss_with_rollout_correction 中
with torch.no_grad():
    rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
        compute_rollout_correction_and_rejection_mask(
            old_log_prob=log_prob,  # 当前策略 π_θ
            rollout_log_prob=rollout_log_prob,  # Rollout 策略
            response_mask=eos_mask,
            rollout_is=rollout_is,
            ...
        )
    )

# 应用 IS weights 到 policy gradient
if rollout_is_weights is not None:
    pg_losses = -advantages * log_prob * rollout_is_weights
else:
    pg_losses = -advantages * log_prob
```

---

## 详细执行流程

### Decoupled Mode 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Rollout (inference engine)                            │
│  生成轨迹并计算 rollout_log_probs                                │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Trainer - Compute old_log_probs                       │
│  (ray_trainer.py:1363)                                         │
├─────────────────────────────────────────────────────────────────┤
│  old_log_prob = actor_rollout_wg.compute_log_prob(batch)       │
│  # π_old(a|s) - 作为稳定的 proximal anchor                      │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Trainer - Compute IS weights & Rejection Sampling     │
│  (ray_trainer.py:1446-1462)                                    │
├─────────────────────────────────────────────────────────────────┤
│  条件检查:                                                       │
│  if (rollout_corr_config is not None                           │
│      and "rollout_log_probs" in batch                          │
│      and not bypass_recomputing_logprobs):  # ← 关键条件!       │
│                                                                 │
│  执行:                                                           │
│  batch, is_metrics = compute_rollout_correction_and_add_to_batch(│
│      batch, rollout_corr_config                                │
│  )                                                              │
│                                                                 │
│  内部计算:                                                       │
│  1. log_ratio = old_log_prob - rollout_log_prob                │
│  2. IS weights = truncate(exp(log_ratio), threshold)           │
│  3. Rejection mask = filter_by_threshold(log_ratio)            │
│  4. Veto mask = filter_catastrophic_tokens(log_ratio)          │
│  5. Metrics: KL, PPL, ESS, rejection rate, etc.               │
│                                                                 │
│  添加到 batch:                                                   │
│  - batch["rollout_is_weights"] = IS weights tensor             │
│  - batch["response_mask"] = modified_mask (过滤后)              │
│  - metrics["rollout_corr/*"] = 统计信息                         │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Trainer - Compute Advantages                          │
│  (ray_trainer.py:1469-1477)                                    │
├─────────────────────────────────────────────────────────────────┤
│  batch = compute_advantage(                                     │
│      batch,                                                     │
│      adv_estimator="gae" / "grpo" / etc.,                      │
│      gamma, lam, ...                                            │
│  )                                                              │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Trainer - Send to Actor Worker                        │
│  (ray_trainer.py:1495)                                         │
├─────────────────────────────────────────────────────────────────┤
│  actor_output = actor_rollout_wg.update_actor(batch)           │
│                                                                 │
│  Batch 包含:                                                     │
│  - old_log_probs                                                │
│  - rollout_log_probs (optional, for metrics)                   │
│  - rollout_is_weights (如果配置了 rollout_is)                    │
│  - response_mask (已应用 RS 和 veto 过滤)                        │
│  - advantages                                                   │
│  - responses, input_ids, attention_mask, etc.                  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Actor - Extract IS weights                            │
│  (dp_actor.py:477-479)                                         │
├─────────────────────────────────────────────────────────────────┤
│  # Extract pre-computed rollout correction weights if present  │
│  # Weights are computed centrally in trainer                   │
│  rollout_is_weights = model_inputs.get("rollout_is_weights", None)│
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7: Actor - Forward Pass                                  │
│  (dp_actor.py:459-461)                                         │
├─────────────────────────────────────────────────────────────────┤
│  entropy, log_prob = self._forward_micro_batch(                │
│      model_inputs, temperature=temperature, ...                │
│  )                                                              │
│  # log_prob = π_θ(a|s) - 当前训练策略                            │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 8: Actor - Compute Policy Loss                           │
│  (dp_actor.py:486-494, core_algos.py:966-967)                 │
├─────────────────────────────────────────────────────────────────┤
│  policy_loss_fn = get_policy_loss_fn(loss_mode)  # "vanilla"   │
│  pg_loss, pg_metrics = policy_loss_fn(                         │
│      old_log_prob=old_log_prob,    # π_old                     │
│      log_prob=log_prob,            # π_θ                       │
│      advantages=advantages,                                     │
│      response_mask=response_mask,  # 已过滤                     │
│      rollout_is_weights=rollout_is_weights,  # 传递 IS weights │
│      ...                                                        │
│  )                                                              │
│                                                                 │
│  在 compute_policy_loss_vanilla 内部:                           │
│  1. ratio = exp(log_prob - old_log_prob) = π_θ / π_old         │
│  2. pg_losses1 = -advantages * ratio                            │
│  3. pg_losses2 = -advantages * clip(ratio, 1±ε)                │
│  4. pg_losses = max(pg_losses1, pg_losses2)  # PPO clip        │
│  5. if rollout_is_weights is not None:                         │
│         pg_losses = pg_losses * rollout_is_weights  # ← 应用IS! │
│  6. pg_loss = mean(pg_losses)                                  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 9: Actor - Backward & Update                             │
│  (dp_actor.py: loss.backward() + optimizer.step())            │
├─────────────────────────────────────────────────────────────────┤
│  loss.backward()        # 计算梯度                              │
│  optimizer.step()       # 更新模型参数 θ ← θ - lr * ∇L          │
│                                                                 │
│  ✅ 模型在这里真正更新！                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Bypass Mode 流程对比

#### Bypass + PPO Loss (默认)

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Rollout - 生成轨迹                                      │
│  rollout_log_probs = π_rollout(a|s)                            │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Trainer - Bypass Mode                                 │
│  (ray_trainer.py:1351-1360)                                    │
├─────────────────────────────────────────────────────────────────┤
│  apply_rollout_correction(batch, ...)                          │
│                                                                 │
│  内部操作:                                                        │
│  batch["old_log_probs"] = batch["rollout_log_probs"]  # 直接赋值│
│  # ⚡ 零成本！跳过昂贵的 compute_log_prob() 调用                  │
│                                                                 │
│  ❌ 不执行 IS/RS 计算（因为 bypass_recomputing_logprobs=True）   │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Trainer - Compute Advantages                          │
│  batch = compute_advantage(...)                                │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Actor - Standard PPO Loss                             │
│  (no IS weights in batch)                                      │
├─────────────────────────────────────────────────────────────────┤
│  ratio = π_θ / π_rollout  # 注意：这里 π_old = π_rollout!       │
│  pg_losses = PPO_clip(ratio, advantages)                       │
│  pg_loss = mean(pg_losses)                                     │
│  loss.backward()                                                │
│  optimizer.step()                                               │
└─────────────────────────────────────────────────────────────────┘
```

#### Bypass + Policy Gradient

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1-3: Same as above (bypass old_log_prob computation)     │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Actor - Compute IS/RS on-the-fly                      │
│  (core_algos.py:1632-1646)                                     │
├─────────────────────────────────────────────────────────────────┤
│  loss_mode = "rollout_correction"                              │
│                                                                 │
│  在 compute_policy_loss_with_rollout_correction 中:            │
│  with torch.no_grad():                                          │
│      rollout_is_weights, modified_mask, metrics = (            │
│          compute_rollout_correction_and_rejection_mask(        │
│              old_log_prob=log_prob,  # π_θ (当前策略)          │
│              rollout_log_prob=rollout_log_prob,  # π_rollout   │
│              ...                                                │
│          )                                                      │
│      )                                                          │
│                                                                 │
│  # Policy Gradient (REINFORCE):                                │
│  if rollout_is_weights is not None:                            │
│      pg_losses = -advantages * log_prob * rollout_is_weights   │
│  else:                                                          │
│      pg_losses = -advantages * log_prob                         │
│                                                                 │
│  pg_loss = mean(pg_losses)  # 无 PPO clipping!                 │
│  loss.backward()                                                │
│  optimizer.step()                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心算法实现

### Importance Sampling (IS) Weights

**目的**：纠正 π_old 和 π_rollout 之间的分布偏差

**数学公式**：
```
w(s,a) = π_old(a|s) / π_rollout(a|s)
       = exp(log π_old(a|s) - log π_rollout(a|s))
```

**截断 (Truncation)**：
```
w_truncated = clip(w, 1/threshold, threshold)
```

**聚合模式**（`rollout_is` 参数）：

1. **Token-level** (`rollout_is="token"`):
   ```
   w_token[t] = clip(exp(old_log_prob[t] - rollout_log_prob[t]), 1/τ, τ)
   ```
   每个 token 有独立的 weight

2. **Sequence-level** (`rollout_is="sequence"`):
   ```
   log_ratio_seq = sum(old_log_prob - rollout_log_prob)  # 序列级别
   w_seq = clip(exp(log_ratio_seq), 1/τ, τ)
   w_token[t] = w_seq  # 广播到所有 tokens
   ```

**应用到梯度**：
```python
pg_losses = PPO_loss(ratio, advantages)
pg_losses = pg_losses * rollout_is_weights  # 影响梯度大小
```

**代码位置**：
- 计算：`rollout_corr_helper.py:compute_rollout_correction_weights()`
- 应用：`core_algos.py:966-967`, `core_algos.py:1663`

---

### Rejection Sampling (RS)

**目的**：过滤掉 ratio 过大或过小的异常样本

**数学公式**：
```
reject_mask[t] = 1  if  1/τ_lower < ratio[t] < τ_upper
                 0  otherwise
```

**聚合模式**（`rollout_rs` 参数）：

1. **Token-level** (`rollout_rs="token"`):
   ```python
   # 逐 token 判断
   token_reject = (log_ratio < log_lower) | (log_ratio > log_upper)
   response_mask[token_reject] = 0  # 屏蔽异常 token
   ```

2. **Sequence-level** (`rollout_rs="sequence"`):
   ```python
   # 整个序列的平均 ratio
   seq_log_ratio = sum(log_ratio * response_mask) / sum(response_mask)
   seq_reject = (seq_log_ratio < log_lower) | (seq_log_ratio > log_upper)
   response_mask[seq_reject, :] = 0  # 屏蔽整个序列
   ```

3. **Geometric** (`rollout_rs="geometric"`):
   ```python
   # 序列的几何平均
   seq_log_ratio = sum(log_ratio * response_mask) / sum(response_mask)
   # 同 sequence-level
   ```

**Veto 机制**（灾难性异常值）：
```python
if rollout_token_veto_threshold is not None:
    # 如果序列中有任何 token 的 ratio < veto_threshold
    # 则拒绝整个序列
    catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)
    veto_mask = (~has_catastrophic).float()
    response_mask = response_mask * veto_mask
```

**代码位置**：
- 计算：`rollout_corr_helper.py:compute_rollout_rejection_mask()`
- Veto：`rollout_corr_helper.py:544-695` (Step 4)

---

## 关键代码位置总结

### Trainer (ray_trainer.py)

| 行号 | 功能 | 描述 |
|------|------|------|
| 1343-1360 | Bypass mode 判断 | 检查 `bypass_mode`，调用 `apply_rollout_correction` |
| 1363-1386 | 计算 old_log_probs | Decoupled mode: `actor_rollout_wg.compute_log_prob()` |
| 1446-1462 | 计算 IS/RS | `compute_rollout_correction_and_add_to_batch()` |
| 1495 | 发送到 Actor | `actor_rollout_wg.update_actor(batch)` |

### Rollout Correction Helper (rollout_corr_helper.py)

| 行号 | 函数 | 功能 |
|------|------|------|
| 544-695 | `compute_rollout_correction_and_rejection_mask` | 统一接口：IS + RS + Veto + Metrics |
| 807-865 | `compute_rollout_correction_and_add_to_batch` | Trainer 调用的封装函数 |
| 908-948 | `apply_rollout_correction` | Bypass mode 处理 |

### Actor (dp_actor.py)

| 行号 | 功能 | 描述 |
|------|------|------|
| 371-588 | `update_policy` | Actor 主循环 |
| 477-479 | 提取 IS weights | `rollout_is_weights = model_inputs.get(...)` |
| 486-494 | 计算 loss | 调用 policy loss 函数 |

### Policy Loss (core_algos.py)

| 行号 | 函数 | 功能 |
|------|------|------|
| 888-976 | `compute_policy_loss_vanilla` | 标准 PPO loss（支持 IS weights） |
| 966-967 | 应用 IS weights | `pg_losses *= rollout_is_weights` |
| 1555-1689 | `compute_policy_loss_with_rollout_correction` | Bypass + PG: 实时计算 IS/RS |
| 1692-1753 | `compute_policy_loss_rollout_correction_wrapper` | 注册为 "rollout_correction" loss |

---

## 模式选择指南

### 何时使用 Decoupled Mode

✅ **推荐场景：**
- Rollout 和 Training 有显著差异：
  - 不同精度（vLLM BF16 vs FSDP FP32）
  - 不同实现（vLLM vs Megatron）
  - 较大的模型更新间隔
- 需要精确控制 off-policy 问题
- 对统计指标有详细监控需求

**优点：**
- 稳定的 π_old 作为 proximal anchor
- IS/RS 在 trainer 计算一次，actor 直接使用（高效）
- 完整的 off-policy 诊断 metrics

**缺点：**
- 需要额外的前向计算（`compute_log_prob`）
- 内存占用略高（存储 IS weights）

### 何时使用 Bypass Mode

✅ **推荐场景：**
- Rollout 和 Training 差异很小：
  - 同一模型实现
  - 更新频繁（如每个 batch 都更新）
  - 精度差异可忽略
- 计算资源受限
- 追求最大训练速度

**Bypass + PPO Loss (默认)：**
- 适合标准 on-policy 或 near-on-policy 场景
- 最高效（零额外计算）
- 仍然保留 PPO clipping 的稳定性

**Bypass + Policy Gradient：**
- 适合需要灵活 IS/RS 调整的场景
- 在 actor 内部实时计算（每个 mini-batch 动态调整）
- 无 PPO clipping（可能不太稳定）

---

## 配置示例

### 示例 1：Decoupled Mode with Full Correction

```yaml
# 适用于 vLLM (BF16) rollout + FSDP (FP32) training
algorithm:
  rollout_correction:
    bypass_mode: false
    
    # Importance Sampling
    rollout_is: "token"
    rollout_is_threshold: 2.0
    rollout_is_batch_normalize: false
    
    # Rejection Sampling
    rollout_rs: "token"
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5
    
    # Veto (灾难性异常值过滤)
    rollout_token_veto_threshold: 0.1

actor_rollout_ref:
  rollout:
    calculate_log_probs: true  # 必须开启！
```

### 示例 2：Bypass Mode (高效模式)

```yaml
# 适用于频繁更新、差异小的场景
algorithm:
  rollout_correction:
    bypass_mode: true
    use_policy_gradient: false  # 使用标准 PPO loss

actor_rollout_ref:
  rollout:
    calculate_log_probs: false  # 可以关闭以节省计算
```

### 示例 3：Bypass + Policy Gradient (灵活模式)

```yaml
# 适用于需要动态 IS 调整的场景
algorithm:
  rollout_correction:
    bypass_mode: true
    use_policy_gradient: true  # 启用 policy gradient
    
    # 这些参数会在 actor 内部使用
    rollout_is: "token"
    rollout_is_threshold: 2.0
    rollout_rs: "token"
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5

actor_rollout_ref:
  rollout:
    calculate_log_probs: true  # 必须开启（用于 metrics）
```

---

## 监控指标

### Rollout Correction Metrics (前缀: `rollout_corr/`)

#### IS Weights 统计
- `rollout_is_mean`: IS weights 平均值（理想值: 1.0）
- `rollout_is_std`: IS weights 标准差（越小越好）
- `rollout_is_max`: 最大 IS weight
- `rollout_is_min`: 最小 IS weight
- `rollout_is_ess`: Effective Sample Size（有效样本数，越大越好）

#### Rejection Sampling 统计
- `rollout_rs_rejection_rate`: 被拒绝样本的比例
- `rollout_rs_kept_fraction`: 保留样本的比例

#### Veto 统计
- `rollout_is_veto_fraction`: 被 veto 的序列比例
- `rollout_is_catastrophic_token_fraction`: 灾难性 token 比例

#### Off-Policy 诊断
- `rollout_kl`: KL 散度 (π_old || π_rollout)
- `rollout_ppl_diff`: Perplexity 差异
- `rollout_chi2`: χ² 散度

### 其他相关 Metrics

#### Actor Metrics (前缀: `actor/`)
- `actor/ppo_kl`: KL(π_θ || π_old)
- `actor/probs_diff_mean`: log prob 差异（如果有 rollout_log_probs）
- `actor/log_seq_iw_mean`: 序列级别 importance weight

---

## 常见问题 (FAQ)

### Q1: Decoupled Mode 为什么需要三个策略？

**A:** 
- **π_rollout**: 生成轨迹（可能是旧模型或不同实现）
- **π_old**: 作为 PPO 的 proximal anchor，需要稳定不变
- **π_θ**: 当前训练策略，在 mini-batch 更新中不断变化

如果直接用 π_rollout 作为 π_old，会导致：
1. Off-policy 问题（分布不匹配）
2. PPO ratio 计算不准确

IS weights 的作用就是纠正 π_old 和 π_rollout 之间的差异。

### Q2: Bypass Mode 什么时候需要 use_policy_gradient=True？

**A:** 当满足以下条件时：
- Rollout 和 Training 仍有一定差异（虽然不大）
- 希望动态调整 IS weights（每个 mini-batch 重新计算）
- 不需要 PPO clipping 的稳定性约束

**权衡：**
- ✅ 更灵活的 off-policy correction
- ❌ 计算开销更大（每个 mini-batch 都计算 IS）
- ❌ 可能不如 PPO clipping 稳定

### Q3: IS weights 是如何影响梯度的？

**A:** IS weights 通过乘法应用到 loss 上：
```python
pg_losses = PPO_loss(ratio, advantages)
pg_losses = pg_losses * rollout_is_weights  # w ∈ [1/τ, τ]

# 高权重样本 (w > 1): 梯度放大 → 更新幅度更大
# 低权重样本 (w < 1): 梯度缩小 → 更新幅度更小
# 正常样本 (w ≈ 1): 梯度不变
```

这样可以纠正由于分布偏差导致的梯度估计偏差。

### Q4: Rejection Sampling 和 Veto 有什么区别？

**A:**

| 特性 | Rejection Sampling (RS) | Veto |
|------|------------------------|------|
| **阈值** | 上下界 `[1/τ_lower, τ_upper]` | 单一下界 `τ_veto` |
| **聚合** | Token/Sequence/Geometric | 仅 Token-level |
| **作用** | 过滤轻微异常样本 | 过滤灾难性异常样本 |
| **触发条件** | ratio 超出合理范围 | 任何 token 的 ratio < τ_veto |
| **严格程度** | 较宽松（如 [0.5, 2.0]） | 非常严格（如 0.1） |

**执行顺序**: RS → Veto（Veto 可以覆盖 RS 的决策）

### Q5: 为什么 Trainer 不更新模型？

**A:** 这是分布式训练的典型架构：

**Trainer (Driver Process)**:
- 运行在 CPU 或单个 GPU
- 负责协调、数据准备、统计收集
- 不持有可训练模型（或只是用于计算 old_log_prob 的快照）

**Actor Workers (GPU Processes)**:
- 运行在多个 GPU 上
- 负责密集的梯度计算和参数更新
- 支持数据并行（多个 workers 同时训练）

**好处**：
1. 解耦协调逻辑和计算逻辑
2. Trainer 可以在准备下一个 batch 的同时，Actor 在训练当前 batch
3. 易于扩展到多 GPU、多节点

### Q6: 如何选择 rollout_is_threshold？

**A:** 经验法则：

| Threshold | 适用场景 | 说明 |
|-----------|---------|------|
| 1.5 - 2.0 | 轻微 off-policy | 精度差异、小幅更新 |
| 2.0 - 5.0 | 中等 off-policy | 实现差异、中等更新间隔 |
| 5.0 - 10.0 | 严重 off-policy | 大幅更新、显著分布偏移 |

**调试建议**：
1. 监控 `rollout_corr/rollout_is_max` 和 `rollout_is_min`
2. 如果经常达到 threshold 边界，说明 threshold 太小
3. 如果 ESS (Effective Sample Size) 太低，说明 threshold 可能太大

---

## 参考文献

### 相关论文
1. **Importance Sampling**: 
   - Precup et al., "Eligibility Traces for Off-Policy Policy Evaluation" (2000)
   
2. **PPO Algorithm**: 
   - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
   https://arxiv.org/abs/1707.06347

3. **Off-Policy Correction**:
   - Munos et al., "Safe and Efficient Off-Policy Reinforcement Learning" (2016)

### veRL 相关
- 原始实现作者: szrlee <szrlee@gmail.com>
- EuroSys 2025 论文（HybridFlow framework）

---

## 变更历史

- **2025-01**: 初始文档创建
- 本文档基于 veRL codebase 分析生成