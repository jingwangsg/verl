# Rollout Correction 机制详解

## 概述

Rollout Correction 是 veRL 中用于解决 **off-policy 问题** 的核心机制。它通过 Importance Sampling (IS) 和 Rejection Sampling (RS) 来纠正 rollout policy 和 training policy 之间的分布偏差。

### 适用场景

1. **Policy 实现差异**：rollout 使用 vLLM BFloat16，training 使用 FSDP FP32
2. **模型更新滞后**：训练时使用的是旧 checkpoint 生成的轨迹
3. **一般分布偏移**：数据收集和训练之间的任何分布差异

---

## 核心理解：Trainer vs Actor 的职责分工

### ⚠️ 关键澄清

**Trainer 计算的 IS weights 会被传递给 Actor 使用！**

```python
# Trainer 层面 (ray_trainer.py:1456-1462)
batch, is_metrics = compute_rollout_correction_and_add_to_batch(...)
# 内部会执行 (rollout_corr_helper.py:863-864):
batch = batch.union(rollout_is_weights)  # ← IS weights 添加到 batch!

# Actor 层面 (dp_actor.py:394-395)
if "rollout_is_weights" in data.batch.keys():
    select_keys.append("rollout_is_weights")  # ← 从 batch 中提取

# 应用到 loss (core_algos.py:966-967)
pg_losses = pg_losses * rollout_is_weights  # ← 影响梯度
```

### Trainer 的职责（Driver Process）

1. ✅ **计算 IS weights 和 rejection mask**（如果在 decoupled mode）
2. ✅ **将 IS weights 添加到 batch**（`batch["rollout_is_weights"]`）
3. ✅ **收集和记录统计 metrics**
4. ✅ **发送包含 IS weights 的 batch 到 Actor**
5. ❌ **不执行反向传播**
6. ❌ **不更新模型参数**

### Actor 的职责（GPU Worker Process）

1. ✅ **接收包含 IS weights 的 batch**
2. ✅ **从 batch 中提取 `rollout_is_weights`**
3. ✅ **前向计算当前策略 π_θ 的 log_prob**
4. ✅ **将 IS weights 应用到 loss 计算**（`pg_losses *= rollout_is_weights`）
5. ✅ **执行反向传播**（`loss.backward()`）
6. ✅ **更新模型参数**（`optimizer.step()`）

### 完整数据流

```
Trainer                                      Actor
   │                                           │
   ├─ 计算 IS weights                          │
   │  rollout_is_weights = ...                │
   │                                           │
   ├─ 添加到 batch ──────────────────────────> ├─ 接收 batch
   │  batch["rollout_is_weights"] = weights    │  
   │                                           │
   ├─ 发送 batch ───────────────────────────> ├─ 提取 IS weights
   │                                           │  rollout_is_weights = batch.get(...)
   │                                           │
   │                                           ├─ 前向计算
   │                                           │  log_prob = model(...)
   │                                           │
   │                                           ├─ 计算 loss
   │                                           │  pg_losses *= rollout_is_weights
   │                                           │
   │                                           ├─ 反向传播
   │                                           │  loss.backward()
   │                                           │
   │                                           ├─ 更新模型 ✅
   │                                           │  optimizer.step()
```

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

3. **添加到 batch**（`rollout_corr_helper.py:863-864`）：
   ```python
   # ALWAYS update response_mask with rejection applied
   batch.batch["response_mask"] = modified_response_mask
   
   # Add IS weights to batch if computed
   if rollout_is_weights is not None:
       batch = batch.union(rollout_is_weights)  # ← 添加到 batch!
   ```
   结果：
   - `batch["rollout_is_weights"]`: IS weights tensor
   - `batch["response_mask"]`: 修改后的 mask（已应用 RS 过滤）
   - `metrics`: 统计信息（KL、PPL、ESS 等）

4. **Actor 层面**（`dp_actor.py:394-395, 477-494`）：
   ```python
   # 检测并提取 IS weights
   if "rollout_is_weights" in data.batch.keys():
       select_keys.append("rollout_is_weights")
   
   # 在 micro batch 中使用
   rollout_is_weights = model_inputs.get("rollout_is_weights", None)
   
   # 应用到 loss 计算
   pg_loss, pg_metrics = policy_loss_fn(
       old_log_prob=old_log_prob,
       log_prob=log_prob,
       advantages=advantages,
       rollout_is_weights=rollout_is_weights,  # ← 传递给 loss 函数
       ...
   )
   ```

5. **Loss 函数中应用**（`core_algos.py:966-967`）：
   ```python
   pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
   
   # Apply rollout correction weights if provided
   if rollout_is_weights is not None:
       pg_losses = pg_losses * rollout_is_weights  # ← 影响梯度!
   
   pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
   ```

6. **反向传播和更新**（`dp_actor.py`）：
   ```python
   loss.backward()        # 计算梯度
   optimizer.step()       # 更新模型参数 ✅
   ```

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
- ❌ **不计算 IS weights 和 RS**（Trainer 和 Actor 都不计算）
- ❌ **batch 中没有 `rollout_is_weights`**
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
- ❌ **Trainer 不计算 IS weights**（batch 中没有 `rollout_is_weights`）
- ✅ **Actor 内部实时计算 IS weights 和 RS**
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

## IS Weights 的两种计算方式对比

### Decoupled Mode：Trainer 计算 + Actor 使用

```python
# ========== Trainer (ray_trainer.py:1456-1462) ==========
batch, is_metrics = compute_rollout_correction_and_add_to_batch(
    batch, rollout_corr_config
)
# 内部：rollout_is_weights = exp(old_log_prob - rollout_log_prob)
# 添加：batch["rollout_is_weights"] = rollout_is_weights

# ========== Actor (dp_actor.py:477-494) ==========
rollout_is_weights = model_inputs.get("rollout_is_weights", None)  # 从 batch 提取
pg_loss, pg_metrics = policy_loss_fn(
    ...,
    rollout_is_weights=rollout_is_weights,  # 传递
)

# ========== Loss Function (core_algos.py:966-967) ==========
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights  # 应用
```

**计算时机**：每个 data batch 一次（在所有 mini-batch 和 epochs 中重复使用）  
**使用的 policies**：π_old vs π_rollout（稳定）

---

### Bypass + Policy Gradient：Actor 实时计算

```python
# ========== Trainer ==========
# 不计算 IS weights，batch 中没有 "rollout_is_weights"

# ========== Actor (core_algos.py:1632-1646) ==========
# 在 compute_policy_loss_with_rollout_correction 内部
with torch.no_grad():
    rollout_is_weights, modified_mask, metrics = (
        compute_rollout_correction_and_rejection_mask(
            old_log_prob=log_prob,  # 当前 π_θ（动态）
            rollout_log_prob=rollout_log_prob,
            ...
        )
    )

# 直接应用
pg_losses = -advantages * log_prob * rollout_is_weights
```

**计算时机**：每个 mini-batch 一次（动态计算）  
**使用的 policies**：π_θ vs π_rollout（动态，每个 mini-batch 不同）

---

## 模式对比总结表

| 特性 | Decoupled Mode | Bypass + PPO Loss | Bypass + Policy Gradient |
|------|----------------|-------------------|--------------------------|
| **Policies 数量** | 3 (π_rollout, π_old, π_θ) | 2 (π_rollout=π_old, π_θ) | 2 (π_rollout=π_old, π_θ) |
| **Trainer 计算 old_log_probs** | ✅ 是 (1363行) | ❌ 否（直接复用） | ❌ 否（直接复用） |
| **Trainer 计算 IS weights** | ✅ 是 (1456行) | ❌ 否 | ❌ 否 |
| **添加到 batch** | ✅ `batch["rollout_is_weights"]` | ❌ 无 | ❌ 无 |
| **Actor 提取 IS weights** | ✅ 从 batch 提取 | ❌ 无 | ❌ 无（内部计算） |
| **Actor 计算 IS weights** | ❌ 否（使用预计算） | ❌ 否 | ✅ 是 (1632行) |
| **IS weights 计算时机** | 每个 data batch 一次 | 不计算 | 每个 mini-batch 一次 |
| **IS weights 使用的 policies** | π_old vs π_rollout | N/A | π_θ vs π_rollout |
| **Loss 类型** | PPO with clipping | PPO with clipping | Policy Gradient (REINFORCE) |
| **模型更新位置** | ✅ Actor | ✅ Actor | ✅ Actor |
| **计算成本** | Trainer 前向一次 | 最低（零额外） | Actor 每 mini-batch 计算 |
| **内存占用** | 需传输 IS weights | 最低 | 无需额外内存 |

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
│  内部计算 (rollout_corr_helper.py:544-695):                     │
│  1. log_ratio = old_log_prob - rollout_log_prob                │
│  2. IS weights = truncate(exp(log_ratio), threshold)           │
│  3. Rejection mask = filter_by_threshold(log_ratio)            │
│  4. Veto mask = filter_catastrophic_tokens(log_ratio)          │
│  5. Metrics: KL, PPL, ESS, rejection rate, etc.               │
│                                                                 │
│  添加到 batch (rollout_corr_helper.py:863-864):                │
│  batch.batch["response_mask"] = modified_response_mask         │
│  if rollout_is_weights is not None:                            │
│      batch = batch.union(rollout_is_weights)  # ← 添加!         │
│                                                                 │
│  结果:                                                           │
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
│  - rollout_is_weights ← 已添加!                                 │
│  - response_mask (已应用 RS 和 veto 过滤)                        │
│  - advantages                                                   │
│  - responses, input_ids, attention_mask, etc.                  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: Actor - Select and Extract Data                       │
│  (dp_actor.py:394-395)                                         │
├─────────────────────────────────────────────────────────────────┤
│  # 检测并提取 IS weights                                         │
│  if "rollout_is_weights" in data.batch.keys():                 │
│      select_keys.append("rollout_is_weights")  # ← 从 batch 提取│
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
│  (dp_actor.py:477-494, core_algos.py:966-967)                 │
├─────────────────────────────────────────────────────────────────┤
│  # 提取预计算的 IS weights                                       │
│  rollout_is_weights = model_inputs.get("rollout_is_weights", None)│
│                                                                 │
│  policy_loss_fn = get_policy_loss_fn(loss_mode)  # "vanilla"   │
│  pg_loss, pg_metrics = policy_loss_fn(                         │
│      old_log_prob=old_log_prob,    # π_old                     │
│      log_prob=log_prob,            # π_θ                       │
│      advantages=advantages,                                     │
│      response_mask=response_mask,  # 已过滤                     │
│      rollout_is_weights=rollout_is_weights,  # ← 传递 IS weights│
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
│  内部操作 (rollout_corr_helper.py:936-937):                     │
│  batch["old_log_probs"] = batch["rollout_log_probs"]  # 直接赋值│
│  # ⚡ 零成本！跳过昂贵的 compute_log_prob() 调用                  │
│                                                                 │
│  ❌ 不执行 IS/RS 计算（因为 bypass_recomputing_logprobs=True）   │
│  ❌ batch 中没有 "rollout_is_weights"                           │
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
│  rollout_is_weights = model_inputs.get("rollout_is_weights", None)│
│  # → rollout_is_weights = None (不存在)                         │
│                                                                 │
│  ratio = π_θ / π_rollout  # 注意：这里 π_old = π_rollout!       │
│  pg_losses = PPO_clip(ratio, advantages)                       │
│  # 不应用 IS weights (因为 rollout_is_weights is None)          │
│  pg_loss = mean(pg_losses)                                     │
│  loss.backward()                                                │
│  optimizer.step()                                               │
└─────────────────────────────────────────────────────────────────┘
```

#### Bypass + Policy Gradient

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1-3: Same as above (bypass old_log_prob computation)     │
│  ❌ batch 中没有 "rollout_is_weights"                           │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Actor - Compute IS/RS on-the-fly                      │
│  (core_algos.py:1632-1646)                                     │
├─────────────────────────────────────────────────────────────────┤
│  loss_mode = "rollout_correction"                              │
│                                                                 │
│  在 compute_policy_loss_with_rollout_correction 中:            │
│  # ✅ 在 Actor 内部实时计算 IS weights                          │
│  with torch.no_grad():                                          │
│      rollout_is_weights_proto, modified_mask, metrics = (      │
│          compute_rollout_correction_and_rejection_mask(        │
│              old_log_prob=log_prob,  # π_θ (当前策略，动态)     │
│              rollout_log_prob=rollout_log_prob,  # π_rollout   │
│              response_mask=eos_mask,                            │
│              rollout_is=rollout_is,                             │
│              ...                                                │
│          )                                                      │
│      )                                                          │
│                                                                 │
│  # 提取 IS weights                                              │
│  rollout_is_weights = (rollout_is_weights_proto.batch["rollout_is_weights"]│
│                        if rollout_is_weights_proto else None)  │
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
- 添加到 batch：`rollout_corr_helper.py:863-864`
- Actor 提取：`dp_actor.py:394-395, 479`
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
| 863-864 | 添加 IS weights 到 batch | `batch = batch.union(rollout_is_weights)` |
| 908-948 | `apply_rollout_correction` | Bypass mode 处理 |

### Actor (dp_actor.py)

| 行号 | 功能 | 描述 |
|------|------|------|
| 371-588 | `update_policy` | Actor 主循环 |
| 394-395 | 检测 IS weights | `if "rollout_is_weights" in data.batch.keys()` |
| 479 | 提取 IS weights | `rollout_is_weights = model_inputs.get("rollout_is_weights", None)` |
| 486-494 | 计算 loss | 调用 policy loss 函数，传递 `rollout_is_weights` |

### Policy Loss (core_algos.py)

| 行号 | 函数 | 功能 |
|------|------|------|
| 888-976 | `compute_policy_loss_vanilla` | 标准 PPO loss（支持 IS weights） |
| 966-967 | 应用 IS weights | `pg_losses *= rollout_is_weights` |
| 1555-1689 | `compute_policy_loss_with_rollout_correction` | Bypass + PG: 实时计算 IS/RS |
| 1632-1646 | Actor 内部计算 IS | 调用 `compute_rollout_correction_and_rejection_mask` |
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
- IS/RS 在 trainer 计算一次，actor 在所有 mini-batch 和 epochs 中重复使用（高效）
- 完整的 off-policy 诊断 metrics

**缺点：**
- 需要额外的前向计算（`compute_log_prob`）
- 内存占用略高（存储 IS weights）
- 需要在 trainer 和 actor 之间传输 IS weights

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
- 最高效（零额外计算，无额外内存）
- 仍然保留 PPO clipping 的稳定性

**Bypass + Policy Gradient：**
- 适合需要灵活 IS/RS 调整的场景
- 在 actor 内部实时计算（每个 mini-batch 动态调整，基于最新的 π_θ）
- 无 PPO clipping（可能不太稳定）
- 计算开销更大（每个 mini-batch 都计算 IS/RS）

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
- 希望动态调整 IS weights（每个 mini-batch 重新计算，基于最新的 π_θ）
- 不需要 PPO clipping 的稳定性约束

**权衡：**
- ✅ 更灵活的 off-policy correction（动态）
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

### Q5: Trainer 计算的 IS weights 真的会被 Actor 使用吗？

**A:** 是的！这是一个重要澄清：

**在 Decoupled Mode 中：**
1. ✅ Trainer 计算 IS weights（第1456行）
2. ✅ 添加到 batch：`batch["rollout_is_weights"]`（第863-864行）
3. ✅ Actor 从 batch 中提取（第394-395, 479行）
4. ✅ Actor 应用到 loss 计算（第966-967行）
5. ✅ 影响梯度和模型更新

**关键点：**
- Trainer 负责"准备数据"（计算并添加 IS weights 到 batch）
- Actor 负责"使用数据"（提取 IS weights 并应用到梯度计算）
- IS weights 在所有 mini-batch 和 epochs 中被重复使用
- 模型参数的真正更新在 Actor 中完成（`optimizer.step()`）

**这是典型的分布式训练架构**：
- Driver Process（Trainer）：协调、数据准备、统计
- Worker Process（Actor）：密集计算、梯度更新、模型训练

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

---

## 深入理解：Rollout Correction 解决的两类偏差

### 偏差 1: 模型更新滞后偏差 (Model Staleness Bias)

**问题描述：**

在 off-policy RL 中，我们训练时使用的是从旧版本模型生成的轨迹数据：
- **π_rollout**: 生成轨迹时使用的模型（如 checkpoint t-10）
- **π_old**: 当前训练开始时的模型（如 checkpoint t）
- **π_θ**: 正在更新的模型参数（在 mini-batch 中不断变化）

**偏差来源：**
```
π_rollout (旧版本) ≠ π_old (当前版本)
```

由于模型在 rollout 和 training 之间经历了多次更新，两个策略的输出分布已经不同。

**数学表达：**
```
真实期望梯度（on-policy）:
∇J = E_{s,a ~ π_old} [∇ log π_θ(a|s) * A^{π_old}(s,a)]

实际计算梯度（off-policy，无校正）:
∇J_biased = E_{s,a ~ π_rollout} [∇ log π_θ(a|s) * A^{π_rollout}(s,a)]
                                    ↑ 偏差！不同分布

正确的梯度估计（有 IS 校正）:
∇J_corrected = E_{s,a ~ π_rollout} [w(s,a) * ∇ log π_θ(a|s) * A^{π_old}(s,a)]
where w(s,a) = π_old(a|s) / π_rollout(a|s)  ← IS weight
```

**Rollout Correction 的解决方案：**

1. **重新计算 old_log_probs**（`ray_trainer.py:1363-1386`）:
   ```python
   old_log_prob = actor_rollout_wg.compute_log_prob(batch)
   # 使用当前版本模型计算 π_old(a|s)
   ```

2. **计算 Importance Sampling weights**:
   ```python
   log_ratio = old_log_prob - rollout_log_prob
   w = exp(log_ratio) = π_old / π_rollout
   w_truncated = clip(w, 1/τ, τ)  # 截断避免高方差
   ```

3. **应用到梯度计算**:
   ```python
   pg_losses = PPO_loss(ratio, advantages)
   pg_losses = pg_losses * w_truncated  # IS 校正
   ```

**效果：**
- ✅ 将 off-policy 数据转换为 on-policy 梯度估计
- ✅ 允许使用较旧的 rollout 数据（提高数据效率）
- ✅ 减少 policy distribution shift 的负面影响

---

### 偏差 2: 推理-训练精度不匹配偏差 (Precision Mismatch Bias)

**问题描述：**

即使使用完全相同的模型参数和 checkpoint，由于推理引擎和训练框架的实现差异，计算出的 log_probs 也会不同：

| 组件 | 实现 | 数值精度 |
|------|------|---------|
| **Rollout (vLLM/SGLang)** | 优化的推理引擎 | BFloat16 |
| **Training (FSDP/Megatron)** | 训练框架 | FP32 (或 mixed precision) |

**偏差来源：**
```
π_rollout_vLLM(a|s) ≠ π_old_FSDP(a|s)  （即使参数相同！）
```

**为什么会有差异？**

1. **浮点精度差异**:
   ```
   BFloat16:  1 sign bit + 8 exponent bits + 7 mantissa bits
   FP16:      1 sign bit + 5 exponent bits + 10 mantissa bits
   FP32:      1 sign bit + 8 exponent bits + 23 mantissa bits
   ```

2. **累积舍入误差**:
   ```python
   # 自回归生成（逐 token 计算）
   for t in range(seq_len):
       logits[t] = matmul(hidden[t], W)  # 每次都有舍入误差
       log_prob[t] = log_softmax(logits[t])  # 进一步累积

   # BF16 每个 token 的舍入误差: ~10^-3
   # 序列长度 100: 累积误差可能达到 0.1 - 0.5（显著！）
   ```

3. **不同的算子实现**:
   - vLLM: 使用 PagedAttention、FlashAttention、自定义 CUDA kernels
   - FSDP: 使用 PyTorch 标准算子
   - 即使在相同精度下，计算顺序和融合方式也可能不同

**真实影响示例**:

```python
# 相同模型，不同实现
rollout_log_prob (vLLM BF16):  [-0.523, -1.234, -0.891, ...]
old_log_prob (FSDP FP32):      [-0.518, -1.229, -0.887, ...]
                                  ↑       ↑       ↑
                              差异小但累积后显著

# 序列级别的 log ratio
log_ratio = sum(old_log_prob - rollout_log_prob)
          = sum([0.005, 0.005, 0.004, ...])  # 100 tokens
          = 0.5  # 累积后的差异

# 对应的 ratio
ratio = exp(0.5) ≈ 1.65  # 16.5% 的概率差异！
```

**Rollout Correction 的解决方案：**

**方法 1：使用相同精度 + Rollout Correction (推荐)**

1. **重新计算 old_log_probs**（使用训练环境）:
   ```python
   # 使用 FSDP/Megatron（FP32）重新计算
   old_log_prob = actor_rollout_wg.compute_log_prob(batch)
   # 现在 old_log_prob 和 training 环境完全匹配！
   ```

2. **计算 IS weights 纠正剩余差异**:
   ```python
   log_ratio = old_log_prob - rollout_log_prob
   # 纠正 vLLM BF16 和 FSDP FP32 的差异
   w = clip(exp(log_ratio), 1/τ, τ)
   ```

3. **优势**:
   - ✅ 完全消除精度不匹配
   - ✅ 同时纠正模型更新滞后和精度偏差
   - ✅ 获得准确的 off-policy metrics

**方法 2：统一使用 FP16 (最佳实践，无需 Rollout Correction)**

**为什么 FP16 优于 BFloat16？**

| 特性 | BFloat16 | FP16 |
|------|----------|------|
| **Mantissa bits** | 7 | 10 |
| **精度** | ~10^-3 | ~10^-4 |
| **相对精度** | 1x (baseline) | 8x better |
| **数值范围** | 更大 (8 exponent bits) | 较小 (5 exponent bits) |
| **适用场景** | 前向传播、激活值 | 精确计算、RL log_probs |

**FP16 的精度优势：**

```
BFloat16 mantissa: 7 bits
→ 2^7 = 128 discrete values per exponent
→ 相对误差: 1/128 ≈ 0.78% per operation

FP16 mantissa: 10 bits
→ 2^10 = 1024 discrete values per exponent
→ 相对误差: 1/1024 ≈ 0.098% per operation

改善: 1024/128 = 8x 更精确
```

**累积效应（序列长度 100）：**

```python
# BFloat16 累积误差
per_token_error_bf16 = 0.0078  # 0.78%
cumulative_error_bf16 = 100 * 0.0078 = 0.78
# 对应 ratio = exp(0.78) ≈ 2.18 (118% 差异！)

# FP16 累积误差
per_token_error_fp16 = 0.00098  # 0.098%
cumulative_error_fp16 = 100 * 0.00098 = 0.098
# 对应 ratio = exp(0.098) ≈ 1.10 (10% 差异)

# 改善: 2.18 / 1.10 ≈ 2x 更接近理想值 1.0
```

**实证研究支持：**

根据最新研究（2024-2025）：
- **arXiv:2510.26788**: "FP16 vs BFloat16 for RLHF"
  - 发现 FP16 在 RL 训练中几乎完全消除了 training-inference mismatch
  - BF16 在长序列上累积误差显著影响 advantage 估计

- **Ian Barber's Analysis** (2025/11/03):
  - "Let's All Switch to FP16"
  - 详细对比了 FP16 vs BF16 在 LLM 推理中的精度差异
  - 建议对于需要精确 log_probs 的应用使用 FP16

**配置建议：**

```yaml
# 推荐配置：FP16 everywhere
actor_rollout_ref:
  rollout:
    name: "vllm"
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.5
    enforce_eager: true
    dtype: "float16"  # ← 使用 FP16!

  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 2048
    ppo_micro_batch_size_per_gpu: 2
    fsdp_config:
      param_dtype: "fp32"      # 训练时仍用 FP32（精度）
      reduce_dtype: "fp32"
      buffer_dtype: "fp32"

algorithm:
  rollout_correction:
    bypass_mode: true  # ← 如果使用 FP16，可以启用 bypass!
    use_policy_gradient: false
```

**如果必须使用 BFloat16（如 NPU、特定硬件限制）：**

```yaml
# BFloat16 + Rollout Correction
actor_rollout_ref:
  rollout:
    dtype: "bfloat16"  # 硬件限制

algorithm:
  rollout_correction:
    bypass_mode: false  # ← 必须使用 decoupled mode!
    rollout_is: "token"
    rollout_is_threshold: 2.0  # 纠正精度偏差
    rollout_rs: "token"
    rollout_rs_threshold: 2.0
```

---

### 两种偏差的关系

**重要理解：Rollout Correction 同时解决两种偏差！**

```python
# 完整的偏差分解
rollout_log_prob = π_rollout_vLLM_BF16(a|s)  # 旧模型 + vLLM + BF16
old_log_prob     = π_old_FSDP_FP32(a|s)      # 新模型 + FSDP + FP32

log_ratio = old_log_prob - rollout_log_prob
          = [π_old(a|s) - π_rollout(a|s)]  ← 模型更新偏差
            + [precision_FSDP_FP32 - precision_vLLM_BF16]  ← 精度偏差

IS weight = exp(log_ratio)  # 同时纠正两种偏差！
```

**Decoupled Mode 的双重校正：**

1. **重新计算 old_log_probs**:
   - 消除精度实现偏差（使用训练环境重新计算）
   - 获得当前模型的准确概率

2. **计算 IS weights**:
   - 消除模型更新偏差（通过 importance sampling）
   - 将 off-policy 数据转换为 on-policy 梯度

**效果对比：**

| 方法 | 模型更新偏差 | 精度偏差 | 计算成本 |
|------|------------|---------|---------|
| **无校正 (Bypass + BF16)** | ❌ 存在 | ❌ 存在 | 最低 |
| **Bypass + FP16** | ❌ 存在（轻微） | ✅ 几乎消除 | 低 |
| **Decoupled + BF16** | ✅ 校正 | ✅ 校正 | 中等（额外前向） |
| **Decoupled + FP16** | ✅ 校正 | ✅ 完全消除 | 中等 |

**最佳实践建议：**

1. **首选**：FP16 everywhere + Bypass mode
   - 精度偏差几乎为零
   - 计算成本最低
   - 仅在模型更新间隔较大时需要 Decoupled mode

2. **次选**：Decoupled mode + FP16
   - 完全消除两种偏差
   - 适合严格的 off-policy 场景
   - 额外计算成本可接受

3. **不推荐**：Bypass + BFloat16（除非硬件限制）
   - 两种偏差都存在
   - 可能导致训练不稳定
   - 仅在频繁更新且精度要求低时使用

4. **权宜之计**：Decoupled + BFloat16（硬件限制）
   - 通过 IS weights 部分纠正精度偏差
   - 需要仔细调整 threshold
   - 监控 `rollout_corr/rollout_kl` 和 ESS 指标

---

## 变更历史

- **2025-01-初始**: 初始文档创建
- **2025-01-更新1**: 澄清 Trainer 计算的 IS weights 会传递给 Actor 使用
- **2025-01-更新2**: 添加两种偏差的深入分析（模型滞后 vs 精度不匹配）、FP16 vs BFloat16 对比、配置建议
- 本文档基于 veRL codebase 分析生成