# Rollout Correction 中的重要性采样(IS)和拒绝采样(RS)详解

**作者**: Claude Code
**创建时间**: 2025-11-26
**适用版本**: verl 0.6.1+

---

## 目录

1. [核心问题：Off-Policy RL](#核心问题off-policy-rl)
2. [重要性采样 (Importance Sampling, IS)](#重要性采样-importance-sampling-is)
3. [拒绝采样 (Rejection Sampling, RS)](#拒绝采样-rejection-sampling-rs)
4. [Veto机制（灾难性token否决）](#veto机制灾难性token否决)
5. [完整流程](#完整流程)
6. [使用场景](#使用场景)
7. [代码实现细节](#代码实现细节)
8. [配置示例](#配置示例)
9. [参考资料](#参考资料)

---

## 核心问题：Off-Policy RL

在 LLM 强化学习训练中，存在**三个策略**：

1. **π_rollout** - 用于生成轨迹的策略（如 vLLM BF16）
2. **π_old** - 训练时的参考策略（如 FSDP FP32，固定用于一个batch）
3. **π_θ** - 当前正在更新的策略

### Off-Policy 问题的来源

当 `π_rollout ≠ π_old` 时，训练数据分布与训练策略分布不匹配，导致 **off-policy 问题**。常见原因：

1. **策略不匹配**（精度差异）
   - Rollout: vLLM BF16
   - Training: FSDP FP32
   - 即使权重相同，数值计算也会产生差异

2. **模型更新延迟**
   - Rollout使用旧checkpoint
   - Training已经更新多轮
   - 异步训练系统中常见

3. **后端实现差异**
   - vLLM vs SGLang vs FSDP vs Megatron
   - 不同的注意力实现
   - 不同的采样策略

### 问题影响

- 梯度估计偏差
- 训练不稳定
- **策略崩溃** (Policy Collapse)
- 性能下降

**解决方案**: Rollout Correction 使用 IS 权重和 RS 过滤来校正分布偏移。

---

## 重要性采样 (Importance Sampling, IS)

### 数学原理

IS 通过权重重新加权样本，校正分布偏移：

```
IS权重: ρ = π_old(token) / π_rollout(token)

期望梯度校正:
E_π_old[∇L] = E_π_rollout[ρ · ∇L]

实际应用:
∇L = E_π_rollout[ ρ · ∇_θ log π_θ(token) · advantage ]
```

**关键思想**: 用 π_rollout 收集的数据，通过重加权，估计 π_old 下的期望梯度。

### 实现逻辑

文件位置: `verl/trainer/ppo/rollout_corr_helper.py:318-427`

#### **Step 1: 计算 log ratio**

```python
# 输入:
#   old_log_prob: log π_old(token) - Shape (batch_size, seq_length)
#   rollout_log_prob: log π_rollout(token) - Shape (batch_size, seq_length)
#   response_mask: 有效token标记 (1=有效, 0=padding)

log_ratio = old_log_prob - rollout_log_prob  # log(π_old / π_rollout)
```

**数值稳定性**: 在对数空间计算，避免直接除法导致的溢出/下溢。

#### **Step 2: 聚合级别选择**

##### **Token-level IS** (`rollout_is="token"`)

```python
# 每个token独立计算权重
log_ratio_safe = torch.clamp(log_ratio, min=-20, max=20)  # 防止溢出
rollout_is_weights = torch.exp(log_ratio_safe)
# Shape: (batch_size, seq_length)
```

**特性**:

- ✅ **低方差**: 每个token独立，方差较小
- ❌ **有偏估计**: 数学上不是无偏估计
- 📊 **阈值范围**: 1.5 - 5.0（典型值）

**适用场景**:

- 策略差异较小（如仅精度差异）
- 需要稳定训练
- 短序列任务

##### **Sequence-level IS** (`rollout_is="sequence"`)

```python
# 序列权重 = 所有token权重的乘积
# ρ_seq = ∏_t ρ_t = exp(∑_t log ρ_t)
log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1)  # (batch_size, 1)
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-20, max=20)
rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)
# 整个序列的所有token共享相同的权重
```

**特性**:

- ✅ **无偏估计**: 数学上无偏
- ❌ **高方差**: 序列越长，方差越大
- 📊 **阈值范围**: 2.0 - 10.0（典型值）

**适用场景**:

- 策略差异较大（如模型更新延迟）
- 长序列任务需谨慎
- 需要理论保证的场景

**数学细节**:

```text
序列权重 ρ_seq = ∏_{t=1}^T [π_old(t) / π_rollout(t)]
            = exp(∑_{t=1}^T log[π_old(t) / π_rollout(t)])
            = exp(∑_{t=1}^T [log π_old(t) - log π_rollout(t)])
```

#### **Step 3: 截断 (Truncation)**

```python
# TIS (Truncated Importance Sampling) - 减少方差
rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)
```

**作用**:

- 限制极端权重，防止单个样本主导梯度
- **引入偏差**，但**显著降低方差**
- 实践中效果优于无截断IS

**例子**:

```python
# threshold=2.0
原始权重: [0.5, 1.0, 3.5, 0.1, 10.0]
截断后:   [0.5, 1.0, 2.0, 0.1, 2.0]
```

权重 > 2.0 的样本被限制为 2.0，防止过度影响梯度。

#### **Step 4: 安全边界**

```python
SAFETY_BOUND = 20.0  # exp(20) ≈ 4.85×10^8, exp(-20) ≈ 2×10^-9

log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
```

**原因**:

- `exp(20) ≈ 4.85亿` - 上界防止溢出
- `exp(-20) ≈ 2×10^-9` - 下界防止下溢
- 即使在 FP16 下也安全

#### **Step 5: 去除梯度并应用mask**

```python
rollout_is_weights = rollout_is_weights.detach()  # IS权重不参与梯度计算
rollout_is_weights = rollout_is_weights * response_mask  # padding位置权重为0
```

**理论依据**:

- IS权重改变的是**测度**（measure），不是目标函数
- 权重本身不应该被优化
- 详见数学推导: `docs/algo/rollout_corr_math.md` §3.2.2

#### **Step 6: 可选的批归一化**

```python
if rollout_is_batch_normalize:
    if rollout_is == "token":
        # Token-level: 对所有token权重归一化
        weights_mean = masked_mean(rollout_is_weights, response_mask)
    elif rollout_is == "sequence":
        # Sequence-level: 对序列权重归一化
        seq_weights_mean = masked_mean(rollout_is_weights, response_mask, axis=-1)
        weights_mean = seq_weights_mean.mean()

    if weights_mean > 1e-8:
        rollout_is_weights = rollout_is_weights / weights_mean  # 归一化到均值=1.0
```

**效果**:

- 确保每个batch的平均权重为1.0
- 减少batch间的权重尺度差异
- **仅影响权重值**，不改变拒绝采样逻辑
- 默认: `False`（使用原始截断权重）

### IS权重的使用

在策略梯度中应用：

```python
# verl/trainer/ppo/core_algos.py
policy_loss = -advantages * log_probs  # 基础损失
if rollout_is_weights is not None:
    policy_loss = policy_loss * rollout_is_weights  # 应用IS权重
```

**效果**: 高权重样本对梯度贡献更大，低权重样本贡献更小。

---

## 拒绝采样 (Rejection Sampling, RS)

### 数学原理

RS 通过**硬过滤**剔除异常样本，而不是重新加权：

```text
决策规则:
if ρ ∈ [lower_threshold, upper_threshold]:
    保留样本 (mask = 1)
else:
    拒绝样本 (mask = 0)
```

**关键区别**:

- IS: 连续权重调整（0.1x, 0.5x, 2.0x, ...）
- RS: 二值决策（保留 or 拒绝）

### 实现逻辑

文件位置: `verl/trainer/ppo/rollout_corr_helper.py:82-193`

#### **Step 1: 计算 IS 权重**

与IS相同，先计算 log_ratio 和权重。

#### **Step 2: 聚合级别选择**

##### **Token-level RS** (`rollout_rs="token"`)

```python
# 每个token独立判断
log_ratio_safe = torch.clamp(log_ratio, min=-20, max=20)
rollout_is_weights = torch.exp(log_ratio_safe)

# 生成mask: 1=保留, 0=拒绝
mask = (rollout_is_weights >= lower_threshold) & \
       (rollout_is_weights <= upper_threshold)
mask = mask.float()
```

**效果**:

- 逐token判断，拒绝异常token
- 序列中**部分token**可能被mask掉
- 剩余token仍参与训练

**例子**:

```python
# threshold_lower=0.5, threshold_upper=2.0
权重:  [0.8, 1.5, 3.0, 0.3, 1.2]
Mask:  [1.0, 1.0, 0.0, 0.0, 1.0]  # 第3、4个token被拒绝
```

##### **Sequence-level RS** (`rollout_rs="sequence"`)

```python
# 序列级权重（所有token共享）
log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1)  # (batch_size, 1)
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-20, max=20)
rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)

# 生成mask（广播到整个序列）
mask = (rollout_is_weights >= lower_threshold) & \
       (rollout_is_weights <= upper_threshold)
```

**效果**:

- 基于整个序列的权重判断
- **要么全保留，要么全拒绝**（整个序列）
- 更激进的过滤策略

**例子**:

```python
# 序列权重 = 2.5（超过threshold_upper=2.0）
序列中所有token: mask = 0.0  # 整个序列被拒绝
```

##### **Geometric RS** (`rollout_rs="geometric"`)

```python
# 几何平均 = exp(mean(log_ratio))
log_ratio_mean = masked_mean(log_ratio, response_mask, axis=-1)  # (batch_size, 1)
log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-20, max=20)
rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)

mask = (rollout_is_weights >= lower_threshold) & \
       (rollout_is_weights <= upper_threshold)
```

**特性**:

- 使用几何平均而非算术平均
- **对异常值极度敏感**
- 任何一个极端token都会显著影响几何平均

**数学细节**:

```text
几何平均: (∏_{t=1}^T ρ_t)^(1/T) = exp((1/T) ∑_{t=1}^T log ρ_t)

vs 算术平均: (1/T) ∑_{t=1}^T ρ_t
```

**阈值**:

- 必须非常接近1.0（如 **1.0002 - 1.001**，即 ±0.02% - ±0.1%）
- 配合 veto 机制使用
- 最严格的异常检测

**适用场景**:

- 需要极高质量样本
- 零容忍异常token
- 通常与 veto 机制配合

#### **Step 3: 计算阈值**

```python
upper_threshold = rollout_rs_threshold  # 用户指定
lower_threshold = rollout_rs_threshold_lower if rollout_rs_threshold_lower is not None \
                  else 1.0 / upper_threshold  # 默认为上阈值的倒数
```

**例子**:

```python
upper_threshold = 2.0
lower_threshold = 1.0 / 2.0 = 0.5
# 保留范围: [0.5, 2.0]
```

#### **Step 4: 应用拒绝mask**

```python
modified_response_mask = response_mask * mask
```

**效果**:

- 原本有效的token (response_mask=1) 如果被RS拒绝 (mask=0)，最终mask=0
- 被拒绝的token**不参与损失计算**和**梯度更新**

#### **Step 5: 统计指标**

```python
# Token级别拒绝率
metrics["rollout_rs_masked_fraction"] = masked_mean(1 - mask, response_mask).item()

# Sequence级别拒绝率
if rollout_rs == "token":
    # 如果序列中有任何token被拒绝，算作序列被拒绝
    seq_has_masked = masked_sum(1 - mask, response_mask, axis=-1) > 0
    metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
else:
    # Sequence/geometric: 检查第一个token的mask（所有token相同）
    metrics["rollout_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()
```

---

## Veto机制（灾难性token否决）

### 原理

独立于IS和RS的**额外安全机制**，用于捕获极端异常token。

文件位置: `verl/trainer/ppo/rollout_corr_helper.py:648-673`

### 实现

```python
if rollout_token_veto_threshold is not None:
    # 计算veto阈值（对数空间）
    log_veto_threshold = torch.log(torch.tensor(rollout_token_veto_threshold))

    # 找到灾难性token（权重极低的token）
    # 检查的是**未截断**的log_ratio，在安全边界之前
    catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask.bool()

    # 如果序列中有**任何一个**灾难性token，拒绝整个序列
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)  # (batch_size, 1)

    # 生成veto mask (0=拒绝序列, 1=保留序列)
    veto_mask = (~has_catastrophic).float()  # (batch_size, 1)

    # 应用veto到response mask（覆盖之前的RS mask）
    modified_response_mask = modified_response_mask * veto_mask
```

### 关键特性

**1. 独立性**

- 与 `rollout_is` 和 `rollout_rs` 设置**完全独立**
- 即使 IS/RS 都是 null，veto 仍可单独启用

**2. 序列级别拒绝**

- 只要有一个token是灾难性的，**整个序列**被拒绝
- 防止部分极端token污染训练

**3. 使用未截断的ratio**

- 在安全边界 clamp 之前检查
- 捕获真正的极端outlier

**4. 典型阈值**

- `1e-4` (0.0001): token比rollout策略低10,000倍
- `1e-6` (0.000001): token比rollout策略低1,000,000倍

### 使用场景

```python
# 配置示例
algorithm:
  rollout_correction:
    rollout_token_veto_threshold: 1e-4  # 启用veto
```

**适用于**:

- 高风险场景（生产环境）
- 模型更新延迟严重
- 发现过catastrophic forgetting
- 需要额外保护措施

**统计指标**:

```python
metrics["rollout_is_veto_fraction"]  # 被veto的序列比例
metrics["rollout_is_catastrophic_token_fraction"]  # 灾难性token比例
```

---

## 完整流程

### 流程图

```text
输入: old_log_prob, rollout_log_prob, response_mask
  │
  ├─ old_log_prob: π_old的log概率 (batch_size, seq_length)
  ├─ rollout_log_prob: π_rollout的log概率 (batch_size, seq_length)
  └─ response_mask: 有效token标记 (1=有效, 0=padding)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ [Step 1] 计算 log ratio                                │
│   log_ratio = old_log_prob - rollout_log_prob          │
│   log_ratio = log(π_old / π_rollout)                   │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ [Step 2] 重要性采样 IS（如果 rollout_is != null）      │
│                                                         │
│  Token-level (rollout_is="token"):                     │
│    ├─ log_ratio_safe = clamp(log_ratio, -20, 20)       │
│    ├─ weights = exp(log_ratio_safe)                    │
│    ├─ weights = weights.clamp(max=threshold)  # TIS    │
│    ├─ weights = weights.detach()                       │
│    └─ weights = weights * response_mask                │
│                                                         │
│  Sequence-level (rollout_is="sequence"):               │
│    ├─ log_sum = masked_sum(log_ratio, mask, axis=-1)   │
│    ├─ log_sum_safe = clamp(log_sum, -20, 20)           │
│    ├─ weights = exp(log_sum_safe).expand_as(log_ratio) │
│    ├─ weights = weights.clamp(max=threshold)           │
│    ├─ weights = weights.detach()                       │
│    └─ weights = weights * response_mask                │
│                                                         │
│  输出: rollout_is_weights (batch_size, seq_length)      │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ [Step 3] 拒绝采样 RS（如果 rollout_rs != null）        │
│                                                         │
│  Token-level (rollout_rs="token"):                     │
│    ├─ weights = exp(clamp(log_ratio, -20, 20))         │
│    └─ mask = (weights >= lower) & (weights <= upper)   │
│       → 逐token判断，部分mask                           │
│                                                         │
│  Sequence-level (rollout_rs="sequence"):               │
│    ├─ log_sum = masked_sum(log_ratio, mask, axis=-1)   │
│    ├─ weights = exp(clamp(log_sum, -20, 20))           │
│    └─ mask = (weights >= lower) & (weights <= upper)   │
│       → 整序列判断，全部mask                            │
│                                                         │
│  Geometric (rollout_rs="geometric"):                   │
│    ├─ log_mean = masked_mean(log_ratio, mask, axis=-1) │
│    ├─ weights = exp(clamp(log_mean, -20, 20))          │
│    └─ mask = (weights >= lower) & (weights <= upper)   │
│       → 几何平均，对异常值极度敏感                       │
│                                                         │
│  modified_response_mask = response_mask * mask          │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ [Step 4] Veto机制（如果 veto_threshold != null）       │
│                                                         │
│  检测灾难性token:                                       │
│    ├─ catastrophic = (log_ratio < log(veto_threshold)) │
│    ├─ has_catastrophic = catastrophic.any(dim=-1)      │
│    └─ veto_mask = ~has_catastrophic                    │
│                                                         │
│  应用veto（覆盖RS结果）:                                │
│    └─ modified_response_mask *= veto_mask              │
│                                                         │
│  如果序列有任何灾难性token → 整个序列被拒绝             │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ [Step 5] 计算诊断指标                                  │
│                                                         │
│  Off-policy指标:                                        │
│    ├─ KL散度: KL(π_rollout || π_old)                   │
│    ├─ 困惑度: PPL_old, PPL_rollout                      │
│    ├─ χ²散度: E[ρ²] - 1 (token/sequence)               │
│    └─ 有效样本量: ESS = 1 / E[(w/E[w])²]               │
│                                                         │
│  IS指标:                                                │
│    ├─ rollout_is_mean/max/min/std                      │
│    ├─ rollout_is_ratio_fraction_high/low               │
│    └─ rollout_is_seq_* (序列级统计)                     │
│                                                         │
│  RS指标:                                                │
│    ├─ rollout_rs_masked_fraction (token拒绝率)         │
│    ├─ rollout_rs_seq_masked_fraction (序列拒绝率)       │
│    └─ rollout_rs_* (统计信息)                           │
│                                                         │
│  Veto指标:                                              │
│    ├─ rollout_is_veto_fraction                         │
│    └─ rollout_is_catastrophic_token_fraction           │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 输出                                                    │
│  ├─ rollout_is_weights: IS权重 (或 None)                │
│  ├─ modified_response_mask: 更新后的mask                │
│  └─ metrics: 所有诊断指标 (带 "rollout_corr/" 前缀)     │
└─────────────────────────────────────────────────────────┘
```

### 调用入口

**主函数**: `compute_rollout_correction_and_rejection_mask()`

```python
# verl/trainer/ppo/rollout_corr_helper.py:545-696
rollout_is_weights_proto, modified_response_mask, metrics = \
    compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=config.rollout_is,
        rollout_is_threshold=config.rollout_is_threshold,
        rollout_rs=config.rollout_rs,
        rollout_rs_threshold=config.rollout_rs_threshold,
        rollout_rs_threshold_lower=config.rollout_rs_threshold_lower,
        rollout_token_veto_threshold=config.rollout_token_veto_threshold,
        rollout_is_batch_normalize=config.rollout_is_batch_normalize,
    )
```

**集成点**:

- `verl/trainer/ppo/ray_trainer.py:1446-1462` - Decoupled模式
- `verl/trainer/ppo/ray_trainer.py:1344-1360` - Bypass模式
- `verl/workers/actor/dp_actor.py` - 分布式worker

---

## 使用场景

### 场景对比表

| 配置方法 | IS级别 | RS级别 | Veto | 用途 | 特点 |
|---------|-------|-------|------|------|------|
| `decoupled_token_is()` | token | null | null | 低方差校正 | 有偏但稳定 |
| `decoupled_seq_is()` | sequence | null | null | 无偏校正 | 无偏但高方差 |
| `decoupled_seq_is_rs()` | sequence | sequence | null | IS + RS双重保护 | 平衡方案 |
| `decoupled_geo_rs()` | null | geometric | 1e-4 | 最严格过滤 | 零容忍异常 |
| `ppo_is_bypass()` | token | null | null | 性能优化 | 跳过old_log_prob |
| `pg_is()` | sequence | null | null | 策略梯度+IS | 无PPO裁剪 |
| `pg_rs()` | null | geometric | 1e-4 | 策略梯度+RS | 最激进 |
| `disabled()` | null | null | null | 仅监控 | 不应用校正 |

### 详细使用建议

#### 1. **策略差异较小**（如仅精度差异 BF16 vs FP32）

```python
config = RolloutCorrectionConfig.decoupled_token_is(threshold=2.0)
```

**原因**:

- Token-level IS 足以处理小的分布偏移
- 低方差，训练稳定
- 计算开销小

#### 2. **策略差异中等**（如模型更新1-2轮）

```python
config = RolloutCorrectionConfig.decoupled_seq_is_rs(
    is_threshold=2.0,
    rs_threshold=2.0,
)
```

**原因**:

- Sequence-level IS 提供无偏校正
- RS 过滤掉极端outlier
- 平衡准确性和稳定性

#### 3. **策略差异较大**（如模型更新多轮、异步训练）

```python
config = RolloutCorrectionConfig.decoupled_geo_rs(
    rs_threshold=1.001,  # ±0.1%
    veto_threshold=1e-4,
)
```

**原因**:

- 几何RS对异常值极度敏感
- Veto机制提供额外保护
- 严格过滤，保证样本质量

#### 4. **性能优化**（减少计算开销）

```python
config = RolloutCorrectionConfig.ppo_is_bypass(threshold=2.0)
```

**原因**:

- Bypass模式跳过 old_log_prob 计算
- 节省一次forward pass
- 适用于策略差异很小的场景

#### 5. **研究/实验**（使用纯策略梯度）

```python
config = RolloutCorrectionConfig.pg_is(threshold=2.0)
# 或
config = RolloutCorrectionConfig.pg_rs(
    rs_threshold=1.001,
    veto_threshold=1e-4,
)
```

**原因**:

- 去除PPO裁剪，使用纯策略梯度
- 配合IS/RS校正分布偏移
- 适合算法研究

### 关键区别总结

**IS vs RS**:

- **IS权重**: 连续重加权，减少梯度估计方差
- **RS mask**: 二值过滤，完全剔除极端outlier
- **可以同时使用**: IS负责variance reduction，RS负责outlier rejection

**Token-level vs Sequence-level**:

- **Token-level**: 低方差，有偏，适合小差异
- **Sequence-level**: 无偏，高方差，适合大差异

**Geometric RS**:

- 最敏感的异常检测
- 必须配合极小阈值（~1.001）
- 通常与veto配合使用

---

## 代码实现细节

### 关键函数

#### 1. `compute_rollout_correction_weights()`

**位置**: `verl/trainer/ppo/rollout_corr_helper.py:318-427`

**功能**: 计算IS权重（仅IS，不含RS）

**返回**:

```python
rollout_is_weights: torch.Tensor  # (batch_size, seq_length)
metrics: dict[str, float]  # IS统计指标
```

#### 2. `compute_rollout_rejection_mask()`

**位置**: `verl/trainer/ppo/rollout_corr_helper.py:82-193`

**功能**: 计算拒绝采样mask（仅RS，不含IS）

**返回**:

```python
modified_response_mask: torch.Tensor  # (batch_size, seq_length)
metrics: dict[str, float]  # RS统计指标
```

#### 3. `compute_rollout_correction_and_rejection_mask()`

**位置**: `verl/trainer/ppo/rollout_corr_helper.py:545-696`

**功能**: 统一接口，计算IS + RS + Veto

**返回**:

```python
rollout_is_weights_proto: Optional[DataProto]  # IS权重（或None）
modified_response_mask: torch.Tensor  # 最终mask
metrics_scalar: dict[str, float]  # 所有指标（带"rollout_corr/"前缀）
```

#### 4. `compute_offpolicy_metrics()`

**位置**: `verl/trainer/ppo/rollout_corr_helper.py:699+`

**功能**: 计算off-policy诊断指标

**指标**:

- `kl`: KL散度 KL(π_rollout || π_old)
- `k3_kl`: K3 KL估计器（更稳定）
- `training_ppl`: 训练策略困惑度
- `rollout_ppl`: Rollout策略困惑度
- `log_ppl_diff`: log困惑度差异
- `ppl_ratio`: 困惑度比值
- `chi2_token`: Token级χ²散度
- `chi2_seq`: Sequence级χ²散度

### 辅助函数

```python
# verl/utils/torch_functional.py
masked_sum(tensor, mask, axis)  # 带mask的求和
masked_mean(tensor, mask, axis)  # 带mask的求均值
```

### 数值稳定性技巧

**1. 对数空间计算**

```python
# ❌ 不稳定
ratio = π_old / π_rollout  # 可能溢出/下溢

# ✅ 稳定
log_ratio = log(π_old) - log(π_rollout)
ratio = exp(clamp(log_ratio, -20, 20))
```

**2. 安全边界**

```python
SAFETY_BOUND = 20.0
# exp(20) ≈ 4.85×10^8 - 上界
# exp(-20) ≈ 2×10^-9 - 下界
```

**3. 阈值检查在对数空间**

```python
# 精确检查（避免exp后的精度损失）
log_threshold = torch.log(torch.tensor(threshold))
exceeds = log_ratio > log_threshold  # 对数空间比较
```

**4. 方差计算的clamp**

```python
# 避免极端值平方后溢出
weights_for_std = weights.clamp(min=lower, max=upper)
var = masked_mean(weights_for_std.square(), mask) - mean.square()
std = torch.sqrt(torch.clamp(var, min=0.0))  # 防止负值
```

---

## 配置示例

### 1. 基础配置（Token-level IS）

```yaml
# verl/trainer/config/ppo_trainer.yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: null
    rollout_is_batch_normalize: false
    bypass_mode: false
    use_policy_gradient: false

# 必需：启用log prob计算
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### 2. 推荐配置（Sequence IS + RS）

```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: sequence
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: null  # 自动计算为 1/2.0 = 0.5
    rollout_is_batch_normalize: false
    bypass_mode: false
    use_policy_gradient: false

actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### 3. 最严格配置（Geometric RS + Veto）

```yaml
algorithm:
  rollout_correction:
    rollout_is: null  # 不使用IS权重
    rollout_rs: geometric
    rollout_rs_threshold: 1.001  # ±0.1%
    rollout_rs_threshold_lower: null  # 自动计算
    rollout_token_veto_threshold: 1e-4  # 10,000x阈值
    rollout_is_batch_normalize: false
    bypass_mode: false
    use_policy_gradient: false

actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### 4. Bypass模式（性能优化）

```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: null
    bypass_mode: true  # 🔑 跳过old_log_prob计算
    use_policy_gradient: false

actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### 5. 纯策略梯度 + IS

```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: null
    bypass_mode: true
    use_policy_gradient: true  # 🔑 使用PG而非PPO

actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### 6. 仅监控（不应用校正）

```yaml
algorithm:
  rollout_correction:
    rollout_is: null
    rollout_rs: null
    bypass_mode: false

# 仍然计算指标，但不应用任何校正
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### 7. 命令行覆盖示例

```bash
python -m verl.trainer.main_ppo \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_rs=sequence \
    algorithm.rollout_correction.rollout_rs_threshold=2.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    ...
```

---

## 参考资料

### 论文和博客

1. **When Speed Kills Stability** (Liu, Li, Fu, Wang, Liu, Shen, 2025)

   - URL: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
   - 主题: RL训练崩溃的根本原因（训练-推理不匹配）

2. **Off-Policy RL理论基础** (Yao Feng)

   - URL: https://fengyao.notion.site/off-policy-rl
   - 主题: 重要性采样的理论推导

### 代码文件

**核心实现**:

- `verl/trainer/ppo/rollout_corr_helper.py` - IS/RS核心逻辑
- `verl/trainer/ppo/core_algos.py` - PPO与IS集成
- `verl/trainer/ppo/ray_trainer.py` - Bypass模式实现
- `verl/workers/actor/dp_actor.py` - 分布式worker逻辑

**配置**:

- `verl/trainer/config/algorithm.py` - `RolloutCorrectionConfig` 数据类
- `verl/trainer/config/algorithm/rollout_correction.yaml` - 默认配置
- `verl/trainer/config/ppo_trainer.yaml` - 主配置文件

**文档**:

- `docs/algo/rollout_corr.md` - 使用指南（本文参考）
- `docs/algo/rollout_corr_math.md` - 数学推导
- `docs/examples/config.rst` - 参数文档

**测试**:

- `tests/trainer/ppo/test_rollout_corr.py` - 单元测试
- `tests/trainer/ppo/test_rollout_corr_integration.py` - 集成测试

**示例**:

- `recipe/dapo/run_dapo_qwen2.5_32b_rollout_corr.sh` - DAPO示例
- `examples/rollout_correction/run_with_rollout_corr.sh` - 基础示例

### 相关概念

- **Importance Sampling (IS)**: 重要性采样
- **Rejection Sampling (RS)**: 拒绝采样
- **Truncated IS (TIS)**: 截断重要性采样
- **Effective Sample Size (ESS)**: 有效样本量
- **KL Divergence**: KL散度
- **Perplexity (PPL)**: 困惑度
- **χ² Divergence**: 卡方散度
- **Off-Policy RL**: 离策略强化学习

---

## 附录：常见问题

### Q1: IS和RS可以同时使用吗？

**A**: 可以！这是推荐的配置。

```python
config = RolloutCorrectionConfig.decoupled_seq_is_rs()
```

- IS权重: 连续调整梯度贡献
- RS mask: 剔除极端outlier
- 两者协同工作，互不冲突

### Q2: 为什么有安全边界（SAFETY_BOUND=20）？

**A**: 防止数值溢出/下溢。

- `exp(20) ≈ 4.85×10^8` - FP16/BF16的安全上界
- `exp(-20) ≈ 2×10^-9` - 安全下界
- 即使在极端情况下也不会崩溃

### Q3: Token-level和Sequence-level如何选择？

**A**: 根据策略差异大小选择。

| 场景 | 推荐 | 原因 |
|------|------|------|
| 仅精度差异 | Token-level | 低方差，稳定 |
| 模型更新1-2轮 | Sequence-level | 无偏，理论保证 |
| 模型更新多轮 | Sequence + RS | 严格过滤 |

### Q4: 什么时候需要Veto？

**A**: 高风险场景或发现过catastrophic forgetting。

```python
# 启用veto
config.rollout_token_veto_threshold = 1e-4
```

**效果**: 任何token < 0.0001倍概率 → 拒绝整个序列

### Q5: Bypass模式的优缺点？

**A**:

**优点**:

- 跳过 old_log_prob 计算，节省时间
- 减少一次forward pass

**缺点**:

- PPO裁剪对象变成 π_rollout（而非 π_old）
- 数学上不如Decoupled模式严格

**适用**: 策略差异很小时的性能优化

### Q6: 指标如何解读？

**A**: 关键指标含义：

```python
# IS权重统计
rollout_corr/rollout_is_mean  # 平均权重（理想值：1.0）
rollout_corr/rollout_is_max   # 最大权重
rollout_corr/rollout_is_ratio_fraction_high  # 超过阈值的比例

# RS拒绝率
rollout_corr/rollout_rs_masked_fraction  # Token拒绝率
rollout_corr/rollout_rs_seq_masked_fraction  # 序列拒绝率

# Off-policy诊断
rollout_corr/kl  # KL散度（越大偏移越严重）
rollout_corr/chi2_seq  # χ²散度（检测分布shift）

# Veto统计
rollout_corr/rollout_is_veto_fraction  # Veto的序列比例
```

**告警阈值**:

- `kl > 0.1`: 分布偏移较大
- `rollout_rs_seq_masked_fraction > 0.3`: 拒绝率过高（30%+）
- `rollout_is_veto_fraction > 0.1`: 灾难性样本过多

### Q7: 如何调试？

**A**: 步骤：

1. **先用disabled模式**，查看指标

   ```yaml
   rollout_is: null
   rollout_rs: null
   ```

2. **查看KL和PPL**，评估分布偏移
   - `kl < 0.01`: 差异很小
   - `0.01 < kl < 0.1`: 差异中等
   - `kl > 0.1`: 差异较大

3. **根据KL选择配置**
   - 小差异: Token IS
   - 中差异: Sequence IS + RS
   - 大差异: Geometric RS + Veto

4. **监控拒绝率**
   - `< 10%`: 正常
   - `10-30%`: 可接受
   - `> 30%`: 过于激进，调整阈值

---

**文档维护**: 如有问题或建议，请提交 Issue 到 verl GitHub 仓库。
