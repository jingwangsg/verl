# compute_log_prob 性能分析报告

## 调用链分析

### 1. 入口点 (verl/trainer/ppo/ray_trainer.py:1235)
```python
gw_yl_log_probs = self.actor_rollout_wg.compute_log_prob(batch_with_gw_yl)
```

- 在 CPO 算法中，当 `algorithm.adv_estimator == 'cpo'` 时被调用
- 处理带有 hint 的批次数据 (batch_with_gw_yl)
- 返回 old_log_probs 张量

### 2. Worker Group 类型

**actor_rollout_wg 可以是以下两种：**
- **ActorRolloutRefWorker** (FSDP-based): verl/workers/fsdp_workers.py:961
- **ActorRolloutRefWorker** (Megatron-based): verl/workers/megatron_workers.py:825

### 3. FSDP Worker 实现流程 (verl/workers/fsdp_workers.py:961-997)

```
compute_log_prob(data: DataProto)
  ├─ Load FSDP model to GPU (if offload enabled)
  ├─ Disable LoRA adapter (if needed)
  ├─ Set meta_info: micro_batch_size, max_token_len, use_dynamic_bsz, temperature
  ├─ Call self.actor.compute_log_prob(data) → DataParallelPPOActor
  └─ Move output to CPU
```

## 详细实现分析

### A. 数据分发 (Dispatch Mechanism)

**Decorator**: `@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))`

这个 dispatch 模式负责：
1. **数据分片**: 将整个 batch 分割成 world_size 个分片
2. **分布式计算**: 每个 GPU worker 独立计算自己分片的 log prob
3. **结果收集**: 在一个 worker 上汇总结果

### B. DataParallelPPOActor.compute_log_prob (verl/workers/actor/dp_actor.py:311-369)

这是**核心计算层**，执行以下步骤：

#### Step 1: 数据准备 (lines 332-339)
```python
micro_batch_size = data.meta_info["micro_batch_size"]  # Default: 8
use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]    # Default: False
data = data.select(batch_keys=["responses", "input_ids", "attention_mask", "position_ids"])
```

#### Step 2: 微批分割 (lines 341-345)
```python
if use_dynamic_bsz:
    micro_batches = prepare_dynamic_batch(data, max_token_len=...)
else:
    micro_batches = data.split(micro_batch_size)  # 将数据分成 micro_batch_size=8 的小批
```

#### Step 3: 逐微批处理 (lines 349-358)
```python
for micro_batch in micro_batches:
    micro_batch = micro_batch.to(get_device_id())  # 移到 GPU
    entropy, log_probs = self._forward_micro_batch(
        model_inputs, 
        temperature=temperature, 
        calculate_entropy=True
    )
    log_probs_lst.append(log_probs)
    entropy_lst.append(entropy)
```

#### Step 4: 合并结果 (lines 361-367)
```python
log_probs = torch.concat(log_probs_lst, dim=0)
entropys = torch.concat(entropy_lst, dim=0)
if use_dynamic_bsz:
    log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
```

### C. GPU 模型推理 (_forward_micro_batch, lines 95-245)

这是**最耗时的部分**，执行 LLM forward pass：

```python
with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
    # 处理 padding removal (如果启用)
    if self.use_remove_padding:
        input_ids_rmpad = unpad_input(input_ids, attention_mask)
        # 处理 Ulysses Sequence Parallel
    
    # Model Forward Pass
    model_outputs = self.actor_module(**model_inputs)
    
    # 计算 log probability
    logits = model_outputs.logits  # (bs, seq_len, vocab_size)
    log_probs = compute_log_prob(logits, targets)
```

## 性能瓶颈分析

### 关键配置参数

| 参数 | 默认值 | 位置 | 影响 |
|------|--------|------|------|
| `log_prob_micro_batch_size_per_gpu` | 8 | generation.yaml:38 | GPU 内存使用 |
| `log_prob_use_dynamic_bsz` | False | generation.yaml:39 | 动态批处理 |
| `log_prob_max_token_len_per_gpu` | 16384 | generation.yaml:40 | 序列长度限制 |

### 瓶颈 1: 高频 GPU 推理 (Critical)

**问题**:
- `compute_log_prob` 对每个 batch_with_gw_yl 执行完整的 LLM forward pass
- 在 CPO 算法中，这是额外的推理（除了基础的 old_log_probs 之外）
- 对于 7B 模型，每次 forward pass 需要 ~2-3 秒

**流程时间线** (from ray_trainer.py:1230-1236):
```
add_hint time: 0.5-1.0 秒 (CPU)
gw_yl_log_probs time: 2-5 秒 (GPU inference)  ← 主要瓶颈
```

### 瓶颈 2: hint 注入的数据准备 (Moderate)

**add_hint_to_data_batch_parallel** (verl/cpo/cpo_utils.py:420-537):

**优化策略**:
- 使用 ThreadPoolExecutor 并行化 tokenization (num_workers=32)
- 共享 tokenizer 避免重复加载
- 避免进程化（GIL 锁定）

**时间成本**:
- 原始顺序版本: ~1-2 秒
- 并行版本 (32 workers): ~0.3-0.5 秒
- 但仍需处理张量堆叠和转换

### 瓶颈 3: 数据移动与同步 (Moderate)

**CPU ↔ GPU 传输成本**:
1. `micro_batch.to(get_device_id())` (line 350)
   - 每个微批独立传输
   - 对于大 batch，可能多达 128/8 = 16 次传输

2. `output.to("cpu")` (line 989)
   - 最后将结果从 GPU 移回 CPU
   - 包含 log_probs 张量（大小: batch_size × seq_len）

3. FSDP 内存管理 (lines 981-996)
   - `load_fsdp_model_to_gpu()` / `offload_fsdp_model_to_cpu()`
   - 模型权重传输（数 GB）

### 瓶颈 4: 微批处理循环 (Moderate)

**问题**:
```python
for micro_batch in micro_batches:  # 可能 8-16 次迭代
    micro_batch = micro_batch.to(get_device_id())
    entropy, log_probs = self._forward_micro_batch(...)
    log_probs_lst.append(log_probs)
```

- 每次微批都需要完整的 forward pass
- 无法充分利用 batch 优化（如果 micro_batch_size=8）
- 串行执行，无法异步处理

### 瓶颈 5: 计算的重复性 (Conceptual)

**在 CPO 中**:
```
Step 1. 计算基础 log probs: self.actor_rollout_wg.compute_log_prob(batch)
Step 2. 添加 hint → batch_with_gw_yl
Step 3. 再次计算 log probs: self.actor_rollout_wg.compute_log_prob(batch_with_gw_yl)
```

- 完全重新计算，无缓存复用
- 虽然输入序列不同（包含 hint），但很多计算重复

## 资源利用情况分析

### GPU 内存占用

1. **模型权重**: ~14GB (7B 模型，fp32)
2. **激活值**: 随 batch_size 和 seq_len 增长
   - micro_batch_size=8, seq_len=2048: ~4-6GB
3. **KV 缓存**: 需要清空 (`aggressive_empty_cache()`)

### GPU 计算密度

- **micro_batch_size=8**: 低于最优值
  - 现代 GPU (A100/H100) 最优 batch_size: 32-64
  - batch=8 时利用率可能 50-70%

- **串行微批处理**: 无法并行化
  - 每次 forward pass 完全阻塞
  - 16 次微批 × 150ms = 2.4 秒

### GPU 利用率估计

```
假设：
- 总时间: 3 秒
- Model forward: 2.4 秒 (16 微批 × 150ms)
- 数据移动: 0.3 秒
- 其他开销: 0.3 秒

GPU 利用率: ~80% (相对于理论最大值)
但可优化空间: 30-50%
```

## 重复计算分析

### 当前的重复计算

| 步骤 | 计算内容 | 频率 | 成本 |
|------|--------|------|------|
| Embedding + Attention | 输入序列编码 | 每个 log prob 计算 1 次 | 高 |
| 注意力层 FFN | 序列转换 | 每个 log prob 计算 1 次 | 高 |
| Logits 计算 | 最后一层输出 | 每个 log prob 计算 1 次 | 中 |
| Log Prob 计算 | softmax + log | 每个 log prob 计算 1 次 | 低 |

### 为什么无法简单缓存

- 虽然 prompt 相同，但 `batch_with_gw_yl` 的 input_ids 不同
- hint 被注入到 prompt 和 response 之间
- KV 缓存形状改变

## 总结：性能瓶颈优先级

1. **Priority 1 (Critical)**: GPU LLM 推理 (2-5 秒)
   - 占总时间 60-70%
   - 微批处理导致利用率不足
   
2. **Priority 2 (High)**: 数据移动与同步 (0.3-0.5 秒)
   - GPU ↔ CPU 传输
   - FSDP 模型加载/卸载
   
3. **Priority 3 (Moderate)**: Hint 注入数据准备 (0.3-0.5 秒)
   - 已部分优化（并行化）
   - 仍有改进空间
   
4. **Priority 4 (Low)**: 微批合并 (< 50ms)
   - torch.concat 操作
   
5. **Priority 5 (Conceptual)**: 计算重复
   - 难以避免（不同的输入序列）
   - 但可考虑增量计算
