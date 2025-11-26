# FrameThinker 从 Multi-Turn 迁移到 Agent 模式 - 完整技术方案

## 一、迁移目标与策略

### 1.1 迁移目标
- **主要目的**: 性能优化 - 利用 Agent 模式的全异步架构提升 GPU 利用率和训练吞吐量
- **改动范围**: 可以重新生成数据集，无需保持向后兼容

### 1.2 迁移策略
FrameThinker 的迁移是**轻量级改动**，而非架构重构：
- **数据层**: 在数据预处理中添加 `agent_name: "tool_agent"` 字段
- **配置层**: 启用 `rollout.mode=async` 和配置 `rollout.agent` 参数
- **代码层**: 无需修改核心代码，工具和 reward 函数完全兼容

### 1.3 Multi-Turn vs Agent 核心差异

| 维度 | Multi-Turn 模式 | Agent 模式 |
|------|----------------|-----------|
| 执行方式 | SGLang/vLLM 批处理 | AsyncLLMServerManager 异步处理 |
| 状态管理 | 隐式（引擎内部） | 显式状态机（PENDING → GENERATING → PROCESSING_TOOLS） |
| 数据标识 | 无需 `agent_name` | 必须有 `agent_name: "tool_agent"` |
| 配置激活 | `multi_turn.enable=True` | `rollout.mode=async` + `agent` 配置 |
| 工具调用 | 框架控制 | Agent 循环控制（更灵活） |
| GPU 利用 | 工具调用时可能空闲 | 异步执行减少空闲 |
| 内存管理 | 基础 | 支持 free_cache_engine（释放 KV cache） |

## 二、数据集准备

### 2.1 创建 Agent 模式数据预处理脚本

**文件**: `recipe/framethinker/data_preprocess/convert_to_rl_parquet_agent.py`

**改动内容**:
```python
# 基于现有的 convert_to_rl_parquet.py，添加一个字段：
data = {
    "data_source": "video_holmes",
    "agent_name": "tool_agent",  # 🔑 关键添加
    "prompt": prompt,
    "answer": answer,
    "extra_info": {
        "index": idx,
        "question": question,
        "total_frames": total_frames,
        "tools_kwargs": {
            "video_think": {
                "create_kwargs": {
                    "video_path": video_path,
                    "fps": fps,
                    "total_frames": total_frames,
                    "width": width,
                    "height": height
                }
            }
        }
    }
}
```

### 2.2 重新生成训练和验证数据

**命令**:
```bash
cd recipe/framethinker/data_preprocess

# Video-Holmes 数据集
python convert_to_rl_parquet_agent.py \
    --input_file /path/to/Video-Holmes/train.json \
    --output_file /path/to/output/train_agent.parquet \
    --media_dir /mnt/amlfs-02/shared/datasets/s3/video_reason/

python convert_to_rl_parquet_agent.py \
    --input_file /path/to/Video-Holmes/test.json \
    --output_file /path/to/output/test_agent.parquet \
    --media_dir /mnt/amlfs-02/shared/datasets/s3/video_reason/
```

**验证数据格式**:
```python
import pandas as pd
df = pd.read_parquet("train_agent.parquet")
assert "agent_name" in df.columns[0]  # 检查 agent_name 字段存在
assert df.iloc[0]["agent_name"] == "tool_agent"
```

## 三、配置文件修改

### 3.1 创建新的训练配置文件

**文件**: `recipe/framethinker/config/framethinker_grpo_agent.yaml`

**改动内容**:
```yaml
# 基于 framethinker_grpo.yaml，添加/修改以下部分：

actor_rollout_ref:
  rollout:
    name: sglang  # 或 vllm，根据原配置
    mode: async  # 🔑 启用异步模式

    # Agent 配置（新增）
    agent:
      num_workers: 4  # 推荐 2-4 个 worker
      default_agent_loop: "tool_agent"
      agent_loop_config_path: null  # 暂时不需要自定义 agent

    # Multi-turn 配置（保留，agent 会复用）
    multi_turn:
      enable: True  # 保留以提供工具基础设施
      max_assistant_turns: 5
      max_user_turns: 5
      max_parallel_calls: 1  # 保守配置，避免 OOM
      format: think_with_video
      max_tool_response_length: 2048
      tool_response_truncate_side: middle
      tool_response_role: user
      tool_config_path: ${PROJECT_DIR}/recipe/framethinker/config/video_think_tool_config.yaml

    # 性能优化配置
    free_cache_engine: True  # 🔑 重要：释放 KV cache 内存
    calculate_log_probs: True

    # 其他 rollout 参数保持不变
    n: 8
    tensor_model_parallel_size: 2
    max_num_batched_tokens: 32768
    gpu_memory_utilization: 0.5
    dtype: float16
    limit_images: 128
```

### 3.2 创建新的训练脚本

**文件**: `recipe/framethinker/train_frame_thinker_agent.sh`

**改动内容**:
```bash
#!/bin/bash
# FrameThinker Agent Mode Training Script

wandb login <wandb_api_key>
export WANDB_ENTITY="<entity_name>"

set -x
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/framethinker/config"
TOOL_CONFIG_PATH="$CONFIG_PATH/video_think_tool_config.yaml"

# 数据路径 - 使用新生成的 agent 数据
TRAIN_FILES=${TRAIN_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/train_agent.parquet}
VAL_FILES=${VAL_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/test_agent.parquet}
MEDIA_DIA=${MEDIA_DIA:-$PROJECT_DIR/data/video_reason/}

MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/model_weights/ft_coldstart/qwen2_5vl_7b_full_framethinker_sft}
PROJECT_NAME=${PROJECT_NAME:-video_holmes_verl_agent}
EXP_NAME=${EXP_NAME:-framethinker_agent_mode}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/video_reason/}

DTYPE=float16
MODEL_DTYPE=fp32

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='framethinker_grpo_agent' \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.media_dir=${MEDIA_DIA} \
    data.return_raw_chat=True \
    data.message_template=framethinker_add_zoomin \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_loop="tool_agent" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.nframes=8 \
    custom_reward_function.reward_kwargs.lambda_gfn=0.2 \
    $@
```

## 四、性能优化配置

### 4.1 Agent Worker 数量调优

**推荐配置**:
```yaml
# 起始配置（保守）
actor_rollout_ref.rollout.agent.num_workers: 2

# 性能配置（标准）
actor_rollout_ref.rollout.agent.num_workers: 4

# 高性能配置（需要更多 GPU 内存）
actor_rollout_ref.rollout.agent.num_workers: 6-8
```

**调优策略**:
1. 从 `num_workers=2` 开始测试
2. 监控 GPU 内存使用和吞吐量
3. 逐步增加直到达到 OOM 或性能饱和

### 4.2 工具并发调优

```yaml
# 保守配置（避免 OOM）
actor_rollout_ref.rollout.multi_turn.max_parallel_calls: 1

# 平衡配置（推荐）
actor_rollout_ref.rollout.multi_turn.max_parallel_calls: 2

# 高并发（仅在内存充足时）
actor_rollout_ref.rollout.multi_turn.max_parallel_calls: 4
```

### 4.3 内存优化配置

```yaml
# 启用 KV cache 释放（强烈推荐）
actor_rollout_ref.rollout.free_cache_engine: True

# 降低 GPU 内存利用率（如果 OOM）
actor_rollout_ref.rollout.gpu_memory_utilization: 0.4  # 从 0.5 降低

# 减少批处理 token 数（如果 OOM）
actor_rollout_ref.rollout.max_num_batched_tokens: 16384  # 从 32768 降低

# 限制图像数量（如果 OOM）
actor_rollout_ref.rollout.limit_images: 64  # 从 128 降低
```

## 五、测试和验证

### 5.1 小规模测试

**创建调试脚本**: `recipe/framethinker/train_frame_thinker_agent_debug.sh`

```bash
# 基于 train_frame_thinker_agent.sh，修改以下参数：
data.train_batch_size=8  # 减小批次
trainer.total_epochs=1   # 仅 1 个 epoch
trainer.save_freq=999    # 不保存 checkpoint
trainer.test_freq=5      # 频繁验证
```

**运行测试**:
```bash
bash recipe/framethinker/train_frame_thinker_agent_debug.sh
```

### 5.2 验证检查清单

- [ ] **数据加载成功**: 检查日志 `Loading dataset from...`
- [ ] **Agent 初始化成功**: 检查日志 `Initialized tools: {'video_think': ...}`
- [ ] **Agent Loop Workers 启动**: 检查日志 `AgentLoopManager initialized with X workers`
- [ ] **Rollout 成功**: 检查日志 `generate_sequences completed`
- [ ] **工具调用成功**: 检查日志中的工具调用统计（tool_call_counts）
- [ ] **Reward 计算正确**: 检查日志中的 reward score 分布
- [ ] **训练步骤正常**: 检查 PPO loss 下降
- [ ] **无 OOM 错误**: GPU 内存使用稳定
- [ ] **性能指标**: 记录 `agent_loop/generate_sequences/mean` 延迟

### 5.3 性能对比测试

**对比指标**:
```bash
# Multi-Turn 模式基线
bash recipe/framethinker/train_frame_thinker_vhonly.sh

# Agent 模式测试
bash recipe/framethinker/train_frame_thinker_agent.sh

# 对比以下指标：
# 1. Rollout 时长（秒/batch）
# 2. 训练时长（秒/epoch）
# 3. GPU 利用率（%）
# 4. GPU 内存峰值（GB）
# 5. 样本吞吐量（samples/second）
```

## 六、生产部署配置

### 6.1 完整训练脚本

**最终配置**:
```bash
# 使用经过调优的参数
actor_rollout_ref.rollout.agent.num_workers=4
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2
actor_rollout_ref.rollout.free_cache_engine=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.5

# 完整训练参数
trainer.total_epochs=10
trainer.save_freq=20
trainer.test_freq=20
data.train_batch_size=32
```

### 6.2 监控和日志

**启用详细监控**:
```bash
# 启用 MLflow 追踪
actor_rollout_ref.rollout.trace.backend=mlflow
actor_rollout_ref.rollout.trace.token2text=True

# 启用详细日志
export VERL_LOGGING_LEVEL=INFO

# 监控命令
mlflow ui -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:///./mlruns.db
```

**关键监控指标**:
- `agent_loop/generate_sequences/{min,max,mean}`: 推理延迟
- `agent_loop/tool_calls/{min,max,mean}`: 工具执行时间
- `agent_loop/slowest/*`: 最慢样本分析
- `tool_call_counts`: 工具调用统计
- GPU 内存使用率

## 七、故障排查和回滚方案

### 7.1 快速回滚

如果遇到严重问题，可以立即回滚到 Multi-Turn 模式：
```bash
# 使用原始训练脚本和数据
bash recipe/framethinker/train_frame_thinker_vhonly.sh
```

### 7.2 常见问题排查

**问题 1: OOM 错误**
```bash
# 解决方案：
actor_rollout_ref.rollout.agent.num_workers=2  # 减少 worker
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 降低内存
actor_rollout_ref.rollout.max_num_batched_tokens=16384  # 减少批处理
```

**问题 2: Agent Loop 初始化失败**
```bash
# 检查：
# 1. agent_name 字段是否存在于数据中
# 2. tool_config_path 是否正确
# 3. 查看详细错误日志
export VERL_LOGGING_LEVEL=DEBUG
```

**问题 3: 性能没有提升**
```bash
# 可能原因：
# 1. num_workers 太少：增加到 4-8
# 2. free_cache_engine 未启用
# 3. 工具并发度太低：增加 max_parallel_calls
```

## 八、关键文件清单

### 需要创建的文件
1. `recipe/framethinker/data_preprocess/convert_to_rl_parquet_agent.py` - Agent 数据预处理
2. `recipe/framethinker/config/framethinker_grpo_agent.yaml` - Agent 训练配置
3. `recipe/framethinker/train_frame_thinker_agent.sh` - Agent 训练脚本
4. `recipe/framethinker/train_frame_thinker_agent_debug.sh` - Agent 调试脚本

### 需要修改的文件
无需修改现有文件，保持完全向后兼容

### 可以复用的文件（无需改动）
1. `recipe/framethinker/config/video_think_tool_config.yaml` - 工具配置
2. `verl/tools/video_think_tool.py` - 工具实现
3. `verl/utils/reward_score/think_with_video_reward.py` - Reward 函数
4. `verl/utils/dataset/templates/framethinker_*.py` - 消息模板

## 九、预期性能提升

### 9.1 理论收益
- **GPU 利用率**: +10-30%（通过异步执行减少 GPU 空闲）
- **训练吞吐量**: +15-40%（取决于工具调用密度）
- **内存效率**: +10-20%（通过 free_cache_engine）

### 9.2 实际收益评估
需要通过对比测试确认，预期在工具调用频繁的任务上收益最大。

## 十、风险评估与缓解

### 10.1 风险等级：低

**原因**:
- 数据格式改动小（仅添加一个字段）
- 配置改动清晰（新增配置块）
- 可以随时回滚到 Multi-Turn 模式

### 10.2 潜在风险

1. **异步竞态条件**: Agent 模式的异步特性可能引入新的竞态条件
   - **缓解**: 通过小规模测试发现和修复

2. **内存管理复杂度**: 需要细致调优
   - **缓解**: 通过 num_workers 和 gpu_memory_utilization 动态调整

3. **性能提升不明显**: 如果工具调用不频繁
   - **缓解**: 通过对比测试评估，如果收益不明显可以选择不迁移

### 10.3 缓解措施

- 保留 Multi-Turn 模式作为后备方案
- 分阶段测试和部署（调试 → 小规模 → 生产）
- 详细监控和日志记录
- 充分的验证检查清单

## 十一、参考资源

### 11.1 官方文档
- Agent Loop 完整文档: `/Users/bytedance/Codes/verl/docs/advance/agent_loop.rst`
- Agentic RL 快速开始: `/Users/bytedance/Codes/verl/docs/start/agentic_rl.rst`
- Multi-turn 文档: `/Users/bytedance/Codes/verl/docs/sglang_multiturn/multiturn.rst`

### 11.2 代码示例
- GSM8K Agent 训练脚本: `/Users/bytedance/Codes/verl/examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh`
- GSM8K Agent 数据预处理: `/Users/bytedance/Codes/verl/examples/data_preprocess/gsm8k_tool_agent_loop.py`
- GSM8K Multi-turn 数据预处理: `/Users/bytedance/Codes/verl/examples/data_preprocess/gsm8k_multiturn_w_tool.py`

### 11.3 核心实现
- ToolAgentLoop 实现: `/Users/bytedance/Codes/verl/verl/experimental/agent_loop/tool_agent_loop.py`
- AgentLoopManager: `/Users/bytedance/Codes/verl/verl/experimental/agent_loop/agent_loop.py`
- VideoThinkTool: `/Users/bytedance/Codes/verl/verl/tools/video_think_tool.py`

---

## 附录：完整代码模板

### A.1 数据预处理脚本模板

基于现有的 `convert_to_rl_parquet.py`，创建 `convert_to_rl_parquet_agent.py` 时，只需在生成数据字典时添加一行：

```python
# 原有代码保持不变，只在数据字典中添加 agent_name
data = {
    "data_source": "video_holmes",
    "agent_name": "tool_agent",  # 🔑 唯一需要添加的字段
    "prompt": prompt,
    "answer": answer,
    "extra_info": {
        # ... 其他字段保持不变
    }
}
```

### A.2 训练配置完整示例

```yaml
# framethinker_grpo_agent.yaml
defaults:
  - ppo_trainer  # 或 ppo_megatron_trainer

# 数据配置
data:
  train_files: [/path/to/train_agent.parquet]
  val_files: [/path/to/test_agent.parquet]
  max_prompt_length: 8192
  max_response_length: 8192
  train_batch_size: 32
  return_raw_chat: True
  message_template: framethinker_add_zoomin
  media_dir: /mnt/amlfs-02/shared/datasets/s3/video_reason/
  media_reading_kwargs:
    enable: True
    num_frames: 8
    size: 360
    sampling_mode: uniform

# 算法配置
algorithm:
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.0
  use_kl_in_reward: False

# Actor/Rollout/Ref 配置
actor_rollout_ref:
  model:
    path: /path/to/qwen2_5vl_7b_full_framethinker_sft
    use_remove_padding: True

  actor:
    optim:
      lr: 1e-6
      betas: [0.9, 0.95]
      weight_decay: 0.0
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 2
    use_kl_loss: False
    entropy_coeff: 0.0
    fsdp_config:
      param_offload: True
      optimizer_offload: True
      dtype: float16
      model_dtype: fp32

  rollout:
    name: vllm  # 或 sglang
    mode: async  # 🔑 启用异步模式
    n: 8
    tensor_model_parallel_size: 2
    max_num_batched_tokens: 32768
    gpu_memory_utilization: 0.5
    dtype: float16
    calculate_log_probs: True
    free_cache_engine: True  # 🔑 启用 KV cache 释放
    limit_images: 128

    # Agent 配置（新增）
    agent:
      num_workers: 4  # 🔑 Agent loop workers
      default_agent_loop: "tool_agent"  # 🔑 默认 agent

    # Multi-turn 配置（保留以提供工具基础设施）
    multi_turn:
      enable: True
      max_assistant_turns: 5
      max_user_turns: 5
      max_parallel_calls: 1
      format: think_with_video
      max_tool_response_length: 2048
      tool_response_truncate_side: middle
      tool_response_role: user
      tool_config_path: recipe/framethinker/config/video_think_tool_config.yaml

  ref:
    log_prob_micro_batch_size_per_gpu: 2
    fsdp_config:
      param_offload: True
      dtype: float16
      model_dtype: fp32

# Trainer 配置
trainer:
  critic_warmup: 0
  logger: '["console","wandb"]'
  n_gpus_per_node: 8
  nnodes: 1
  save_freq: 20
  test_freq: 20
  val_before_train: False
  total_epochs: 10
  project_name: video_holmes_verl_agent
  experiment_name: framethinker_agent_mode

# Reward 函数配置
custom_reward_function:
  path: verl/utils/reward_score/think_with_video_reward.py
  name: compute_score
  reward_kwargs:
    nframes: 8
    lambda_gfn: 0.2
    lambda_cf: 0.0
    alpha_zoom: 0.0
```

### A.3 完整训练脚本模板

```bash
#!/bin/bash
# train_frame_thinker_agent.sh

wandb login <your_api_key>
export WANDB_ENTITY="<your_entity>"

set -x
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/framethinker/config"
TOOL_CONFIG_PATH="$CONFIG_PATH/video_think_tool_config.yaml"

# 数据路径
TRAIN_FILES=${TRAIN_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/train_agent.parquet}
VAL_FILES=${VAL_FILES:-$PROJECT_DIR/data/video_reason/Video-Holmes/test_agent.parquet}
MEDIA_DIA=${MEDIA_DIA:-$PROJECT_DIR/data/video_reason/}

# 模型路径
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/model_weights/ft_coldstart/qwen2_5vl_7b_full_framethinker_sft}
PROJECT_NAME=${PROJECT_NAME:-video_holmes_verl_agent}
EXP_NAME=${EXP_NAME:-framethinker_agent_mode}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/video_reason/}

DTYPE=float16
MODEL_DTYPE=fp32

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='framethinker_grpo_agent' \
    "data.train_files=[${TRAIN_FILES}]" \
    "data.val_files=[${VAL_FILES}]" \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.media_dir=${MEDIA_DIA} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=8 \
    data.message_template=framethinker_add_zoomin \
    data.media_reading_kwargs.enable=True \
    data.media_reading_kwargs.num_frames=8 \
    data.media_reading_kwargs.size=360 \
    data.media_reading_kwargs.sampling_mode=uniform \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_loop="tool_agent" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.limit_images=128 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.format=think_with_video \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.tool_response_role=user \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME} \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.95]" \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    custom_reward_function.path=verl/utils/reward_score/think_with_video_reward.py \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.nframes=8 \
    custom_reward_function.reward_kwargs.lambda_gfn=0.2 \
    custom_reward_function.reward_kwargs.lambda_cf=0.0 \
    custom_reward_function.reward_kwargs.alpha_zoom=0.0 \
    $@
```

---

完成！这份迁移计划涵盖了从 Multi-Turn 到 Agent 模式的完整技术方案。
