# VideoThinkTool 兼容性说明

## 关键差异说明

### 问题1: Action Parsing 格式

#### 原始实现 (think_with_video)
```python
action_start = "<action>"
action_end = "</action>"

# 模型生成格式:
<action>zoom in frame 100</action>
```

#### 现在的 ToolAgentLoop
使用 **OpenAI Function Calling** 格式:

**Hermes格式**:
```xml
<tool_call>{"name": "video_think", "arguments": {"action_string": "zoom in frame 100"}}</tool_call>
```

**GPT-OSS格式**:
```xml
<|start|>assistant<|channel|> to=functions.video_think <|constrain|>json<|message|>
{"action_string": "zoom in frame 100"}<|call|>
```

**解决方案**: VideoThinkTool内部使用正则表达式解析action_string，所以在function calling的arguments中仍然可以传递原始格式的action字符串。模型需要生成function calling格式，但action本身保持原有格式。

---

### 问题2: Tool Response的Role ⭐ (已解决)

#### 原始实现
Tool返回结果包在 **`user` role** 里:
```python
chat_template = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant"

# 生成的消息:
{"role": "user", "content": "frame 100: <image>"}
```

#### ToolAgentLoop 默认行为
Tool返回结果放在 **`tool` role** 里:
```python
{"role": "tool", "content": "frame 100: <image>"}
```

#### ✅ 解决方案: 可配置的 `tool_response_role`

修改后的 `ToolAgentLoop` 现在支持通过配置参数控制role：

```yaml
# 配置文件
actor_rollout_ref:
  rollout:
    multi_turn:
      tool_response_role: "user"  # 设置为 "user" 以兼容原始格式
                                   # 默认值: "tool"
```

**代码实现**:
```python
# tool_agent_loop.py:112
cls.tool_response_role = config.actor_rollout_ref.rollout.multi_turn.get("tool_response_role", "tool")

# tool_agent_loop.py:295, 298
message = {"role": self.tool_response_role, "content": content}
```

---

## 使用指南

### 选项1: 完全兼容原始格式 (推荐用于迁移)

如果您需要与原始 `think_with_video` 完全兼容:

```yaml
# config.yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      format: "hermes"  # 或 "gpt-oss"
      tool_response_role: "user"  # ⭐ 关键: 设置为 "user"
      tool_config_path: "path/to/video_think_tool.yaml"
```

**模型需要生成**:
```xml
<tool_call>{"name": "video_think", "arguments": {"action_string": "zoom in frame 100"}}</tool_call>
```

**Tool返回时的role**: `user`（与原始格式一致）

---

### 选项2: 使用标准Tool格式 (推荐用于新项目)

如果您使用新训练的模型:

```yaml
# config.yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      format: "hermes"
      tool_response_role: "tool"  # 默认值，可省略
      tool_config_path: "path/to/video_think_tool.yaml"
```

**模型需要生成**:
```xml
<tool_call>{"name": "video_think", "arguments": {"action_string": "zoom in frame 100"}}</tool_call>
```

**Tool返回时的role**: `tool`（标准OpenAI格式）

---

## 迁移建议

### 场景1: 使用原有模型
如果您使用的是用原始 `think_with_video` 格式训练的模型:

1. ✅ 保留原有的action格式 (`<action>...</action>`)
2. ✅ 设置 `tool_response_role: "user"`
3. ⚠️ 需要微调模型以支持function calling格式，或者自定义ToolParser支持`<action>`标签

### 场景2: 重新训练模型
如果您准备重新训练模型:

1. ✅ 使用标准 function calling 格式
2. ✅ 使用默认 `tool_response_role: "tool"`
3. ✅ Action字符串仍然保持原有格式（在arguments中传递）

---

## 配置示例对比

### 兼容模式配置
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      max_user_turns: 20
      max_assistant_turns: 20
      max_parallel_calls: 1
      format: "hermes"
      tool_response_role: "user"  # ⭐ 兼容原始格式
      tool_config_path: "./configs/video_think_tool.yaml"
```

### 标准模式配置
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      max_user_turns: 20
      max_assistant_turns: 20
      max_parallel_calls: 4  # 支持并发
      format: "hermes"
      tool_response_role: "tool"  # 标准OpenAI格式
      tool_config_path: "./configs/video_think_tool.yaml"
```

---

## 测试验证

### 验证Role配置
```python
import asyncio
from verl.experimental.agent_loop import ToolAgentLoop

# 初始化后检查
print(f"Tool response role: {ToolAgentLoop.tool_response_role}")
# 应输出: "user" 或 "tool"
```

### 完整测试流程
1. 加载配置文件
2. 初始化 ToolAgentLoop
3. 运行一个简单的video action
4. 检查生成的messages中tool response的role
5. 确认与配置一致

---

## 常见问题

### Q: 如果不设置 `tool_response_role` 会怎样？
**A**: 默认使用 `"tool"`，这是标准OpenAI格式。如果您的模型是用原始格式训练的，可能会影响性能。

### Q: 可以在运行时动态修改role吗？
**A**: 不可以。`tool_response_role` 是类级别配置，在 `init_class()` 时设置，之后不能修改。

### Q: Action parsing是在哪里处理的？
**A**: 在 `VideoThinkTool.execute()` 方法内部使用正则表达式处理，与agent loop无关。

### Q: 为什么要保留原始的action格式？
**A**: 因为原始格式更简洁，且已经有大量数据使用这种格式训练。Function calling只是外层包装。

---

## 总结

✅ **已解决**: Tool response role 现在可以通过配置参数 `tool_response_role` 控制
✅ **向后兼容**: 设置 `tool_response_role: "user"` 即可兼容原始格式
✅ **灵活性**: 支持标准OpenAI格式 (`"tool"`) 和原始格式 (`"user"`)
⚠️ **需注意**: Action parsing格式改变需要模型适配或自定义ToolParser

---

## 相关文件

- **ToolAgentLoop修改**: `verl/experimental/agent_loop/tool_agent_loop.py`
  - Line 112: 添加 `tool_response_role` 配置
  - Line 295, 298: 使用可配置的role
- **配置示例**: `verl/tools/video_think_tool_config_example.yaml`
- **工具实现**: `verl/tools/video_think_tool.py`
