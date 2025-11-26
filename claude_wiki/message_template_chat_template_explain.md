# Message Template 与 Chat Template 的工作原理

**作者**: Claude Code
**创建时间**: 2025-11-26
**适用版本**: verl 0.6.1+

---

## 目录

1. [核心问题](#核心问题)
2. [完整数据处理流程](#完整数据处理流程)
3. [Message Template 详解](#message-template-详解)
4. [Chat Template 详解](#chat-template-详解)
5. [两者的关系与区别](#两者的关系与区别)
6. [FrameThinker 案例分析](#framethinker-案例分析)
7. [自定义 Template 指南](#自定义-template-指南)
8. [常见问题](#常见问题)

---

## 核心问题

### 问题 1: Custom Chat Template 没有 system role，system role 是从哪里来的？

**答案**: System role **不是**从 chat template 来的，而是由 **message template** 动态添加的。

### 问题 2: Message Template 和 Chat Template 有什么区别？

| 维度 | Message Template | Chat Template |
|------|------------------|---------------|
| **作用** | 决定**消息内容和结构** | 决定**消息格式** |
| **输入** | 原始数据（user message only） | 完整 messages（含 system） |
| **输出** | 完整 messages list | 格式化的文本 prompt |
| **主要任务** | 添加 system prompt、组织多模态占位符 | 渲染成特定格式（ChatML等） |
| **示例** | `framethinker_add_zoomin` | Qwen2.5-VL 的 Jinja2 模板 |
| **配置** | `data.message_template` | `actor_rollout_ref.model.custom_chat_template` |

---

## 完整数据处理流程

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 原始数据集 (parquet)                                         │
│                                                              │
│ {                                                            │
│   "prompt": [{"role": "user", "content": "问题..."}],        │
│   "extra_info": {"total_frames": 9000, ...},                │
│   "multi_modal_data": {"image": [...]},                     │
│   ...                                                        │
│ }                                                            │
│                                                              │
│ 注意：原始数据中 **没有 system role**                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: RLDataset._build_messages()                         │
│                                                              │
│ 文件: verl/utils/dataset/rl_dataset.py:273-278              │
│                                                              │
│ def _build_messages(self, example: dict):                   │
│     messages = example.pop(self.prompt_key)                 │
│     # messages = [{"role": "user", "content": "问题..."}]    │
│                                                              │
│     # 获取 message template                                 │
│     template_name = self.config.message_template            │
│     # template_name = "framethinker_add_zoomin"             │
│                                                              │
│     apply_message_template = get_message_template(...)      │
│     messages = apply_message_template(messages, **example)  │
│     # 🔑 这里添加 system role                                │
│     ...                                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Message Template 处理                                │
│                                                              │
│ 文件: verl/utils/dataset/templates/framethinker_add_zoomin.py│
│                                                              │
│ def apply_message_template(messages, **kwargs):             │
│     # 断言：原始数据不应有 system                            │
│     assert messages[0]["role"] != "system"                  │
│                                                              │
│     total_frames = kwargs["extra_info"]["total_frames"]     │
│     num_frames = len(kwargs["multi_modal_data"]["image"])   │
│                                                              │
│     # 生成图像占位符                                         │
│     image_placeholders = get_image_placeholders(...)        │
│     # "frame 0:<image>\nframe 1125:<image>\n..."            │
│                                                              │
│     question = messages[0]["content"]                       │
│                                                              │
│     # 🔑 构造完整 messages，添加 system role                 │
│     messages = [                                            │
│         {                                                    │
│             "role": "system",                               │
│             "content": get_system_prompt(total_frames)      │
│         },                                                   │
│         {                                                    │
│             "role": "user",                                 │
│             "content": question + "\n" + image_placeholders │
│         }                                                    │
│     ]                                                        │
│     return messages                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 多模态占位符处理                                      │
│                                                              │
│ 文件: verl/utils/dataset/rl_dataset.py:288-310              │
│                                                              │
│ 将 <image> 转换为结构化格式:                                  │
│   {"type": "image"} 或 {"type": "video"}                    │
│                                                              │
│ 最终 messages:                                               │
│ [                                                            │
│   {                                                          │
│     "role": "system",                                       │
│     "content": "You are an expert AI assistant..."          │
│   },                                                         │
│   {                                                          │
│     "role": "user",                                         │
│     "content": [                                            │
│       {"type": "text", "text": "What happens at 1:30?\n"},  │
│       {"type": "text", "text": "frame 0:"},                 │
│       {"type": "image"},                                    │
│       {"type": "text", "text": "\nframe 1125:"},            │
│       {"type": "image"},                                    │
│       ...                                                    │
│     ]                                                        │
│   }                                                          │
│ ]                                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Chat Template 应用                                   │
│                                                              │
│ 文件: verl/utils/dataset/rl_dataset.py:199-200              │
│                                                              │
│ raw_prompt = self.processor.apply_chat_template(            │
│     messages,                                               │
│     add_generation_prompt=True,                             │
│     tokenize=False,                                         │
│     **apply_kwargs                                          │
│ )                                                            │
│                                                              │
│ 使用 custom_chat_template (Jinja2 模板) 渲染                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 最终生成的 Prompt                                     │
│                                                              │
│ <|im_start|>system                                          │
│ You are an expert AI assistant that answers questions...    │
│ Possible actions are:                                       │
│ 1. `choose frames between START_FRAME and END_FRAME`...     │
│ 2. `get frame number at time MM:SS`...                      │
│ 3. `zoom in frame FRAME_INDEX`...                           │
│ 4. `output answer: OPTION`...<|im_end|>                     │
│ <|im_start|>user                                            │
│ What happens at 1:30?                                       │
│ frame 0:<vision_start><image_pad><vision_end>               │
│ frame 1125:<vision_start><image_pad><vision_end>            │
│ frame 2250:<vision_start><image_pad><vision_end>            │
│ ...<|im_end|>                                               │
│ <|im_start|>assistant                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Message Template 详解

### 定义

Message Template 负责**构造消息的内容和结构**，包括：
- 添加 system prompt
- 组织 user message 内容
- 插入多模态占位符
- 设置 message 的角色（role）

### 注册机制

**文件**: `verl/utils/dataset/templates/__init__.py`

```python
from .default import apply_message_template as apply_message_template_default
from .framethinker_add_zoomin import (
    apply_message_template as apply_message_template_framethinker_add_zoomin,
)

MESSAGE_TEMPLATES = {
    "default": apply_message_template_default,
    "framethinker_add_zoomin": apply_message_template_framethinker_add_zoomin,
    "framethinker_default": apply_message_template_framethinker_default,
    "videor1": apply_message_template_videor1,
}

def get_message_template(name: str):
    assert name in MESSAGE_TEMPLATES, f"Message template {name} not found"
    return MESSAGE_TEMPLATES[name]
```

### 内置 Templates

#### 1. **default** (默认模板)

**文件**: `verl/utils/dataset/templates/default.py`

```python
def apply_message_template(messages, **kwargs):
    return messages  # 不做任何修改
```

**用途**: 数据集已经包含完整 messages（含 system）时使用。

#### 2. **framethinker_add_zoomin** (FrameThinker + Zoom 功能)

**文件**: `verl/utils/dataset/templates/framethinker_add_zoomin.py`

```python
def apply_message_template(messages, **kwargs):
    # 验证原始数据格式
    assert messages[0]["role"] != "system", "No system in original data"
    assert len(messages) == 1, "Only one user message allowed"

    # 获取参数
    total_frames = kwargs["extra_info"]["total_frames"]
    num_frames = len(kwargs["multi_modal_data"]["image"])
    is_video = num_frames > 1

    # 生成图像占位符
    if is_video:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        image_placeholders = "\n".join([f"frame {idx}:<image>" for idx in frame_indices])
    else:
        image_placeholders = "<image>"

    question = messages[0]["content"]

    # 构造完整 messages
    messages = [
        {"role": "system", "content": get_system_prompt(num_frames=total_frames, is_video=is_video)},
        {"role": "user", "content": question + "\n" + image_placeholders},
    ]

    return messages
```

**System Prompt** (视频版本):
```
You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. You MUST choose frames from 0 to {total_frames - 1}.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time.
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.
```

**System Prompt** (图像版本):
```
You are an expert AI assistant that answers questions about an image by analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `zoom in`: Zoom in on the image and return the high-resolution image.
2. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.
```

**特点**:
- ✅ 支持 zoom in 功能
- ✅ 区分视频和图像任务
- ✅ 动态生成 frame indices
- ✅ System prompt 包含 `total_frames` 信息

#### 3. **framethinker_default** (FrameThinker 基础版)

**文件**: `verl/utils/dataset/templates/framethinker_default.py`

与 `framethinker_add_zoomin` 类似，但：
- ❌ 不支持 `zoom in` 动作
- ✅ 仅支持 `choose frames` 和 `get frame number`

**System Prompt**:
```
You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. The number of frames is fixed, currently 8.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time.
3. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.
```

### 配置方式

在训练脚本中指定：

```bash
python -m verl.trainer.main_ppo \
    data.message_template=framethinker_add_zoomin \
    ...
```

或在 YAML 配置中：

```yaml
data:
  message_template: framethinker_add_zoomin
```

---

## Chat Template 详解

### 定义

Chat Template 负责**将消息列表渲染成特定格式的文本**，包括：
- 添加特殊 token（如 `<|im_start|>`, `<|im_end|>`）
- 处理多模态占位符（`<image>` → `<vision_start><image_pad><vision_end>`）
- 格式化不同角色的消息
- 添加 generation prompt

### 配置位置

**文件**: `verl/workers/config/model.py:66`

```python
@dataclass
class HFModelConfig(BaseConfig):
    # custom chat template for the model
    custom_chat_template: Optional[str] = None
```

### 设置流程

#### **FSDP Workers**

**文件**: `verl/workers/fsdp_workers.py:306-310`

```python
if self.config.model.get("custom_chat_template", None) is not None:
    if self.processor is not None:
        self.processor.chat_template = self.config.model.custom_chat_template
    else:
        self.tokenizer.chat_template = self.config.model.custom_chat_template
```

#### **Agent Loop**

**文件**: `verl/experimental/agent_loop/agent_loop.py:290-293`

```python
if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
    if self.processor is not None:
        self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
    self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
```

### Qwen2.5-VL 的 Chat Template

**文件**: `recipe/framethinker/config/think_with_video.yaml:18-19`

```yaml
actor_rollout_ref:
  model:
    custom_chat_template: "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}{% if message['content'] is string %}{{ message['content'] }}<|im_end|>{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}"
```

#### 格式化后（便于阅读）:

```jinja2
{% set image_count = namespace(value=0) %}
{% set video_count = namespace(value=0) %}
{% for message in messages %}
  <|im_start|>{{ message['role'] }}
  {% if message['content'] is string %}
    {{ message['content'] }}<|im_end|>
  {% else %}
    {% for content in message['content'] %}
      {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
        {% set image_count.value = image_count.value + 1 %}
        {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
        <|vision_start|><|image_pad|><|vision_end|>
      {% elif content['type'] == 'video' or 'video' in content %}
        {% set video_count.value = video_count.value + 1 %}
        {% if add_vision_id %}Video {{ video_count.value }}: {% endif %}
        <|vision_start|><|video_pad|><|vision_end|>
      {% elif 'text' in content %}
        {{ content['text'] }}
      {% endif %}
    {% endfor %}
    <|im_end|>
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant{% endif %}
```

#### 模板解析

**变量初始化**:
```jinja2
{% set image_count = namespace(value=0) %}
{% set video_count = namespace(value=0) %}
```
- 用于计数图像和视频数量

**遍历消息**:
```jinja2
{% for message in messages %}
  <|im_start|>{{ message['role'] }}
```
- 为每个消息添加 ChatML 开始标记和角色

**处理消息内容**:

1. **纯文本内容**:
```jinja2
{% if message['content'] is string %}
  {{ message['content'] }}<|im_end|>
```

2. **结构化内容**（多模态）:
```jinja2
{% for content in message['content'] %}
  {% if content['type'] == 'image' ... %}
    <|vision_start|><|image_pad|><|vision_end|>
  {% elif content['type'] == 'video' ... %}
    <|vision_start|><|video_pad|><|vision_end|>
  {% elif 'text' in content %}
    {{ content['text'] }}
  {% endif %}
{% endfor %}
<|im_end|>
```

**添加生成 prompt**:
```jinja2
{% if add_generation_prompt %}
  <|im_start|>assistant
{% endif %}
```

### 应用示例

**输入 messages**:
```python
[
    {
        "role": "system",
        "content": "You are an expert AI assistant..."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens at 1:30?\n"},
            {"type": "text", "text": "frame 0:"},
            {"type": "image"},
            {"type": "text", "text": "\nframe 1125:"},
            {"type": "image"},
        ]
    }
]
```

**输出 prompt**:
```
<|im_start|>system
You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. You MUST choose frames from 0 to 8999.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time.
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.<|im_end|>
<|im_start|>user
What happens at 1:30?
frame 0:<vision_start><image_pad><vision_end>
frame 1125:<vision_start><image_pad><vision_end>
frame 2250:<vision_start><image_pad><vision_end>
...<|im_end|>
<|im_start|>assistant
```

---

## 两者的关系与区别

### 分离关注点 (Separation of Concerns)

```
┌─────────────────────────────────────────────────────────┐
│                   数据处理管道                           │
└─────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
┌──────────────────────┐          ┌──────────────────────┐
│  Message Template    │          │   Chat Template      │
│                      │          │                      │
│  决定 "说什么"        │          │   决定 "怎么说"       │
│  (What to say)       │          │   (How to say)       │
├──────────────────────┤          ├──────────────────────┤
│ • 添加 system prompt │          │ • 添加特殊 token     │
│ • 组织 user content  │          │ • 格式化消息         │
│ • 插入占位符         │          │ • 处理多模态         │
│ • 设置 role          │          │ • 渲染成文本         │
└──────────────────────┘          └──────────────────────┘
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                    最终 Prompt 文本
```

### 协同工作

| 阶段 | Message Template 输出 | Chat Template 输出 |
|------|-----------------------|-------------------|
| **输入** | `[{"role": "user", "content": "问题"}]` | - |
| **处理** | 添加 system、组织内容 | - |
| **中间结果** | `[{"role": "system", ...}, {"role": "user", ...}]` | - |
| **输入** | - | 上述 messages |
| **处理** | - | 应用 Jinja2 模板 |
| **最终输出** | - | `"<|im_start|>system\n...<|im_end|>\n..."` |

### 为什么要分离？

1. **灵活性**
   - 不同任务可以用不同的 message template
   - 不同模型可以用不同的 chat template
   - 组合使用，适配各种场景

2. **可维护性**
   - Message template 关注业务逻辑（如 FrameThinker 的动作定义）
   - Chat template 关注模型格式（如 Qwen2.5-VL 的 ChatML）
   - 修改互不影响

3. **可扩展性**
   - 添加新任务：只需新增 message template
   - 支持新模型：只需更新 chat template
   - 无需修改核心代码

---

## FrameThinker 案例分析

### 完整配置

**训练脚本**: `recipe/framethinker/train_frame_thinker_vhonly.sh:53`
```bash
data.message_template=framethinker_add_zoomin
```

**配置文件**: `recipe/framethinker/config/think_with_video.yaml:16-19`
```yaml
actor_rollout_ref:
  model:
    custom_chat_template: "{% set image_count = namespace(value=0) %}..."
```

### 数据样例

**原始数据** (parquet):
```python
{
    "prompt": [{"role": "user", "content": "What action happens at timestamp 1:30 in the video?"}],
    "answer": "A",
    "extra_info": {
        "total_frames": 9000,
        "fps": 30,
        "video_path": "video123.mp4"
    },
    "multi_modal_data": {
        "image": [<8 PIL Images>]
    }
}
```

### 处理过程

#### Step 1: Message Template 处理

**函数**: `framethinker_add_zoomin.apply_message_template()`

**输入**:
```python
messages = [{"role": "user", "content": "What action happens at timestamp 1:30 in the video?"}]
kwargs = {
    "extra_info": {"total_frames": 9000, ...},
    "multi_modal_data": {"image": [8 images]}
}
```

**处理**:
1. 生成 frame indices: `[0, 1125, 2250, 3375, 4500, 5625, 6750, 7875]`
2. 生成 image placeholders: `"frame 0:<image>\nframe 1125:<image>\n..."`
3. 生成 system prompt (包含 `total_frames=9000`)

**输出**:
```python
messages = [
    {
        "role": "system",
        "content": "You are an expert AI assistant that answers questions about a video by iteratively analyzing it.\n..."
    },
    {
        "role": "user",
        "content": "What action happens at timestamp 1:30 in the video?\nframe 0:<image>\nframe 1125:<image>\n..."
    }
]
```

#### Step 2: 多模态占位符处理

**函数**: `RLDataset._build_messages()` 后续处理

**转换 `<image>` 为结构化格式**:
```python
messages = [
    {
        "role": "system",
        "content": "You are an expert AI assistant..."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What action happens at timestamp 1:30 in the video?\n"},
            {"type": "text", "text": "frame 0:"},
            {"type": "image"},
            {"type": "text", "text": "\nframe 1125:"},
            {"type": "image"},
            {"type": "text", "text": "\nframe 2250:"},
            {"type": "image"},
            # ... 共8个 image
        ]
    }
]
```

#### Step 3: Chat Template 渲染

**函数**: `processor.apply_chat_template(messages, add_generation_prompt=True)`

**使用模板**: Qwen2.5-VL custom chat template

**最终输出**:
```
<|im_start|>system
You are an expert AI assistant that answers questions about a video by iteratively analyzing it.
Your task is to output your reasoning within a <think> </think> tag, followed by a specific action within an <action> </action> tag.
Possible actions are:
1. `choose frames between START_FRAME and END_FRAME`: Request a more detailed view of a specific video segment. You MUST choose frames from 0 to 8999.
2. `get frame number at time MM:SS`: Get the exact frame number for a specific time. Convert hours to minutes if needed (e.g., for 1 hour, 2 minutes, and 30 seconds, use 62:30).
3. `zoom in frame FRAME_INDEX`: Zoom in on a specific frame and return the high-resolution image. The frame index must be an integer that appears in previous conversations. You MUST choose frames from 0 to 8999.
4. `output answer: OPTION`: Provide the final answer (e.g., A, B, C...) when you are confident.<|im_end|>
<|im_start|>user
What action happens at timestamp 1:30 in the video?
frame 0:<vision_start><image_pad><vision_end>
frame 1125:<vision_start><image_pad><vision_end>
frame 2250:<vision_start><image_pad><vision_end>
frame 3375:<vision_start><image_pad><vision_end>
frame 4500:<vision_start><image_pad><vision_end>
frame 5625:<vision_start><image_pad><vision_end>
frame 6750:<vision_start><image_pad><vision_end>
frame 7875:<vision_start><image_pad><vision_end><|im_end|>
<|im_start|>assistant
```

### 关键设计亮点

1. **动态 System Prompt**
   - 根据 `total_frames` 动态生成（如 "0 to 8999"）
   - 区分视频和图像任务
   - 包含所有允许的动作

2. **Frame Indices 显示**
   - 使用 `np.linspace` 均匀采样
   - 显示真实的 frame index（如 0, 1125, 2250...）
   - 帮助模型理解视频时间线

3. **多模态占位符**
   - `<image>` → `<vision_start><image_pad><vision_end>`
   - 保留 frame index 信息（如 "frame 1125:"）
   - 视觉 token 和文本 token 正确交织

---

## 自定义 Template 指南

### 创建自定义 Message Template

#### 步骤 1: 创建模板文件

**文件**: `verl/utils/dataset/templates/my_custom_template.py`

```python
def apply_message_template(messages, **kwargs):
    """
    自定义 message template

    Args:
        messages: 原始消息列表，通常为 [{"role": "user", "content": "..."}]
        **kwargs: 额外参数，包括:
            - config: 数据集配置
            - extra_info: 数据样本的 extra_info 字段
            - multi_modal_data: 多模态数据
            - 其他数据样本字段

    Returns:
        处理后的消息列表，通常包含 system 和 user 消息
    """
    # 1. 验证输入
    assert messages[0]["role"] != "system", "System should not be in original data"

    # 2. 提取参数
    question = messages[0]["content"]
    # custom_param = kwargs.get("custom_param", default_value)

    # 3. 构造 system prompt
    system_prompt = "You are a helpful assistant..."

    # 4. 构造完整 messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    return messages
```

#### 步骤 2: 注册模板

**文件**: `verl/utils/dataset/templates/__init__.py`

```python
from .my_custom_template import apply_message_template as apply_message_template_my_custom

MESSAGE_TEMPLATES = {
    "default": apply_message_template_default,
    "my_custom": apply_message_template_my_custom,  # 添加这行
    ...
}
```

#### 步骤 3: 使用模板

```bash
python -m verl.trainer.main_ppo \
    data.message_template=my_custom \
    ...
```

### 创建自定义 Chat Template

#### 步骤 1: 编写 Jinja2 模板

**文件**: `my_custom_chat_template.jinja2`

```jinja2
{% for message in messages %}
  [{{ message['role'] }}]
  {{ message['content'] }}
  [/{{ message['role'] }}]
{% endfor %}
{% if add_generation_prompt %}
  [assistant]
{% endif %}
```

#### 步骤 2: 配置使用

**方式 1: YAML 配置**

```yaml
actor_rollout_ref:
  model:
    custom_chat_template: "{% for message in messages %}[{{ message['role'] }}]{{ message['content'] }}[/{{ message['role'] }}]{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}"
```

**方式 2: 从文件加载** (需要自行实现)

```python
# 在数据预处理脚本中
with open("my_custom_chat_template.jinja2") as f:
    chat_template = f.read()

# 保存到配置或直接设置
processor.chat_template = chat_template
```

### 最佳实践

#### Message Template

1. **始终验证输入**
   ```python
   assert messages[0]["role"] != "system", "System should not be in original data"
   ```

2. **使用参数化 system prompt**
   ```python
   def get_system_prompt(param1, param2):
       return f"You are ... with {param1} and {param2}..."
   ```

3. **处理边界情况**
   ```python
   # 区分视频和图像
   is_video = len(multi_modal_data["image"]) > 1

   # 处理空数据
   if not question:
       raise ValueError("Question cannot be empty")
   ```

4. **保持简洁**
   - Message template 只负责消息结构
   - 不要做复杂的文本处理（留给 chat template）

#### Chat Template

1. **处理多种内容类型**
   ```jinja2
   {% if message['content'] is string %}
     {{ message['content'] }}
   {% else %}
     {% for content in message['content'] %}
       {% if content['type'] == 'text' %}
         {{ content['text'] }}
       {% elif content['type'] == 'image' %}
         <image_token>
       {% endif %}
     {% endfor %}
   {% endif %}
   ```

2. **添加必要的特殊 token**
   ```jinja2
   <|start|>{{ role }}<|sep|>{{ content }}<|end|>
   ```

3. **支持生成 prompt**
   ```jinja2
   {% if add_generation_prompt %}
     <|assistant|>
   {% endif %}
   ```

4. **测试模板**
   ```python
   # 单独测试 chat template
   test_messages = [
       {"role": "system", "content": "You are helpful."},
       {"role": "user", "content": "Hello!"}
   ]

   result = processor.apply_chat_template(
       test_messages,
       add_generation_prompt=True,
       tokenize=False
   )
   print(result)
   ```

---

## 常见问题

### Q1: 为什么我的 system prompt 没有生效？

**A**: 检查以下几点：

1. **是否配置了 message template？**
   ```bash
   data.message_template=framethinker_add_zoomin  # 必须指定
   ```

2. **Message template 是否正确添加了 system role？**
   ```python
   messages = [
       {"role": "system", "content": get_system_prompt(...)},  # 必须有这行
       {"role": "user", "content": ...}
   ]
   ```

3. **Chat template 是否正确处理 system role？**
   ```jinja2
   {% for message in messages %}  {# 必须遍历所有消息，包括 system #}
     <|im_start|>{{ message['role'] }}
     {{ message['content'] }}<|im_end|>
   {% endfor %}
   ```

### Q2: 原始数据集应该包含 system role 吗？

**A**: **不应该**。

- ❌ 不要在数据集中包含 system role
- ✅ 让 message template 动态添加 system role

**原因**:
- System prompt 通常与任务相关，不是数据的一部分
- 便于修改 system prompt（只需改模板，不需重新处理数据）
- 保持数据集简洁（只包含问题和答案）

### Q3: 如何调试 message template？

**A**: 在 message template 函数中添加打印：

```python
def apply_message_template(messages, **kwargs):
    print("=" * 80)
    print("Input messages:", messages)
    print("Extra info:", kwargs.get("extra_info"))
    print("Multi-modal data keys:", kwargs.get("multi_modal_data", {}).keys())

    # ... 处理逻辑 ...

    print("Output messages:", messages)
    print("=" * 80)

    return messages
```

### Q4: Chat template 中的 `add_generation_prompt` 是什么？

**A**: 控制是否添加 assistant 开始 token。

```jinja2
{% if add_generation_prompt %}
  <|im_start|>assistant
{% endif %}
```

**效果**:
- `add_generation_prompt=True`: 添加 `<|im_start|>assistant`，准备生成
- `add_generation_prompt=False`: 不添加，用于评估已有回复

**使用场景**:
- 训练/推理时: `add_generation_prompt=True`
- 评估完整对话时: `add_generation_prompt=False`

### Q5: 多模态占位符 `<image>` 是如何转换的？

**A**: 两步转换：

**Step 1: Message template 添加占位符**
```python
content = "What is this?\n<image>"  # 文本占位符
```

**Step 2: RLDataset 转换为结构化格式**
```python
# verl/utils/dataset/rl_dataset.py:288-310
content = [
    {"type": "text", "text": "What is this?\n"},
    {"type": "image"}  # 结构化占位符
]
```

**Step 3: Chat template 渲染为特殊 token**
```jinja2
{% if content['type'] == 'image' %}
  <|vision_start|><|image_pad|><|vision_end|>
{% endif %}
```

### Q6: 如何支持多轮对话？

**A**: Message template 需要处理多轮历史：

```python
def apply_message_template(messages, **kwargs):
    # messages 可能包含多轮对话
    # [
    #   {"role": "user", "content": "Q1"},
    #   {"role": "assistant", "content": "A1"},
    #   {"role": "user", "content": "Q2"}
    # ]

    # 添加 system 到第一条消息之前
    system_message = {"role": "system", "content": get_system_prompt()}
    messages = [system_message] + messages

    return messages
```

**Chat template 自动处理多轮**：
```jinja2
{% for message in messages %}
  <|im_start|>{{ message['role'] }}
  {{ message['content'] }}<|im_end|>
{% endfor %}
```

### Q7: 如何验证 template 正确性？

**A**: 单元测试：

```python
import pytest
from verl.utils.dataset.templates.framethinker_add_zoomin import apply_message_template

def test_message_template():
    # 准备测试数据
    messages = [{"role": "user", "content": "Test question"}]
    kwargs = {
        "extra_info": {"total_frames": 100},
        "multi_modal_data": {"image": [None] * 8}
    }

    # 应用模板
    result = apply_message_template(messages, **kwargs)

    # 验证
    assert len(result) == 2  # system + user
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert "0 to 99" in result[0]["content"]  # total_frames-1
    assert "frame 0:<image>" in result[1]["content"]

    print("✅ Message template test passed!")

def test_chat_template():
    from transformers import AutoProcessor

    # 加载 processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # 设置自定义模板
    processor.chat_template = "..."

    # 测试数据
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image"}
        ]}
    ]

    # 应用模板
    result = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # 验证
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result
    assert "<|vision_start|><|image_pad|><|vision_end|>" in result
    assert "<|im_start|>assistant" in result

    print("✅ Chat template test passed!")

if __name__ == "__main__":
    test_message_template()
    test_chat_template()
```

---

## 总结

### 核心要点

1. **System role 来自 message template，不是数据集**
   - 原始数据: 只有 user message
   - Message template: 添加 system message
   - Chat template: 格式化所有 messages

2. **Message template 决定"说什么"，Chat template 决定"怎么说"**
   - Message template: 业务逻辑（system prompt、占位符）
   - Chat template: 模型格式（特殊 token、多模态）

3. **分离设计带来灵活性**
   - 不同任务：切换 message template
   - 不同模型：切换 chat template
   - 组合使用：适配各种场景

### 配置清单

训练 FrameThinker 时需要配置：

```bash
# 1. Message template (决定 system prompt)
data.message_template=framethinker_add_zoomin

# 2. Chat template (决定格式)
actor_rollout_ref.model.custom_chat_template="{% set image_count = namespace(value=0) %}..."
```

### 参考文件

**Message Template**:
- `verl/utils/dataset/templates/__init__.py` - 注册表
- `verl/utils/dataset/templates/framethinker_add_zoomin.py` - FrameThinker 模板
- `verl/utils/dataset/rl_dataset.py:273-310` - 应用逻辑

**Chat Template**:
- `verl/workers/config/model.py:66` - 配置定义
- `verl/workers/fsdp_workers.py:306-310` - 设置逻辑
- `verl/utils/dataset/rl_dataset.py:199-200` - 应用逻辑

**配置示例**:
- `recipe/framethinker/train_frame_thinker_vhonly.sh:53` - Message template
- `recipe/framethinker/config/think_with_video.yaml:18-19` - Chat template

---

**文档维护**: 如有问题或建议，请提交 Issue 到 verl GitHub 仓库。
