# VideoThinkTool Usage Guide

## Overview

`VideoThinkTool` is an async implementation of the `think_with_video` tool from FrameThinker-RL, designed to work with the existing `ToolAgentLoop` infrastructure.

## Features

- **Async execution**: Non-blocking video processing for better scalability
- **Three core actions**:
  1. Get frame number from timestamp
  2. Zoom in on specific frames (high-res 360px)
  3. Sample frame ranges (low-res 256px, 8-12 frames)
- **Lazy initialization**: Video decoders are initialized only when needed
- **Retry logic**: Handles resource contention with exponential backoff
- **Adaptive sampling**: Adjusts frame count based on video duration

## Quick Start

### 1. Import the tool

```python
from verl.tools import VideoThinkTool
from verl.tools.schemas import OpenAIFunctionToolSchema
```

### 2. Define the tool schema

```python
tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "video_think",
        "description": "Interactive video exploration tool for frame-level analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "action_string": {
                    "type": "string",
                    "description": "Action to perform (see supported actions)"
                }
            },
            "required": ["action_string"]
        }
    }
})
```

### 3. Initialize the tool

```python
config = {
    "max_workers": 4,
    "timeout": 30
}

tool = VideoThinkTool(config=config, tool_schema=tool_schema)
```

### 4. Create a tool instance for a video

```python
instance_id, _ = await tool.create(
    create_kwargs={
        "video_path": "/path/to/video.mp4",
        "fps": 30,
        "total_frames": 9000,
        "width": 1920,
        "height": 1080
    }
)
```

### 5. Execute actions

```python
# Action 1: Get frame number from timestamp
response, reward, metrics = await tool.execute(
    instance_id=instance_id,
    parameters={"action_string": "get frame number at time 01:30"}
)
print(response.text)  # "Frame number at time 01:30 is: 2700."

# Action 2: Zoom in on specific frame
response, reward, metrics = await tool.execute(
    instance_id=instance_id,
    parameters={"action_string": "zoom in frame 2700"}
)
print(response.image[0])  # PIL Image (high-res)

# Action 3: Sample frame range
response, reward, metrics = await tool.execute(
    instance_id=instance_id,
    parameters={"action_string": "choose frames between 100 and 500"}
)
print(response.text)      # "frame 101: <image>\nframe 151: <image>\n..."
print(len(response.image))  # 8-12 PIL Images (low-res)
```

### 6. Release the instance

```python
await tool.release(instance_id)
```

## Supported Actions

### 1. Get Frame Number at Time
**Format**: `get frame number at time MM:SS`

**Example**: `get frame number at time 01:30`

**Returns**: Text response with frame number

```python
{
    "text": "Frame number at time 01:30 is: 2700.",
    "image": None
}
```

### 2. Zoom In Frame
**Format**: `zoom in frame <FRAME_INDEX>`

**Example**: `zoom in frame 2700`

**Returns**: High-resolution frame image (360px target size)

```python
{
    "text": "Zoomed in on frame 2700",
    "image": [PIL.Image]  # Single high-res frame
}
```

### 3. Choose Frames Between
**Format**: `choose frames between <START> and <END>`

**Example**: `choose frames between 100 and 500`

**Returns**: Sampled frames from range (8-12 frames, 256px target size)

```python
{
    "text": "frame 101: <image>\nframe 151: <image>\n...",
    "image": [PIL.Image, PIL.Image, ...]  # 8-12 low-res frames
}
```

## Integration with ToolAgentLoop

### Configuration

Add to your agent config YAML:

```yaml
tools:
  - name: video_think
    class: verl.tools.VideoThinkTool
    config:
      max_workers: 4
      timeout: 30
    tool_schema:
      type: function
      function:
        name: video_think
        description: "Interactive video exploration tool"
        parameters:
          type: object
          properties:
            action_string:
              type: string
              description: "Action string (see documentation)"
          required: ["action_string"]
```

### Usage in Agent Loop

The tool will be automatically available to the agent loop. Pass video metadata in `create_kwargs`:

```python
agent_loop = ToolAgentLoop(...)
result = await agent_loop.run(
    sampling_params=sampling_params,
    tools_kwargs={
        "create_kwargs": {
            "video_path": "/path/to/video.mp4",
            "fps": 30,
            "total_frames": 9000,
            "width": 1920,
            "height": 1080
        }
    }
)
```

## Technical Details

### Video Decoder Management

- **Low-res decoder** (256px): Used for frame range sampling
- **High-res decoder** (360px): Used for single frame zoom
- **Lazy initialization**: Decoders created on first use
- **Retry logic**: 3 attempts with exponential backoff for resource contention
- **Cleanup**: Decoders released in `release()` method

### Frame Sampling Strategy

- **Short videos** (≤ 5 minutes): 8 frames per sample
- **Long videos** (> 5 minutes): 12 frames per sample
- **Uniform sampling**: Frames evenly distributed across range
- **Deduplication**: Duplicate indices removed

### Async Implementation

All blocking operations (video decoding) are executed in a thread pool executor:

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(None, blocking_function, *args)
```

This ensures the async event loop remains responsive during video processing.

## Comparison with Original Implementation

| Feature | Original (think_with_video) | New (VideoThinkTool) |
|---------|---------------------------|---------------------|
| Execution | Synchronous | Async |
| Interface | Custom ToolBase | Standard BaseTool |
| Registration | Metaclass registry | Import-based |
| Resource lifecycle | Manual reset() | Standard create/release |
| Response format | Dict/string | ToolResponse schema |
| Chat template | Manual formatting | Handled by agent loop |

## Error Handling

The tool returns negative rewards and error messages for common issues:

- **Invalid action format**: -0.05 reward
- **Frame out of range**: -0.05 reward
- **Invalid time format**: -0.05 reward
- **Extraction failure**: -0.10 reward
- **Range too small**: -0.05 reward

All errors are logged at appropriate levels (WARNING/ERROR) for debugging.

## Performance Considerations

1. **Decoder caching**: Decoders persist across tool calls within a trajectory
2. **Resolution tiers**: Low-res for browsing, high-res for inspection
3. **Async execution**: Non-blocking I/O for concurrent operations
4. **Resource pooling**: Consider Ray-based pooling for high-throughput scenarios

## Example: Complete Workflow

```python
import asyncio
from verl.tools import VideoThinkTool
from verl.tools.schemas import OpenAIFunctionToolSchema

async def main():
    # Initialize tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({...})
    tool = VideoThinkTool(config={"max_workers": 4}, tool_schema=tool_schema)

    # Create instance
    instance_id, _ = await tool.create(
        create_kwargs={
            "video_path": "video.mp4",
            "fps": 30,
            "total_frames": 3000,
            "width": 1920,
            "height": 1080
        }
    )

    try:
        # Get frame at timestamp
        resp1, _, _ = await tool.execute(
            instance_id, {"action_string": "get frame number at time 00:30"}
        )
        print(resp1.text)

        # Zoom on that frame
        resp2, _, _ = await tool.execute(
            instance_id, {"action_string": "zoom in frame 900"}
        )
        resp2.image[0].save("zoomed_frame.jpg")

        # Sample nearby frames
        resp3, _, _ = await tool.execute(
            instance_id, {"action_string": "choose frames between 850 and 950"}
        )
        print(f"Got {len(resp3.image)} frames")

    finally:
        # Always release resources
        await tool.release(instance_id)

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Issue: "Resource temporarily unavailable"
**Solution**: The retry logic should handle this automatically. If it persists, increase `max_retries` in decoder initialization.

### Issue: Out of memory
**Solution**: Reduce concurrent instances or implement Ray-based pooling for decoder sharing.

### Issue: Frames not extracted
**Solution**: Check video file integrity and ensure ffmpeg/torchcodec is properly installed.

### Issue: Wrong frame count
**Solution**: Verify fps and total_frames metadata matches actual video properties.

## Future Enhancements

- Ray-based decoder pooling for high concurrency
- Support for more video formats
- Configurable resolution targets
- Caching of frequently accessed frames
- Metrics/telemetry integration
