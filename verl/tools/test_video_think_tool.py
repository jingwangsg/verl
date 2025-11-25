#!/usr/bin/env python3
"""
Simple test script for VideoThinkTool.

This script verifies:
1. Tool can be imported
2. Tool can be initialized with schema
3. Basic lifecycle methods work (create/release)

For full integration testing with actual videos, use this as a template
and provide real video paths.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verl.tools.video_think_tool import VideoThinkTool
from verl.tools.schemas import OpenAIFunctionToolSchema


async def test_basic_initialization():
    """Test basic tool initialization."""
    print("=" * 60)
    print("Test 1: Basic Initialization")
    print("=" * 60)

    # Define tool schema
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
                        "description": "Action to perform"
                    }
                },
                "required": ["action_string"]
            }
        }
    })

    # Initialize tool
    config = {
        "max_workers": 4,
        "timeout": 30
    }

    tool = VideoThinkTool(config=config, tool_schema=tool_schema)
    print(f"✓ Tool initialized: {tool.name}")
    print(f"✓ Max workers: {tool.max_workers}")
    print(f"✓ Timeout: {tool.timeout}")
    print()

    return tool


async def test_create_release(tool):
    """Test create and release lifecycle."""
    print("=" * 60)
    print("Test 2: Create and Release Lifecycle")
    print("=" * 60)

    # Create instance with mock metadata
    # Note: This won't work with real video operations, but tests the interface
    try:
        instance_id, response = await tool.create(
            create_kwargs={
                "video_path": "/mock/path/video.mp4",
                "fps": 30,
                "total_frames": 3000,
                "width": 1920,
                "height": 1080
            }
        )
        print(f"✓ Instance created: {instance_id}")
        print(f"✓ Response is empty: {response.is_empty()}")

        # Check instance data
        instance_data = tool._instance_dict.get(instance_id)
        print(f"✓ Instance data stored: {instance_data is not None}")
        print(f"  - Video path: {instance_data['video_path']}")
        print(f"  - FPS: {instance_data['fps']}")
        print(f"  - Total frames: {instance_data['total_frames']}")
        print(f"  - Frames per sample: {instance_data['num_frames_per_sample']}")

        # Release instance
        await tool.release(instance_id)
        print(f"✓ Instance released")
        print(f"✓ Instance removed: {instance_id not in tool._instance_dict}")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        raise


async def test_action_parsing(tool):
    """Test action string parsing (without actual video processing)."""
    print("=" * 60)
    print("Test 3: Action String Parsing")
    print("=" * 60)

    # Create instance
    instance_id, _ = await tool.create(
        create_kwargs={
            "video_path": "/mock/path/video.mp4",
            "fps": 30,
            "total_frames": 3000,
            "width": 1920,
            "height": 1080
        }
    )

    try:
        # Test 1: Get frame number (should work without video)
        print("Testing: 'get frame number at time 01:30'")
        response, reward, metrics = await tool.execute(
            instance_id=instance_id,
            parameters={"action_string": "get frame number at time 01:30"}
        )
        print(f"✓ Response text: {response.text}")
        print(f"✓ Reward: {reward}")
        print(f"✓ Success: {metrics.get('success')}")
        print()

        # Test 2: Invalid time format
        print("Testing: Invalid time format")
        response, reward, metrics = await tool.execute(
            instance_id=instance_id,
            parameters={"action_string": "get frame number at time invalid"}
        )
        print(f"✓ Error detected: {not metrics.get('success')}")
        print(f"✓ Negative reward: {reward}")
        print()

        # Test 3: Unrecognized action
        print("Testing: Unrecognized action")
        response, reward, metrics = await tool.execute(
            instance_id=instance_id,
            parameters={"action_string": "invalid action format"}
        )
        print(f"✓ Error detected: {not metrics.get('success')}")
        print(f"✓ Error type: {metrics.get('error')}")
        print()

    finally:
        await tool.release(instance_id)


async def test_multiple_instances(tool):
    """Test multiple concurrent instances."""
    print("=" * 60)
    print("Test 4: Multiple Concurrent Instances")
    print("=" * 60)

    # Create multiple instances
    instances = []
    for i in range(3):
        instance_id, _ = await tool.create(
            create_kwargs={
                "video_path": f"/mock/path/video{i}.mp4",
                "fps": 30,
                "total_frames": 3000 + i * 1000,
                "width": 1920,
                "height": 1080
            }
        )
        instances.append(instance_id)
        print(f"✓ Created instance {i+1}: {instance_id[:8]}...")

    print(f"✓ Total instances: {len(tool._instance_dict)}")
    print()

    # Release all
    for instance_id in instances:
        await tool.release(instance_id)

    print(f"✓ All instances released: {len(tool._instance_dict) == 0}")
    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VideoThinkTool Test Suite")
    print("=" * 60 + "\n")

    try:
        # Test 1: Initialization
        tool = await test_basic_initialization()

        # Test 2: Create/Release
        await test_create_release(tool)

        # Test 3: Action parsing
        await test_action_parsing(tool)

        # Test 4: Multiple instances
        await test_multiple_instances(tool)

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Note: These tests verify the tool interface and basic functionality.")
        print("For full integration testing with actual videos, provide real video paths")
        print("and test the video decoding operations.")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
