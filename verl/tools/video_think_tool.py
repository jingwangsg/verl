# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import random
import re
import time
from functools import partial
from typing import Any, Optional, Tuple, List
from uuid import uuid4

import numpy as np
import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder
from torchvision.transforms.functional import resize as tv_resize

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_target_size(width: int, height: int, size: int) -> tuple[int, int]:
    """
    Compute target size for video frame resizing.

    Args:
        width: Original width
        height: Original height
        size: Target size for the longer dimension

    Returns:
        Tuple of (target_width, target_height)
    """
    if width > height:
        return size, int(size * height / width)
    else:
        return int(size * width / height), size


class VideoThinkTool(BaseTool):
    """A tool for interactive video exploration with frame-level analysis.

    This tool provides functionality for:
    - Getting frame numbers from timestamps
    - Zooming in on specific frames (high-resolution)
    - Selecting frame ranges for analysis

    The tool uses regex-based action parsing internally for compatibility
    with existing models, while providing a standard OpenAI tool schema
    for registration.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a video trajectory
        execute: Execute video exploration actions
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance and clean up resources
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize the VideoThinkTool.

        Args:
            config: Configuration dictionary containing:
                - max_workers: Number of worker threads (default: 4)
                - timeout: Timeout for operations in seconds (default: 30)
            tool_schema: OpenAI function tool schema for registration
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Configuration
        self.max_workers = config.get("max_workers", 4)
        self.timeout = config.get("timeout", 30)

        logger.info(f"Initialized VideoThinkTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema in OpenAI format.

        Note: The tool uses regex parsing internally, but this schema
        is used for tool registration and documentation.
        """
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance for a video trajectory.

        Args:
            instance_id: Optional unique identifier for the instance
            **kwargs: Should contain 'create_kwargs' with video metadata:
                - video_path: Path to the video file
                - fps: Frames per second of the video
                - total_frames: Total number of frames
                - width: Video width
                - height: Video height

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Extract video metadata
        video_path = kwargs.get("video_path")
        fps = kwargs.get("fps")
        total_frames = kwargs.get("total_frames")
        width = kwargs.get("width")
        height = kwargs.get("height")

        if not video_path:
            raise ValueError("Missing required 'video_path' parameter in kwargs")
        if fps is None or total_frames is None:
            raise ValueError("Missing required 'fps' or 'total_frames' parameter in kwargs")

        # Calculate num_frames_per_sample based on video duration
        duration_seconds = total_frames / fps
        if duration_seconds > 300:  # Long video (> 5 minutes)
            num_frames_per_sample = 12
        else:  # Short video (<= 5 minutes)
            num_frames_per_sample = 8

        # Store instance data (decoders will be initialized lazily)
        self._instance_dict[instance_id] = {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "num_frames_per_sample": num_frames_per_sample,
            "vr": None,  # Low-res decoder (256px) - lazy init
            "vr_highres": None,  # High-res decoder (360px) - lazy init
        }

        logger.info(
            f"Created VideoThinkTool instance {instance_id}: "
            f"video={video_path}, fps={fps}, frames={total_frames}, "
            f"duration={duration_seconds:.1f}s, samples={num_frames_per_sample}"
        )

        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute a video exploration action.

        Supported actions (parsed via regex):
        1. "get frame number at time MM:SS" - Convert timestamp to frame number
        2. "zoom in frame <FRAME_INDEX>" - Extract high-res single frame
        3. "choose frames between <START> and <END>" - Sample frame range

        Args:
            instance_id: The instance ID of the tool
            parameters: Dictionary containing 'action_string' with the action to execute

        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        instance_data = self._instance_dict.get(instance_id)
        if not instance_data:
            return (
                ToolResponse(text="Error: Invalid instance ID"),
                -0.1,
                {"success": False, "error": "invalid_instance"}
            )

        # Extract action string from parameters
        action_string = parameters.get("action_string", "")
        if not action_string:
            return (
                ToolResponse(text="Error: Missing action_string parameter"),
                -0.05,
                {"success": False, "error": "missing_action"}
            )

        # Parse and execute action
        try:
            # Action 1: Get frame number at time
            time_match = re.match(r"get frame number at time\s+(\S+)", action_string.strip())
            if time_match:
                return await self._handle_get_frame_number(instance_data, time_match.group(1))

            # Action 2: Zoom in on specific frame
            zoom_match = re.match(r"zoom in frame\s+(\d+)", action_string.strip())
            if zoom_match:
                frame_idx = int(zoom_match.group(1))
                return await self._handle_zoom_in_frame(instance_data, frame_idx)

            # Action 3: Choose frames between range
            range_match = re.search(r"choose frames between (\d+) and (\d+)", action_string)
            if range_match:
                start_frame = int(range_match.group(1))
                end_frame = int(range_match.group(2))
                return await self._handle_choose_frames(instance_data, start_frame, end_frame)

            # No match found
            return (
                ToolResponse(text=f"Error: Unrecognized action format: '{action_string}'"),
                -0.05,
                {"success": False, "error": "unrecognized_action"}
            )

        except Exception as e:
            logger.error(f"Error executing action '{action_string}': {e}")
            return (
                ToolResponse(text=f"Error executing action: {str(e)}"),
                -0.1,
                {"success": False, "error": "execution_error"}
            )

    async def _handle_get_frame_number(
        self, instance_data: dict, time_str: str
    ) -> tuple[ToolResponse, float, dict]:
        """Handle 'get frame number at time MM:SS' action.

        Args:
            instance_data: Instance data dictionary
            time_str: Time string in MM:SS format

        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        try:
            minutes, seconds = map(int, time_str.split(":"))
            total_seconds = minutes * 60 + seconds
            frame_number = int(total_seconds * instance_data["fps"])

            message = f"Frame number at time {time_str} is: {frame_number}."
            logger.info(f"Computed frame number: time={time_str} -> frame={frame_number}")

            return (
                ToolResponse(text=message),
                0.0,
                {"success": True, "action": "get_frame_number", "frame": frame_number}
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid time format: {time_str} - {e}")
            return (
                ToolResponse(text=f"Error: Invalid time format '{time_str}'. Expected MM:SS."),
                -0.05,
                {"success": False, "error": "invalid_time_format"}
            )

    async def _handle_zoom_in_frame(
        self, instance_data: dict, frame_idx: int
    ) -> tuple[ToolResponse, float, dict]:
        """Handle 'zoom in frame <FRAME_INDEX>' action.

        Extracts a single high-resolution frame (360px target size).

        Args:
            instance_data: Instance data dictionary
            frame_idx: Frame index to extract

        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        # Validate frame index
        if frame_idx < 0 or frame_idx >= instance_data["total_frames"]:
            return (
                ToolResponse(
                    text=f"Error: Frame index {frame_idx} out of range [0, {instance_data['total_frames']-1}]"
                ),
                -0.05,
                {"success": False, "error": "frame_out_of_range"}
            )

        try:
            # Initialize high-res decoder if needed
            if instance_data["vr_highres"] is None:
                await self._init_highres_decoder(instance_data)

            # Extract frame in executor (blocking operation)
            loop = asyncio.get_running_loop()
            frame_image = await loop.run_in_executor(
                None,
                self._extract_highres_frame,
                instance_data,
                frame_idx
            )

            logger.info(f"Zoomed in on frame {frame_idx}")

            return (
                ToolResponse(
                    text=f"Zoomed in on frame {frame_idx}",
                    image=[frame_image]
                ),
                0.0,
                {"success": True, "action": "zoom_in_frame", "frame": frame_idx}
            )

        except Exception as e:
            logger.error(f"Failed to zoom in on frame {frame_idx}: {e}")
            return (
                ToolResponse(text=f"Error: Failed to zoom in on frame {frame_idx}"),
                -0.1,
                {"success": False, "error": "zoom_failed"}
            )

    async def _handle_choose_frames(
        self, instance_data: dict, start_frame: int, end_frame: int
    ) -> tuple[ToolResponse, float, dict]:
        """Handle 'choose frames between <START> and <END>' action.

        Samples 8-12 frames uniformly from the specified range.

        Args:
            instance_data: Instance data dictionary
            start_frame: Start frame index
            end_frame: End frame index

        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        total_frames = instance_data["total_frames"]
        num_frames_per_sample = instance_data["num_frames_per_sample"]

        # Validate frame range
        if start_frame >= total_frames:
            return (
                ToolResponse(text=f"Error: Start frame {start_frame} exceeds total frames {total_frames}"),
                -0.05,
                {"success": False, "error": "invalid_range"}
            )

        # Clamp end_frame
        if end_frame > total_frames:
            end_frame = total_frames

        # Check if range is sufficient
        if start_frame >= end_frame - num_frames_per_sample:
            return (
                ToolResponse(
                    text=f"Error: Frame range [{start_frame}, {end_frame}) too small for sampling {num_frames_per_sample} frames"
                ),
                -0.05,
                {"success": False, "error": "range_too_small"}
            )

        try:
            # Initialize low-res decoder if needed
            if instance_data["vr"] is None:
                await self._init_lowres_decoder(instance_data)

            # Extract frames in executor (blocking operation)
            loop = asyncio.get_running_loop()
            prompt_text, frame_images = await loop.run_in_executor(
                None,
                self._extract_frame_range,
                instance_data,
                start_frame,
                end_frame
            )

            if len(frame_images) == 0:
                return (
                    ToolResponse(text="Error: Failed to extract frames from range"),
                    -0.1,
                    {"success": False, "error": "extraction_failed"}
                )

            logger.info(f"Extracted {len(frame_images)} frames from range [{start_frame}, {end_frame})")

            return (
                ToolResponse(
                    text=prompt_text,
                    image=frame_images
                ),
                0.0,
                {
                    "success": True,
                    "action": "choose_frames",
                    "start": start_frame,
                    "end": end_frame,
                    "num_frames": len(frame_images)
                }
            )

        except Exception as e:
            logger.error(f"Failed to extract frame range [{start_frame}, {end_frame}): {e}")
            return (
                ToolResponse(text=f"Error: Failed to extract frame range"),
                -0.1,
                {"success": False, "error": "extraction_failed"}
            )

    async def _init_highres_decoder(self, instance_data: dict) -> None:
        """Initialize high-resolution VideoDecoder (360px target size).

        Args:
            instance_data: Instance data dictionary
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                loop = asyncio.get_running_loop()
                decoder = await loop.run_in_executor(
                    None,
                    partial(VideoDecoder, instance_data["video_path"], num_ffmpeg_threads=0),
                )
                instance_data["vr_highres"] = decoder
                logger.info(f"Initialized high-res decoder for {instance_data['video_path']}")
                return
            except Exception as e:
                if "Resource temporarily unavailable" in str(e) and attempt < max_retries - 1:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(
                        f"[Attempt {attempt + 1}/{max_retries}] Failed to open video for high-res. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[Attempt {attempt + 1}/{max_retries}] Failed to open video for high-res: {e}")
                    raise

    async def _init_lowres_decoder(self, instance_data: dict) -> None:
        """Initialize low-resolution VideoDecoder (256px target size).

        Args:
            instance_data: Instance data dictionary
        """
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                loop = asyncio.get_running_loop()
                decoder = await loop.run_in_executor(
                    None,
                    partial(VideoDecoder, instance_data["video_path"], num_ffmpeg_threads=0),
                )
                instance_data["vr"] = decoder
                logger.info(f"Initialized low-res decoder for {instance_data['video_path']}")
                return
            except Exception as e:
                if "Resource temporarily unavailable" in str(e) and attempt < max_retries - 1:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(
                        f"[Attempt {attempt + 1}/{max_retries}] Failed to open video for low-res. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[Attempt {attempt + 1}/{max_retries}] Failed to open video for low-res: {e}")
                    raise

    def _extract_highres_frame(self, instance_data: dict, frame_idx: int) -> Image.Image:
        """Extract a single high-resolution frame (blocking operation).

        Args:
            instance_data: Instance data dictionary
            frame_idx: Frame index to extract

        Returns:
            PIL Image of the frame
        """
        vr_highres = instance_data["vr_highres"]
        width = instance_data["width"]
        height = instance_data["height"]

        # Calculate target size
        target_h, target_w = compute_target_size(width, height, size=360)

        # Extract frame
        frame_tensor = vr_highres[frame_idx]  # [C, H, W]

        # Resize to target size
        frame_tensor_resized = tv_resize(frame_tensor, [target_h, target_w])

        # Convert to [H, W, C] numpy array
        frame_array = frame_tensor_resized.permute(1, 2, 0).cpu().numpy()
        frame_image = Image.fromarray(frame_array)

        return frame_image

    def _extract_frame_range(
        self, instance_data: dict, start_frame: int, end_frame: int
    ) -> Tuple[str, List[Image.Image]]:
        """Extract a range of frames (blocking operation).

        Args:
            instance_data: Instance data dictionary
            start_frame: Start frame index
            end_frame: End frame index

        Returns:
            Tuple of (prompt_text, frame_images_list)
        """
        vr = instance_data["vr"]
        width = instance_data["width"]
        height = instance_data["height"]
        num_frames_per_sample = instance_data["num_frames_per_sample"]

        # Calculate target size
        target_h, target_w = compute_target_size(width, height, size=256)

        # Sample frame indices
        sample_start = start_frame + 1
        sample_end = end_frame - 1

        frame_indices = sorted(
            list(
                set(
                    map(
                        int,
                        np.linspace(sample_start, sample_end, num_frames_per_sample)
                    )
                )
            )
        )

        if len(frame_indices) == 0:
            return "", []

        # Get frames using torchcodec
        frame_batch = vr.get_frames_at(frame_indices)
        focused_frames_tensor = frame_batch.data  # [N, C, H, W]

        # Resize all frames to target size
        frames_resized = torch.stack([tv_resize(f, [target_h, target_w]) for f in focused_frames_tensor])

        # Convert to [N, H, W, C] numpy array
        focused_frames_array = frames_resized.permute(0, 2, 3, 1).cpu().numpy()

        # Build prompt and convert to PIL Images
        prompt_parts = []
        for frame_idx in frame_indices:
            prompt_parts.append(f"frame {frame_idx}: <image>")

        prompt_text = "\n".join(prompt_parts)
        frame_images = [Image.fromarray(frame) for frame in focused_frames_array]

        return prompt_text, frame_images

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and clean up resources.

        Args:
            instance_id: The instance ID of the tool
        """
        if instance_id in self._instance_dict:
            instance_data = self._instance_dict[instance_id]

            # Clean up decoders
            if instance_data.get("vr") is not None:
                # VideoDecoder cleanup (if needed)
                instance_data["vr"] = None

            if instance_data.get("vr_highres") is not None:
                # VideoDecoder cleanup (if needed)
                instance_data["vr_highres"] = None

            del self._instance_dict[instance_id]
            logger.info(f"Released VideoThinkTool instance {instance_id}")
