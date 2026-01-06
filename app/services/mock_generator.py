"""Mock ComfyUI generator for local development."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass

from app.config import Settings
from app.services.placeholder import PlaceholderGenerator

logger = logging.getLogger(__name__)


@dataclass
class ImageGenerationParams:
    """Parameters for image generation."""

    job_id: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    seed: int | None
    num_inference_steps: int


@dataclass
class ImageGenerationResult:
    """Result of image generation."""

    image_data: bytes
    width: int
    height: int
    seed: int


@dataclass
class ImageEditParams:
    """Parameters for image editing."""

    job_id: str
    input_image_data: bytes
    prompt: str
    edit_type: str
    strength: float
    mask_data: bytes | None
    seed: int | None


@dataclass
class ImageEditResult:
    """Result of image editing."""

    image_data: bytes
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    seed: int


@dataclass
class VideoGenerationParams:
    """Parameters for video generation."""

    job_id: str
    input_image_data: bytes
    prompt: str
    duration_seconds: float
    fps: int
    motion_strength: float
    seed: int | None
    end_image_data: bytes | None = None


@dataclass
class VideoGenerationResult:
    """Result of video generation."""

    video_data: bytes
    width: int
    height: int
    duration_seconds: float
    fps: int
    frame_count: int
    seed: int


class MockGenerator:
    """Mock ComfyUI generator for local development."""

    def __init__(self, settings: Settings):
        """Initialize mock generator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.placeholder = PlaceholderGenerator()

    async def generate_image(self, params: ImageGenerationParams) -> ImageGenerationResult:
        """Generate a mock image.

        Args:
            params: Generation parameters

        Returns:
            ImageGenerationResult with placeholder image
        """
        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Mock generating image",
            extra={
                "job_id": params.job_id,
                "width": params.width,
                "height": params.height,
                "seed": seed,
            },
        )

        # Simulate processing delay (2-4 seconds)
        delay = random.uniform(2.0, 4.0)
        await asyncio.sleep(delay)

        # Generate placeholder image
        image_data = self.placeholder.create_image(
            width=params.width,
            height=params.height,
            prompt=params.prompt,
            job_id=params.job_id,
            seed=seed,
        )

        return ImageGenerationResult(
            image_data=image_data,
            width=params.width,
            height=params.height,
            seed=seed,
        )

    async def edit_image(self, params: ImageEditParams) -> ImageEditResult:
        """Edit a mock image.

        Args:
            params: Edit parameters

        Returns:
            ImageEditResult with placeholder edited image
        """
        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Mock editing image",
            extra={
                "job_id": params.job_id,
                "edit_type": params.edit_type,
                "strength": params.strength,
                "seed": seed,
            },
        )

        # Simulate processing delay (3-5 seconds)
        delay = random.uniform(3.0, 5.0)
        await asyncio.sleep(delay)

        # Generate placeholder edited image
        image_data, orig_w, orig_h, out_w, out_h = self.placeholder.create_edited_image(
            input_image_data=params.input_image_data,
            edit_type=params.edit_type,
            prompt=params.prompt,
            job_id=params.job_id,
            seed=seed,
        )

        return ImageEditResult(
            image_data=image_data,
            original_width=orig_w,
            original_height=orig_h,
            output_width=out_w,
            output_height=out_h,
            seed=seed,
        )

    async def generate_video(self, params: VideoGenerationParams) -> VideoGenerationResult:
        """Generate a mock video.

        Args:
            params: Generation parameters

        Returns:
            VideoGenerationResult with placeholder video
        """
        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Mock generating video",
            extra={
                "job_id": params.job_id,
                "duration": params.duration_seconds,
                "fps": params.fps,
                "seed": seed,
            },
        )

        # Simulate processing delay (5-10 seconds)
        delay = random.uniform(5.0, 10.0)
        await asyncio.sleep(delay)

        # Calculate frame count
        frame_count = int(params.duration_seconds * params.fps)

        # Generate placeholder video (run in thread to avoid blocking)
        loop = asyncio.get_event_loop()
        video_data, width, height = await loop.run_in_executor(
            None,
            lambda: self.placeholder.create_video(
                input_image_data=params.input_image_data,
                prompt=params.prompt,
                job_id=params.job_id,
                duration_seconds=params.duration_seconds,
                fps=params.fps,
                seed=seed,
            ),
        )

        return VideoGenerationResult(
            video_data=video_data,
            width=width,
            height=height,
            duration_seconds=params.duration_seconds,
            fps=params.fps,
            frame_count=frame_count,
            seed=seed,
        )
