"""Mock ComfyUI generator for local development."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass

from typing import Any

from app.config import Settings
from app.services.placeholder import PlaceholderGenerator
from app.services.interfaces import ImageGenerator, ImageEditor, VideoGenerator


from app.models.internal import (
    ImageGenerationParams,
    ImageGenerationResult,
    ImageEditParams,
    ImageEditResult,
    VideoGenerationParams,
    VideoGenerationResult,
    VideoGenerationParams as LTX2VideoParams,
    VideoGenerationResult as LTX2VideoResult,
    KeyframeInterpolationParams,
)

logger = logging.getLogger(__name__)


class MockGenerator(ImageGenerator, ImageEditor, VideoGenerator):
    """Mock ComfyUI generator for local development."""

    def __init__(self, settings: Settings):
        """Initialize mock generator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.placeholder = PlaceholderGenerator()

    # BaseModelGenerator Implementation
    def load_models(self) -> None:
        pass

    def unload_models(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"status": "mock", "is_loaded": True}

    @property
    def _loaded(self) -> bool:
        return True

    # ImageGenerator Implementation
    async def load_lora(self, lora_name: str, weight: float = 1.0) -> None:
        pass

    async def unload_lora(self) -> None:
        pass

    # VideoGenerator Implementation
    def set_upscaler(self, upscaler: Any) -> None:
        pass

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
                "seed": seed,
            },
        )

        # Simulate processing delay (3-5 seconds)
        delay = random.uniform(3.0, 5.0)
        await asyncio.sleep(delay)

        # Generate placeholder edited image
        # Note: We pass default values for removed parameters if the placeholder requires them
        image_data, orig_w, orig_h, out_w, out_h = self.placeholder.create_edited_image(
            input_image_data=params.input_image_data,
            edit_type="style_transfer",  # Defaulting to style_transfer since it was removed
            prompt=params.prompt,
            job_id=params.job_id,
            seed=seed,
        )

        return ImageEditResult(
            image_data=image_data,
            original_width=orig_w,
            original_height=orig_h,
            width=out_w,
            height=out_h,
            seed=seed,
        )

    async def generate_video(
        self, params: VideoGenerationParams | LTX2VideoParams
    ) -> VideoGenerationResult | LTX2VideoResult:
        """Generate a mock video.

        Args:
            params: Generation parameters (VideoGenerationParams or LTX2VideoParams)

        Returns:
            VideoGenerationResult or LTX2VideoResult with placeholder video
        """
        # Check if this is an LTX-2 style params (has frame_rate instead of fps)
        if hasattr(params, 'frame_rate'):
            return await self._generate_ltx2_video(params)
        
        # Legacy VideoGenerationParams handling
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

    # ========================================================================
    # LTX-2 Video Generation Methods (for MockGenerator compatibility)
    # ========================================================================

    async def _generate_ltx2_video(self, params: LTX2VideoParams) -> LTX2VideoResult:
        """Generate a mock LTX-2 I2V video (internal helper).

        Args:
            params: LTX-2 I2V generation parameters

        Returns:
            LTX2VideoResult with placeholder video
        """
        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Mock generating LTX-2 I2V video",
            extra={
                "job_id": params.job_id,
                "duration": params.duration_seconds,
                "frame_rate": params.frame_rate,
                "seed": seed,
            },
        )

        # Simulate processing delay
        delay = random.uniform(3.0, 6.0)
        await asyncio.sleep(delay)

        # Generate placeholder video
        loop = asyncio.get_event_loop()
        video_data, width, height = await loop.run_in_executor(
            None,
            lambda: self.placeholder.create_video(
                input_image_data=params.input_image_data,
                prompt=params.prompt,
                job_id=params.job_id,
                duration_seconds=params.duration_seconds,
                fps=int(params.frame_rate),
                seed=seed,
            ),
        )

        return LTX2VideoResult(
            video_data=video_data,
            width=width,
            height=height,
            duration_seconds=params.duration_seconds,
            frame_rate=params.frame_rate,
            has_audio=False,  # Mock doesn't generate audio
            seed=seed,
        )

    async def generate_keyframe_video(
        self, params: KeyframeInterpolationParams
    ) -> LTX2VideoResult:
        """Generate a mock keyframe interpolation video.

        Args:
            params: Keyframe interpolation generation parameters

        Returns:
            LTX2VideoResult with placeholder video
        """
        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Mock generating keyframe interpolation video",
            extra={
                "job_id": params.job_id,
                "num_keyframes": len(params.keyframes),
                "duration": params.duration_seconds,
                "frame_rate": params.frame_rate,
                "seed": seed,
            },
        )

        # Simulate processing delay
        delay = random.uniform(3.0, 6.0)
        await asyncio.sleep(delay)

        # Use first keyframe for placeholder
        first_keyframe_data = params.keyframes[0][0] if params.keyframes else b""

        # Generate placeholder video
        loop = asyncio.get_event_loop()
        video_data, width, height = await loop.run_in_executor(
            None,
            lambda: self.placeholder.create_video(
                input_image_data=first_keyframe_data,
                prompt=params.prompt,
                job_id=params.job_id,
                duration_seconds=params.duration_seconds,
                fps=int(params.frame_rate),
                seed=seed,
            ),
        )

        return LTX2VideoResult(
            video_data=video_data,
            width=width,
            height=height,
            duration_seconds=params.duration_seconds,
            frame_rate=params.frame_rate,
            has_audio=False,  # Mock doesn't generate audio
            seed=seed,
        )
