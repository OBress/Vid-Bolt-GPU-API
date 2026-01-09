"""LTX-2 Video Generation Service.

This module provides the LTX2Generator service for generating videos using
the LTX-2 model with two modes:
- I2V (Image-to-Video): Generate video from a single start frame
- Keyframe Interpolation: Generate video by interpolating between multiple keyframes

Key features:
- Two-stage pipeline: low-res generation + 2x upsampling with distilled LoRA
- Keyframe conditioning with guiding latents for smooth transitions
- Async wrapper for synchronous PyTorch inference
- Dry-run mode for testing without models
- FP8 support for lower VRAM usage
- Audio generation with synchronized video
- Post-generation trimming to exact requested duration
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from app.services.video_upscaler import StreamDiffVSRUpscaler

from app.config import Settings
from app.services.interfaces import VideoGenerator
from app.models.internal import (
    VideoGenerationParams as LTX2VideoParams,
    KeyframeInterpolationParams,
    VideoGenerationResult as LTX2VideoResult,
)
from app.models.ltx2_generation import round_up_to_valid_frames

logger = logging.getLogger(__name__)


# ============================================================================
# Parameter and Result Dataclasses
# ============================================================================




@dataclass
class LTX2Components:
    """Container for loaded LTX-2 pipeline components."""

    pipeline: Any  # KeyframeInterpolationPipeline instance


class LTX2Generator(VideoGenerator):
    """LTX-2 video generation service.

    This service handles loading the LTX-2 KeyframeInterpolationPipeline
    with 19b-dev checkpoint and distilled LoRA for two-stage video generation.
    Supports both I2V and keyframe interpolation modes.
    
    The pipeline generates synchronized audio alongside video.
    """

    def __init__(self, settings: Settings):
        """Initialize the LTX-2 generator.

        Args:
            settings: Application settings containing model paths and configuration
        """
        self.settings = settings
        self.components: LTX2Components | None = None
        self.is_loaded = False
        self.dry_run = settings.ltx2_dry_run
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._upscaler: StreamDiffVSRUpscaler | None = None

    def set_upscaler(self, upscaler: Any) -> None:
        """Set the video upscaler for post-generation enhancement.

        Args:
            upscaler: StreamDiffVSRUpscaler instance to use for upscaling
        """
        self._upscaler = upscaler
        logger.info("Video upscaler connected to LTX-2 generator")

    def load_models(self) -> None:
        """Load LTX-2 pipeline components.

        This should be called during application startup when MOCK_MODE=false.
        If dry_run is True, this will skip actual model loading and log the
        configuration that would be used.
        """
        if self.dry_run:
            logger.info("LTX-2 dry-run mode enabled - skipping model loading")
            logger.info(f"  Checkpoint: {self.settings.ltx2_checkpoint_path}")
            logger.info(f"  Distilled LoRA: {self.settings.ltx2_distilled_lora_path}")
            logger.info(f"  Spatial upsampler: {self.settings.ltx2_spatial_upsampler_path}")
            logger.info(f"  Gemma root: {self.settings.ltx2_gemma_root}")
            logger.info(f"  Device: {self.settings.ltx2_device}")
            logger.info(f"  FP8 enabled: {self.settings.ltx2_fp8_enabled}")
            self.is_loaded = True
            self._temp_dir = tempfile.TemporaryDirectory(prefix="ltx2_")
            return

        # Validate all required paths exist
        checkpoint_path = Path(self.settings.ltx2_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LTX-2 checkpoint not found at {checkpoint_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2 "
                f"ltx-2-19b-dev.safetensors --local-dir {checkpoint_path.parent}"
            )

        distilled_lora_path = Path(self.settings.ltx2_distilled_lora_path)
        if not distilled_lora_path.exists():
            raise FileNotFoundError(
                f"LTX-2 distilled LoRA not found at {distilled_lora_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2 "
                f"ltx-2-19b-distilled-lora-384.safetensors --local-dir {distilled_lora_path.parent}"
            )

        spatial_upsampler_path = Path(self.settings.ltx2_spatial_upsampler_path)
        if not spatial_upsampler_path.exists():
            raise FileNotFoundError(
                f"LTX-2 spatial upsampler not found at {spatial_upsampler_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2 "
                f"ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir {spatial_upsampler_path.parent}"
            )

        gemma_root = Path(self.settings.ltx2_gemma_root)
        if not gemma_root.exists():
            raise FileNotFoundError(
                f"Gemma text encoder not found at {gemma_root.absolute()}. "
                f"Download with: huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized "
                f"--local-dir {gemma_root}"
            )

        logger.info(f"Loading LTX-2 KeyframeInterpolationPipeline from {checkpoint_path}")

        try:
            import torch
            from ltx_core.loader import LoraPathStrengthAndSDOps
            from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        except ImportError as e:
            raise ImportError(
                "LTX-2 packages are required. Install with: "
                "pip install -e path/to/LTX-2/packages/ltx-pipelines"
            ) from e

        # Prepare distilled LoRA
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                str(distilled_lora_path.absolute()),
                1.0,
                {}  # Use default key mapping
            )
        ]

        # Initialize pipeline
        device = torch.device(self.settings.ltx2_device)
        pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=str(checkpoint_path.absolute()),
            distilled_lora=distilled_lora,
            spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
            gemma_root=str(gemma_root.absolute()),
            loras=[],  # No additional LoRAs
            device=device,
            fp8transformer=self.settings.ltx2_fp8_enabled,
        )

        self.components = LTX2Components(pipeline=pipeline)
        self.is_loaded = True

        # Create temp directory for intermediate files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ltx2_")

        logger.info("LTX-2 KeyframeInterpolationPipeline loaded successfully")

    # ========================================================================
    # I2V (Image-to-Video) Generation
    # ========================================================================

    async def generate_video(self, params: LTX2VideoParams) -> LTX2VideoResult:
        """Generate a video from a single start frame (I2V mode).

        This is a convenience method that converts I2V parameters to keyframe
        format internally (start frame at index 0, optional end frame at last index).

        Args:
            params: I2V generation parameters

        Returns:
            LTX2VideoResult containing the generated video data

        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(
                "LTX-2 models not loaded. Call load_models() first or set "
                "LTX2_DRY_RUN=true for testing."
            )

        # Calculate frames needed for the requested duration
        requested_frames = math.ceil(params.duration_seconds * params.frame_rate) + 1
        num_frames = round_up_to_valid_frames(requested_frames)
        
        # Build keyframes list: start frame at index 0
        keyframes: list[tuple[bytes, int, float]] = [
            (params.input_image_data, 0, 1.0)
        ]
        
        # Add end frame if provided
        if params.end_image_data is not None:
            keyframes.append((params.end_image_data, num_frames - 1, 1.0))

        # Convert to keyframe params
        keyframe_params = KeyframeInterpolationParams(
            job_id=params.job_id,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            keyframes=keyframes,
            duration_seconds=params.duration_seconds,
            frame_rate=params.frame_rate,
            width=params.width,
            height=params.height,
            seed=params.seed,
            enhance_prompt=params.enhance_prompt,
        )

        # Delegate to keyframe generation
        return await self.generate_keyframe_video(keyframe_params)

    # ========================================================================
    # Keyframe Interpolation Generation
    # ========================================================================

    async def generate_keyframe_video(
        self, params: KeyframeInterpolationParams
    ) -> LTX2VideoResult:
        """Generate a video by interpolating between keyframes.

        Args:
            params: Generation parameters including keyframes, prompt, dimensions

        Returns:
            LTX2VideoResult containing the generated video data

        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(
                "LTX-2 models not loaded. Call load_models() first or set "
                "LTX2_DRY_RUN=true for testing."
            )

        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        # Calculate frames needed for the requested duration
        requested_frames = math.ceil(params.duration_seconds * params.frame_rate) + 1
        num_frames = round_up_to_valid_frames(requested_frames)

        logger.info(
            f"Generating keyframe interpolation video",
            extra={
                "job_id": params.job_id,
                "width": params.width,
                "height": params.height,
                "requested_duration": params.duration_seconds,
                "num_frames": num_frames,
                "frame_rate": params.frame_rate,
                "num_keyframes": len(params.keyframes),
                "seed": seed,
                "dry_run": self.dry_run,
            },
        )

        if self.dry_run:
            return await self._generate_dry_run(params, num_frames, seed)

        # Run the synchronous generation in a thread pool
        loop = asyncio.get_event_loop()
        video_data, has_audio = await loop.run_in_executor(
            None,
            lambda: self._generate_sync(params, num_frames, seed),
        )

        # Trim video to exact requested duration
        from app.utils.video_processing import trim_video_to_duration
        trimmed_video = await loop.run_in_executor(
            None,
            lambda: trim_video_to_duration(video_data, params.duration_seconds),
        )

        # Apply video upscaling if upscaler is available and enabled
        final_video = trimmed_video
        upscale_info: dict[str, Any] | None = None

        if (
            self._upscaler is not None
            and self._upscaler.is_loaded
            and self.settings.stream_diffvsr_enabled
        ):
            from app.services.video_upscaler import UpscaleParams

            logger.info(
                f"Upscaling video from 720p to 1080p",
                extra={"job_id": params.job_id},
            )

            upscale_params = UpscaleParams(
                job_id=params.job_id,
                video_data=trimmed_video,
                preserve_audio=True,
            )

            upscale_result = await self._upscaler.upscale_video(upscale_params)
            final_video = upscale_result.video_data

            upscale_info = {
                "original_resolution": f"{upscale_result.original_width}x{upscale_result.original_height}",
                "upscaled_resolution": f"{upscale_result.upscaled_width}x{upscale_result.upscaled_height}",
                "frame_count": upscale_result.frame_count,
                "upscale_time_seconds": round(upscale_result.processing_time_seconds, 2),
                "was_upscaled": upscale_result.was_upscaled,
            }

            logger.info(
                f"Video upscaling completed",
                extra={
                    "job_id": params.job_id,
                    "upscale_time_s": upscale_info["upscale_time_seconds"],
                },
            )

        return LTX2VideoResult(
            video_data=final_video,
            width=params.width,
            height=params.height,
            duration_seconds=params.duration_seconds,
            frame_rate=params.frame_rate,
            has_audio=has_audio,
            seed=seed,
            upscale_info=upscale_info,
        )

    def _generate_sync(
        self, 
        params: KeyframeInterpolationParams, 
        num_frames: int,
        seed: int
    ) -> tuple[bytes, bool]:
        """Synchronous video generation (runs in thread pool).
        
        Returns:
            Tuple of (video_bytes, has_audio)
        """
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

        assert self._temp_dir is not None
        assert self.components is not None

        # Save keyframe images to temp files
        images: list[tuple[str, int, float]] = []
        for idx, (image_data, frame_idx, strength) in enumerate(params.keyframes):
            input_image = Image.open(io.BytesIO(image_data))
            
            # Resize if needed to match target dimensions
            if input_image.size != (params.width, params.height):
                input_image = input_image.resize(
                    (params.width, params.height),
                    Image.Resampling.LANCZOS
                )

            image_path = Path(self._temp_dir.name) / f"{params.job_id}_keyframe_{idx}.png"
            input_image.save(image_path, format="PNG")
            images.append((str(image_path), frame_idx, strength))

        # Configure tiling for video decoding
        tiling_config = TilingConfig.default()

        # Generate video with audio
        video_chunks, audio = self.components.pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            seed=seed,
            height=params.height,
            width=params.width,
            num_frames=num_frames,
            frame_rate=params.frame_rate,
            num_inference_steps=self.settings.ltx2_num_inference_steps,
            cfg_guidance_scale=self.settings.ltx2_cfg_guidance_scale,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=params.enhance_prompt,
        )

        # Encode to MP4 with audio
        output_path = Path(self._temp_dir.name) / f"{params.job_id}_output.mp4"
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        encode_video(
            video=video_chunks,
            fps=params.frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )

        # Read output
        with open(output_path, "rb") as f:
            video_data = f.read()

        # Determine if audio was generated
        has_audio = audio is not None

        # Clean up temp files
        try:
            for idx in range(len(params.keyframes)):
                keyframe_path = Path(self._temp_dir.name) / f"{params.job_id}_keyframe_{idx}.png"
                keyframe_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")

        return video_data, has_audio

    async def _generate_dry_run(
        self, 
        params: KeyframeInterpolationParams, 
        num_frames: int,
        seed: int
    ) -> LTX2VideoResult:
        """Generate a placeholder video for dry-run testing."""
        # Simulate processing time (longer for video)
        await asyncio.sleep(2.0)

        # Create a simple placeholder video using moviepy
        # We use PIL for the image generation to avoid MoviePy TextClip API changes/dependencies
        try:
            from moviepy import ImageClip
        except ImportError:
            try:
                # Fallback for older moviepy
                from moviepy.editor import ImageClip
            except ImportError:
                # Final fallback: create minimal placeholder without moviepy
                return await self._generate_minimal_placeholder(params, seed)

        duration_seconds = params.duration_seconds

        # Use PIL to create the frame content
        from PIL import Image, ImageDraw, ImageFont

        # Create gradient-like background (solid color for simplicity)
        img = Image.new('RGB', (params.width, params.height), color=(50, 100, 150))
        draw = ImageDraw.Draw(img)

        # Draw text
        text_lines = [
            "LTX-2 DRY RUN",
            f"Job: {params.job_id[:8]}...",
            f"Size: {params.width}x{params.height}",
            f"Duration: {duration_seconds}s @ {params.frame_rate}fps",
            f"Keyframes: {len(params.keyframes)}",
            f"Seed: {seed}",
        ]

        # Basic font loading
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Draw centered text
        line_height = 30
        y_offset = (params.height - (len(text_lines) * line_height)) // 2

        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (params.width - text_width) // 2
            
            # Simple shadow
            draw.text((x+2, y_offset+2), line, fill="black", font=font)
            draw.text((x, y_offset), line, fill="white", font=font)
            y_offset += line_height

        # Save to temp image file
        assert self._temp_dir is not None
        temp_img_path = Path(self._temp_dir.name) / f"{params.job_id}_dryrun_frame.png"
        img.save(temp_img_path)

        # Create video from image
        temp_video_path = Path(self._temp_dir.name) / f"{params.job_id}_dryrun.mp4"
        
        try:
            # Note: MoviePy 1.x uses duration in constructor, 2.x supports it too but favors method chaining
            # We use the constructor for broader compatibility if possible, or we could check version
            clip = ImageClip(str(temp_img_path), duration=duration_seconds)
            
            clip.write_videofile(
                str(temp_video_path),
                fps=int(params.frame_rate),
                codec="libx264",
                audio=False,
                logger=None,
            )
            
            # Close clip explicitly to release resources
            clip.close()
            
        except Exception as e:
            logger.error(f"MoviePy generation failed: {e}")
            # Clean up image
            if temp_img_path.exists():
                temp_img_path.unlink()
            # Fallback
            return await self._generate_minimal_placeholder(params, seed)

        # Read result
        with open(temp_video_path, "rb") as f:
            video_data = f.read()

        # Cleanup
        if temp_img_path.exists():
            temp_img_path.unlink()
        if temp_video_path.exists():
            temp_video_path.unlink()



        return LTX2VideoResult(
            video_data=video_data,
            width=params.width,
            height=params.height,
            duration_seconds=duration_seconds,
            frame_rate=params.frame_rate,
            has_audio=False,  # Dry-run doesn't generate audio
            seed=seed,
        )

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the generator."""
        return {
            "generator_type": "LTX2Generator",
            "model": "LTX-2-19b-dev + Distilled LoRA",
            "is_loaded": self.is_loaded,
            "dry_run": self.dry_run,
            "checkpoint_path": self.settings.ltx2_checkpoint_path,
            "distilled_lora_path": self.settings.ltx2_distilled_lora_path,
            "spatial_upsampler_path": self.settings.ltx2_spatial_upsampler_path,
            "gemma_root": self.settings.ltx2_gemma_root,
            "device": self.settings.ltx2_device,
            "fp8_enabled": self.settings.ltx2_fp8_enabled,
            "num_inference_steps": self.settings.ltx2_num_inference_steps,
            "cfg_guidance_scale": self.settings.ltx2_cfg_guidance_scale,
        }

    @property
    def _loaded(self) -> bool:
        """Alias for is_loaded for ModelManager compatibility."""
        return self.is_loaded

    def unload_models(self) -> None:
        """Unload models and free GPU memory.
        
        This is called by ModelManager when switching modes to release VRAM.
        """
        if not self.is_loaded:
            logger.info("LTX-2 models not loaded, nothing to unload")
            return
        
        logger.info("Unloading LTX-2 models...")
        
        if self.components is not None:
            import gc
            try:
                import torch
                
                # Delete the pipeline
                if hasattr(self.components, 'pipeline') and self.components.pipeline is not None:
                    del self.components.pipeline
                
                # Clear the components container
                self.components = None
                self._upscaler = None  # Remove upscaler reference
                
                # Force garbage collection and clear CUDA cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except ImportError:
                self.components = None
                gc.collect()
        
        self.is_loaded = False
        logger.info("LTX-2 models unloaded successfully")

    async def _generate_minimal_placeholder(
        self, 
        params: KeyframeInterpolationParams, 
        seed: int
    ) -> LTX2VideoResult:
        """Generate a minimal placeholder video without moviepy.
        
        Creates a simple MP4 file using raw bytes as a fallback when
        moviepy is not available.
        """
        import struct
        
        # Create a minimal valid MP4 file (solid color video)
        # This is a very basic placeholder - in production you'd want
        # to use a proper video encoding library
        
        # For now, just create an empty placeholder that signals dry-run
        placeholder_data = b"PLACEHOLDER_VIDEO_DRY_RUN"
        
        return LTX2VideoResult(
            video_data=placeholder_data,
            width=params.width,
            height=params.height,
            duration_seconds=params.duration_seconds,
            frame_rate=params.frame_rate,
            has_audio=False,
            seed=seed,
        )

    def __del__(self):
        """Cleanup temporary directory on deletion."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
