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

    distilled_pipeline: Any  # DistilledPipeline instance for I2V (single keyframe)
    keyframe_pipeline: Any  # KeyframeInterpolationPipeline for multi-keyframe


class LTX2Generator(VideoGenerator):
    """LTX-2 video generation service.

    This service handles loading two LTX-2 pipelines:
    - DistilledPipeline: For fast I2V (single start frame) generation
    - KeyframeInterpolationPipeline: For multi-keyframe interpolation
    
    Both use the 19b-distilled checkpoint with FP8 for optimal performance.
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

    def load_models(self) -> None:
        """Load LTX-2 pipeline components.

        This should be called during application startup when MOCK_MODE=false.
        If dry_run is True, this will skip actual model loading and log the
        configuration that would be used.
        """
        if self.dry_run:
            logger.info("LTX-2 dry-run mode enabled - skipping model loading")
            logger.info(f"  Checkpoint: {self.settings.ltx2_checkpoint_path}")
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
                f"ltx-2-19b-distilled-fp8.safetensors --local-dir {checkpoint_path.parent}"
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

        logger.info(f"Loading LTX-2 pipelines from {checkpoint_path}")

        try:
            import torch
            from ltx_pipelines.distilled import DistilledPipeline
            from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
            from ltx_core.loader import LoraPathStrengthAndSDOps
        except ImportError as e:
            raise ImportError(
                "LTX-2 packages are required. Install with: "
                "pip install -e path/to/LTX-2/packages/ltx-pipelines"
            ) from e

        # Validate distilled LoRA path for KeyframeInterpolationPipeline
        distilled_lora_path = Path(self.settings.ltx2_distilled_lora_path)
        if not distilled_lora_path.exists():
            raise FileNotFoundError(
                f"LTX-2 distilled LoRA not found at {distilled_lora_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2 "
                f"ltx-2-19b-distilled-lora-384.safetensors --local-dir {distilled_lora_path.parent}"
            )

        # Initialize pipelines
        device = torch.device(self.settings.ltx2_device)
        
        # DistilledPipeline for I2V (single keyframe, fastest)
        logger.info("Loading DistilledPipeline for I2V generation...")
        distilled_pipeline = DistilledPipeline(
            checkpoint_path=str(checkpoint_path.absolute()),
            spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
            gemma_root=str(gemma_root.absolute()),
            loras=[],  # No extra LoRAs
            device=device,
            fp8transformer=self.settings.ltx2_fp8_enabled,
        )
        
        # KeyframeInterpolationPipeline for multi-keyframe (start + end frames)
        # Uses guiding latents for smooth transitions between keyframes
        logger.info("Loading KeyframeInterpolationPipeline for keyframe interpolation...")
        distilled_lora_spec = LoraPathStrengthAndSDOps(
            path=str(distilled_lora_path.absolute()),
            strength=1.0,
            sd_ops=None,
        )
        keyframe_pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=str(checkpoint_path.absolute()),
            distilled_lora=[distilled_lora_spec],
            spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
            gemma_root=str(gemma_root.absolute()),
            loras=[],  # No extra LoRAs
            device=device,
            fp8transformer=self.settings.ltx2_fp8_enabled,
        )

        self.components = LTX2Components(
            distilled_pipeline=distilled_pipeline,
            keyframe_pipeline=keyframe_pipeline,
        )
        self.is_loaded = True

        # Create temp directory for intermediate files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ltx2_")

        logger.info("LTX-2 pipelines loaded successfully")

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

        # Store original target dimensions (for final crop)
        target_width = params.width
        target_height = params.height
        
        # Calculate 64-divisible padded dimensions for two-stage pipeline
        padded_width = ((target_width + 63) // 64) * 64
        padded_height = ((target_height + 63) // 64) * 64
        
        # Update params to use padded dimensions for generation
        params.width = padded_width
        params.height = padded_height

        # Calculate frames needed for the requested duration
        requested_frames = math.ceil(params.duration_seconds * params.frame_rate) + 1
        num_frames = round_up_to_valid_frames(requested_frames)

        logger.info(
            f"Generating keyframe interpolation video",
            extra={
                "job_id": params.job_id,
                "target_size": f"{target_width}x{target_height}",
                "padded_size": f"{padded_width}x{padded_height}",
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
            lambda: self._generate_sync(
                params, 
                num_frames, 
                seed, 
                target_width=target_width, 
                target_height=target_height
            ),
        )

        return LTX2VideoResult(
            video_data=video_data,
            width=target_width,
            height=target_height,
            duration_seconds=params.duration_seconds,
            frame_rate=params.frame_rate,
            has_audio=has_audio,
            seed=seed,
            upscale_info=None,  # No external upscaling, LTX-2 handles it natively
        )

    def _generate_sync(
        self, 
        params: KeyframeInterpolationParams, 
        num_frames: int,
        seed: int,
        target_width: int,
        target_height: int,
    ) -> tuple[bytes, bool]:
        """Synchronous video generation (runs in thread pool).
        
        Routes to the appropriate pipeline based on keyframe count:
        - 1 keyframe: Uses DistilledPipeline (fastest, for I2V)
        - 2+ keyframes: Uses KeyframeInterpolationPipeline (guiding latents)
        
        Returns:
            Tuple of (video_bytes, has_audio)
        """
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

        # Wrap entire generation in inference_mode to match official LTX-2 CLI pattern
        # This ensures encode_video (which iterates the video generator) runs in the same context
        with torch.inference_mode():
            return self._generate_sync_inner(
                params, num_frames, seed, target_width, target_height,
                TilingConfig, get_video_chunks_number, encode_video, AUDIO_SAMPLE_RATE
            )

    def _generate_sync_inner(
        self,
        params: KeyframeInterpolationParams,
        num_frames: int,
        seed: int,
        target_width: int,
        target_height: int,
        TilingConfig,
        get_video_chunks_number,
        encode_video,
        AUDIO_SAMPLE_RATE,
    ) -> tuple[bytes, bool]:
        """Inner implementation of _generate_sync - runs inside inference_mode context."""

        assert self._temp_dir is not None
        assert self.components is not None

        # Save keyframe images to temp files
        # Preprocess each keyframe: center crop to target aspect ratio, then resize
        images: list[tuple[str, int, float]] = []
        for idx, (image_data, frame_idx, strength) in enumerate(params.keyframes):
            input_image = Image.open(io.BytesIO(image_data))
            orig_w, orig_h = input_image.size
            
            # Center crop to match target aspect ratio (avoids stretching)
            target_aspect = params.width / params.height
            orig_aspect = orig_w / orig_h
            
            if abs(orig_aspect - target_aspect) > 0.01:  # Aspect ratios differ
                if orig_aspect > target_aspect:
                    # Image is wider, crop width
                    new_w = int(orig_h * target_aspect)
                    new_h = orig_h
                    left = (orig_w - new_w) // 2
                    top = 0
                else:
                    # Image is taller, crop height
                    new_w = orig_w
                    new_h = int(orig_w / target_aspect)
                    left = 0
                    top = (orig_h - new_h) // 2
                
                input_image = input_image.crop((left, top, left + new_w, top + new_h))
                logger.info(
                    f"Center cropped keyframe {idx} from {orig_w}x{orig_h} to {new_w}x{new_h}"
                )
            
            # Resize to target dimensions (now with correct aspect ratio)
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

        # Choose pipeline based on keyframe count:
        # - 1 keyframe (I2V): Use DistilledPipeline (fastest, latent replacement)
        # - 2+ keyframes: Use KeyframeInterpolationPipeline (guiding latents for smooth transitions)
        use_keyframe_pipeline = len(params.keyframes) >= 2
        
        if use_keyframe_pipeline:
            logger.info(f"Using KeyframeInterpolationPipeline for {len(params.keyframes)} keyframes")
            video_chunks, audio = self.components.keyframe_pipeline(
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
        else:
            logger.info("Using DistilledPipeline for single-frame I2V")
            video_chunks, audio = self.components.distilled_pipeline(
                prompt=params.prompt,
                seed=seed,
                height=params.height,
                width=params.width,
                num_frames=num_frames,
                frame_rate=params.frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=params.enhance_prompt,
            )

        # Consolidate video chunks into a single tensor for cropping/trimming
        import torch
        if isinstance(video_chunks, torch.Tensor):
            video_tensor = video_chunks
        else:
            # Materialize iterator (list of tensors) and concatenate
            chunks = list(video_chunks)
            if not chunks:
                 raise RuntimeError("No video chunks generated")
            video_tensor = torch.cat(chunks, dim=0)

        # Apply In-Memory Optimization
        logger.info("Applying in-memory cropping and trimming...")
        
        # Calculate target frames based on exact duration
        # +1 frame because start frame is included (e.g. 24fps * 5s = 120 frames, but we might have generated 121 or 129)
        # Actually video_processing logic used exact duration.
        # params.duration_seconds * params.frame_rate gives float.
        # We ceil-ed it for generation. Now we floor/exact it for trim.
        target_frames_exact = int(params.duration_seconds * params.frame_rate)
        
        # Ensure we have at least 1 frame (sanity check)
        target_frames_exact = max(1, target_frames_exact)
        
        video_tensor = self._crop_and_trim_video(
            video_tensor,
            current_width=params.width, # Padded width
            current_height=params.height, # Padded height
            target_width=target_width,
            target_height=target_height,
            target_frames=target_frames_exact
        )

        if audio is not None:
            audio = self._trim_audio(
                audio,
                target_duration=params.duration_seconds,
                sample_rate=AUDIO_SAMPLE_RATE
            )

        # Encode to MP4 with audio
        output_path = Path(self._temp_dir.name) / f"{params.job_id}_output.mp4"
        
        # When passing a single tensor, video_chunks_number is 1
        video_chunks_number = 1

        encode_video(
            video=video_tensor,
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

    def _crop_and_trim_video(
        self,
        video: "torch.Tensor",
        current_width: int,
        current_height: int,
        target_width: int,
        target_height: int,
        target_frames: int,
    ) -> "torch.Tensor":
        """Crop and trim video tensor to target dimensions and frame count.

        Assumes video tensor shape is (Frames, Height, Width, Channels).
        """
        # 1. Trim frames
        if video.shape[0] > target_frames:
            logger.info(f"Trimming video tensor from {video.shape[0]} to {target_frames} frames")
            video = video[:target_frames]

        # 2. Crop dimensions
        if current_width != target_width or current_height != target_height:
            logger.info(f"Cropping video tensor from {current_width}x{current_height} to {target_width}x{target_height}")
            
            # Center crop
            x_center = current_width // 2
            y_center = current_height // 2
            x1 = x_center - target_width // 2
            y1 = y_center - target_height // 2
            x2 = x1 + target_width
            y2 = y1 + target_height
            
            # Slice: (Frames, Height, Width, Channels)
            video = video[:, y1:y2, x1:x2, :]
        
        return video

    def _trim_audio(
        self,
        audio: "torch.Tensor",
        target_duration: float,
        sample_rate: int,
    ) -> "torch.Tensor":
        """Trim audio tensor to target duration."""
        target_samples = int(target_duration * sample_rate)
        
        # Audio shape: (Samples, Channels) or (Samples,)
        if audio.shape[0] > target_samples:
            logger.info(f"Trimming audio tensor from {audio.shape[0]} to {target_samples} samples")
            audio = audio[:target_samples]
            
        return audio

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
                
                # Delete both pipelines
                if hasattr(self.components, 'distilled_pipeline') and self.components.distilled_pipeline is not None:
                    del self.components.distilled_pipeline
                if hasattr(self.components, 'keyframe_pipeline') and self.components.keyframe_pipeline is not None:
                    del self.components.keyframe_pipeline
                
                # Clear the components container
                self.components = None
                
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
