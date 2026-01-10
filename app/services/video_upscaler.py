"""Stream-DiffVSR Video Upscaling Service.

This module provides the StreamDiffVSRUpscaler service for upscaling
720p videos to 1080p using diffusion-based super-resolution.

Key features:
- Auto-regressive temporal guidance for motion-aware upscaling
- 4-step distilled denoiser for real-time performance (~0.328s latency per frame)
- Temporal VAE decoder for enhanced temporal coherence
- Optical flow guidance via RAFT for motion alignment
- Async wrapper for synchronous PyTorch inference
- Dry-run mode for testing without loading models
- Audio preservation during upscaling
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from app.config import Settings
from app.services.interfaces import Upscaler
from app.models.internal import UpscaleParams, UpscaleResult

logger = logging.getLogger(__name__)





@dataclass
class StreamDiffVSRComponents:
    """Container for loaded Stream-DiffVSR pipeline components."""

    pipeline: Any  # StreamDiffVSRPipeline instance
    of_model: Any  # RAFT optical flow model
    device: Any  # torch.device


# ============================================================================
# StreamDiffVSRUpscaler Service
# ============================================================================


class StreamDiffVSRUpscaler(Upscaler):
    """Stream-DiffVSR video upscaling service.

    Uses diffusion-based super-resolution with auto-regressive temporal
    guidance to upscale 720p videos to 1080p while preserving motion,
    details, and temporal coherence.

    The service follows the same patterns as other generators in the codebase:
    - Lazy model loading via load_models()
    - Dry-run mode for testing without GPU
    - Async interface with sync implementation in thread pool

    Performance:
    - ~0.328s latency per 720p frame on RTX 4090
    - ~5-10 seconds total for a 4-second 24fps video
    - ~8GB VRAM usage
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the Stream-DiffVSR upscaler.

        Args:
            settings: Application settings containing model paths and configuration
        """
        self.settings = settings
        self.components: StreamDiffVSRComponents | None = None
        self.is_loaded = False
        self.dry_run = settings.stream_diffvsr_dry_run
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def load_models(self) -> None:
        """Load Stream-DiffVSR pipeline components.

        This should be called during application startup when MOCK_MODE=false
        and STREAM_DIFFVSR_ENABLED=true.

        If dry_run is True, this will skip actual model loading and log the
        configuration that would be used.
        """
        if self.dry_run:
            logger.info("Stream-DiffVSR dry-run mode enabled - skipping model loading")
            logger.info(f"  Model ID: {self.settings.stream_diffvsr_model_id}")
            logger.info(f"  Device: {self.settings.stream_diffvsr_device}")
            logger.info(f"  Inference steps: {self.settings.stream_diffvsr_num_inference_steps}")
            self.is_loaded = True
            self._temp_dir = tempfile.TemporaryDirectory(prefix="stream_diffvsr_")
            return

        logger.info("Loading Stream-DiffVSR models...")

        import sys

        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        # Add Stream-DiffVSR to path for imports
        stream_diffvsr_path = Path(__file__).parent.parent.parent / "Stream-DiffVSR"
        if not stream_diffvsr_path.exists():
            raise FileNotFoundError(
                f"Stream-DiffVSR directory not found at {stream_diffvsr_path}. "
                "Ensure the Stream-DiffVSR repository is cloned in the project root."
            )

        # Insert at beginning to ensure our local copy is used
        sys.path.insert(0, str(stream_diffvsr_path))

        try:
            from diffusers import DDIMScheduler

            from pipeline.stream_diffvsr_pipeline import (
                ControlNetModel,
                StreamDiffVSRPipeline,
                UNet2DConditionModel,
            )
            from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
        except ImportError as e:
            raise ImportError(
                f"Failed to import Stream-DiffVSR components: {e}. "
                "Ensure all dependencies are installed: pip install -r requirements.txt"
            ) from e

        device = torch.device(self.settings.stream_diffvsr_device)
        model_id = self.settings.stream_diffvsr_model_id

        logger.info(f"Loading components from HuggingFace: {model_id}")

        # Load pipeline components from HuggingFace
        # These will be downloaded automatically on first use
        controlnet = ControlNetModel.from_pretrained(model_id, subfolder="controlnet")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        vae = TemporalAutoencoderTiny.from_pretrained(model_id, subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Build the pipeline
        pipeline = StreamDiffVSRPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        # Move pipeline to GPU with bfloat16 for Blackwell compatibility (Issue #10)
        # - We avoid enable_xformers_memory_efficient_attention() because xformers kernels
        #   don't support float32 on Blackwell (Compute Capability 12.0)
        # - Using bfloat16 enables PyTorch's native SDPA which works on all modern GPUs
        # - No CPU offload needed with ample VRAM (RTX 6000 Pro has plenty)
        pipeline = pipeline.to(device, dtype=torch.bfloat16)

        logger.info("Loading RAFT optical flow model...")

        # Load optical flow model for temporal guidance
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        of_model.requires_grad_(False)

        self.components = StreamDiffVSRComponents(
            pipeline=pipeline,
            of_model=of_model,
            device=device,
        )
        self.is_loaded = True

        # Create temp directory for intermediate files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="stream_diffvsr_")

        logger.info("Stream-DiffVSR models loaded successfully")
        logger.info(f"  Device: {device}")
        logger.info(f"  Inference steps: {self.settings.stream_diffvsr_num_inference_steps}")

    async def upscale_video(self, params: UpscaleParams) -> UpscaleResult:
        """Upscale a video from 720p to 1080p.

        This is the main entry point for video upscaling. The method:
        1. Extracts frames and audio from the input video
        2. Upscales each frame using Stream-DiffVSR with temporal guidance
        3. Reconstructs the video with the original audio

        Args:
            params: Upscaling parameters including video data and target resolution

        Returns:
            UpscaleResult containing the upscaled video data and metadata

        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Stream-DiffVSR models not loaded. Call load_models() first or set "
                "STREAM_DIFFVSR_DRY_RUN=true for testing."
            )

        start_time = time.time()

        logger.info(
            f"Starting video upscaling",
            extra={
                "job_id": params.job_id,
                "dry_run": self.dry_run,
            },
        )

        if self.dry_run:
            return await self._upscale_dry_run(params, start_time)

        # Run synchronous upscaling in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._upscale_sync(params, start_time),
        )

        logger.info(
            f"Video upscaling completed",
            extra={
                "job_id": params.job_id,
                "original_resolution": f"{result.original_width}x{result.original_height}",
                "upscaled_resolution": f"{result.upscaled_width}x{result.upscaled_height}",
                "frame_count": result.frame_count,
                "processing_time_s": round(result.processing_time_seconds, 2),
            },
        )

        return result

    def _upscale_sync(self, params: UpscaleParams, start_time: float) -> UpscaleResult:
        """Synchronous video upscaling implementation.

        This runs in a thread pool to avoid blocking the async event loop.
        """
        import cv2
        import numpy as np
        from PIL import Image

        assert self._temp_dir is not None
        assert self.components is not None

        temp_path = Path(self._temp_dir.name)

        # Step 1: Write input video to temp file
        input_video_path = temp_path / f"{params.job_id}_input.mp4"
        with open(input_video_path, "wb") as f:
            f.write(params.video_data)

        # Step 2: Extract frames and audio
        frames, fps, audio_path = self._extract_frames_and_audio(
            input_video_path, params.job_id
        )

        if not frames:
            raise ValueError("No frames extracted from video")

        original_height, original_width = frames[0].size[1], frames[0].size[0]
        frame_count = len(frames)

        logger.info(
            f"Extracted {frame_count} frames at {fps}fps from {original_width}x{original_height}"
        )

        # Step 3: Upscale frames using Stream-DiffVSR
        logger.info("Running Stream-DiffVSR upscaling pipeline...")

        upscaled_output = self.components.pipeline(
            "",  # Empty prompt (unconditional super-resolution)
            frames,
            num_inference_steps=self.settings.stream_diffvsr_num_inference_steps,
            guidance_scale=0,  # No classifier-free guidance for upscaling
            of_model=self.components.of_model,
        )

        # Extract the upscaled images from the pipeline output
        upscaled_frames = upscaled_output.images
        # Each frame is wrapped in a list, extract the first element
        upscaled_frames = [frame[0] for frame in upscaled_frames]

        upscaled_width = upscaled_frames[0].size[0]
        upscaled_height = upscaled_frames[0].size[1]

        logger.info(f"Upscaled to {upscaled_width}x{upscaled_height}")

        # Step 4: Reconstruct video with audio
        output_video_path = temp_path / f"{params.job_id}_upscaled.mp4"
        self._reconstruct_video(
            frames=upscaled_frames,
            fps=fps,
            audio_path=audio_path if params.preserve_audio else None,
            output_path=output_video_path,
        )

        # Read the output video
        with open(output_video_path, "rb") as f:
            video_data = f.read()

        # Cleanup temp files
        self._cleanup_temp_files(params.job_id)

        processing_time = time.time() - start_time

        return UpscaleResult(
            video_data=video_data,
            original_width=original_width,
            original_height=original_height,
            upscaled_width=upscaled_width,
            upscaled_height=upscaled_height,
            frame_count=frame_count,
            processing_time_seconds=processing_time,
            was_upscaled=True,
        )

    def _extract_frames_and_audio(
        self,
        video_path: Path,
        job_id: str,
    ) -> tuple[list[Image.Image], float, Path | None]:
        """Extract frames and audio from a video file.

        Args:
            video_path: Path to the input video
            job_id: Job ID for temp file naming

        Returns:
            Tuple of (frames as PIL Images, fps, audio path or None)
        """
        import cv2
        from PIL import Image

        assert self._temp_dir is not None
        temp_path = Path(self._temp_dir.name)

        # Extract audio using moviepy
        audio_path: Path | None = None
        fps: float = 24.0

        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip

            clip = VideoFileClip(str(video_path))
            fps = clip.fps

            if clip.audio is not None:
                audio_path = temp_path / f"{job_id}_audio.aac"
                clip.audio.write_audiofile(
                    str(audio_path),
                    codec="aac",
                    logger=None,
                )
            clip.close()

        except Exception as e:
            logger.warning(f"Failed to extract audio: {e}")

        # Extract frames using OpenCV (faster than moviepy for frame extraction)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get FPS from OpenCV if moviepy failed
        if fps == 24.0:
            cv_fps = cap.get(cv2.CAP_PROP_FPS)
            if cv_fps > 0:
                fps = cv_fps

        frames: list[Image.Image] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()

        return frames, fps, audio_path

    def _reconstruct_video(
        self,
        frames: list[Image.Image],
        fps: float,
        audio_path: Path | None,
        output_path: Path,
    ) -> None:
        """Reconstruct a video from upscaled frames with optional audio.

        Args:
            frames: List of PIL Images (upscaled frames)
            fps: Target frames per second
            audio_path: Path to audio file or None
            output_path: Path to write the output video
        """
        import cv2
        import numpy as np

        if not frames:
            raise ValueError("No frames to reconstruct")

        # Get dimensions from first frame
        width, height = frames[0].size

        # Create temporary video without audio first
        temp_video_path = output_path.with_suffix(".temp.mp4")

        # Use H.264 codec for broad compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temp_video_path),
            fourcc,
            fps,
            (width, height),
        )

        for frame in frames:
            # Convert RGB (PIL) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

        # Add audio if present
        if audio_path is not None and audio_path.exists():
            try:
                try:
                    from moviepy import AudioFileClip, VideoFileClip
                except ImportError:
                    from moviepy.editor import AudioFileClip, VideoFileClip

                video = VideoFileClip(str(temp_video_path))
                audio = AudioFileClip(str(audio_path))

                # Use set_audio for moviepy 2.x compatibility
                if hasattr(video, "with_audio"):
                    video = video.with_audio(audio)
                else:
                    video = video.set_audio(audio)

                video.write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )

                video.close()
                audio.close()
                temp_video_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"Failed to add audio, saving video without audio: {e}")
                temp_video_path.rename(output_path)
        else:
            # No audio, just rename temp file
            temp_video_path.rename(output_path)

    async def _upscale_dry_run(
        self, params: UpscaleParams, start_time: float
    ) -> UpscaleResult:
        """Dry-run upscaling for testing without models.

        Returns the input video unchanged with simulated metadata.
        """
        # Simulate some processing time
        await asyncio.sleep(0.5)

        processing_time = time.time() - start_time

        logger.info(
            f"Dry-run upscaling completed (passthrough)",
            extra={
                "job_id": params.job_id,
                "processing_time_s": round(processing_time, 2),
            },
        )

        return UpscaleResult(
            video_data=params.video_data,  # Pass through unchanged
            original_width=1280,  # Assume 720p input for dry-run
            original_height=720,
            upscaled_width=1920,  # Assume 1080p output for dry-run
            upscaled_height=1080,
            frame_count=0,  # Unknown in dry-run
            processing_time_seconds=processing_time,
            was_upscaled=False,  # Indicate this was a passthrough
        )

    def _cleanup_temp_files(self, job_id: str) -> None:
        """Clean up temporary files for a job."""
        if self._temp_dir is None:
            return

        temp_path = Path(self._temp_dir.name)
        patterns = [
            f"{job_id}_input.mp4",
            f"{job_id}_upscaled.mp4",
            f"{job_id}_audio.aac",
        ]

        for pattern in patterns:
            path = temp_path / pattern
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the upscaler."""
        return {
            "service_type": "StreamDiffVSRUpscaler",
            "is_loaded": self.is_loaded,
            "dry_run": self.dry_run,
            "enabled": self.settings.stream_diffvsr_enabled,
            "model_id": self.settings.stream_diffvsr_model_id,
            "device": self.settings.stream_diffvsr_device,
            "num_inference_steps": self.settings.stream_diffvsr_num_inference_steps,
            "tensorrt_enabled": self.settings.stream_diffvsr_enable_tensorrt,
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
            logger.info("Stream-DiffVSR models not loaded, nothing to unload")
            return
        
        logger.info("Unloading Stream-DiffVSR models...")
        
        if self.components is not None:
            import gc
            try:
                import torch
                
                # Delete pipeline and optical flow model
                if hasattr(self.components, 'pipeline') and self.components.pipeline is not None:
                    del self.components.pipeline
                if hasattr(self.components, 'of_model') and self.components.of_model is not None:
                    del self.components.of_model
                
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
        logger.info("Stream-DiffVSR models unloaded successfully")

    def __del__(self) -> None:
        """Cleanup temporary directory on deletion."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
