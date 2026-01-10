"""LightX2V Image Editing Service.

This module provides the LightX2VImageEditGenerator service for editing images
using Qwen-Image-Edit-2511 with 8-step distilled LORA. Key features:
- Pipeline initialization with LORA support
- Async wrapper for synchronous PyTorch inference  
- Dry-run mode for testing without models
- CPU offloading for low-VRAM GPUs
"""

import asyncio
import io
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from app.config import Settings
from app.services.interfaces import ImageEditor
from app.models.internal import ImageEditParams, ImageEditResult

logger = logging.getLogger(__name__)


@dataclass
class LightX2VComponents:
    """Container for LightX2V pipeline."""
    
    pipeline: Any  # LightX2VPipeline instance


class LightX2VImageEditGenerator(ImageEditor):
    """LightX2V image editing service using Qwen-Image-Edit-2511.
    
    This service handles loading the LightX2V pipeline with Qwen-Image-Edit-2511
    model and 8-step distilled LORA for fast image editing. Models are loaded
    on startup when MOCK_MODE=false.
    
    Attributes:
        settings: Application settings
        components: Loaded pipeline components (None if dry_run or not loaded)
        is_loaded: Whether models are loaded
        dry_run: Whether running in dry-run mode (workflow testing without models)
    """

    def __init__(self, settings: Settings):
        """Initialize the LightX2V generator.

        Args:
            settings: Application settings containing model paths and configuration
        """
        self.settings = settings
        self.components: Optional[LightX2VComponents] = None
        self.is_loaded = False
        self.dry_run = settings.lightx2v_dry_run
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def load_models(self) -> None:
        """Load LightX2V pipeline with LORA.
        
        This should be called during application startup when MOCK_MODE=false.
        If dry_run is True, this will skip actual model loading and log the
        configuration that would be used.
        """
        if self.dry_run:
            logger.info("LightX2V dry-run mode enabled - skipping model loading")
            logger.info(f"  Model path: {self.settings.lightx2v_model_path}")
            logger.info(f"  LORA path: {self.settings.lightx2v_lora_path}")
            logger.info(f"  LORA filename: {self.settings.lightx2v_lora_filename}")
            logger.info(f"  Device: {self.settings.lightx2v_device}")
            logger.info(f"  Attention mode: {self.settings.lightx2v_attn_mode}")
            logger.info(f"  Inference steps: {self.settings.lightx2v_infer_steps}")
            logger.info(f"  Guidance scale: {self.settings.lightx2v_guidance_scale}")
            logger.info(f"  CPU offload: {self.settings.lightx2v_cpu_offload}")
            logger.info(f"  Text encoder offload: {self.settings.lightx2v_text_encoder_offload}")
            self.is_loaded = True
            return

        # Validate model path exists
        model_path = Path(self.settings.lightx2v_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Qwen-Image-Edit-2511 model not found at {model_path.absolute()}. "
                f"Download with: huggingface-cli download Qwen/Qwen-Image-Edit-2511 "
                f"--local-dir {model_path}"
            )

        # Validate LORA path exists
        lora_path = Path(self.settings.lightx2v_lora_path) / self.settings.lightx2v_lora_filename
        if not lora_path.exists():
            raise FileNotFoundError(
                f"8-step LORA not found at {lora_path.absolute()}. "
                f"Download with: huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning "
                f"--local-dir {self.settings.lightx2v_lora_path}"
            )

        logger.info(f"Loading LightX2V pipeline from {model_path}")

        try:
            from lightx2v import LightX2VPipeline
        except ImportError as e:
            raise ImportError(
                "LightX2V is required for Qwen-Image-Edit-2511 generation. "
                "Install with: pip install -v git+https://github.com/ModelTC/LightX2V.git"
            ) from e

        # Initialize pipeline for image-to-image editing
        pipe = LightX2VPipeline(
            model_path=str(model_path.absolute()),
            model_cls="qwen-image-edit-2511",
            task="i2i",
        )

        # Enable CPU offloading if configured
        if self.settings.lightx2v_cpu_offload or self.settings.lightx2v_text_encoder_offload:
            logger.info("Enabling CPU offloading for lower VRAM usage")
            pipe.enable_offload(
                cpu_offload=self.settings.lightx2v_cpu_offload,
                offload_granularity="block",
                text_encoder_offload=self.settings.lightx2v_text_encoder_offload,
                vae_offload=False,
            )

        # Enable 8-step distilled LORA
        logger.info(f"Loading 8-step distilled LORA from {lora_path}")
        pipe.enable_lora([
            {
                "path": str(lora_path.absolute()),
                "strength": 1.0
            }
        ])

        # Create generator with specified parameters
        pipe.create_generator(
            attn_mode=self.settings.lightx2v_attn_mode,
            resize_mode=self.settings.lightx2v_resize_mode,
            infer_steps=self.settings.lightx2v_infer_steps,
            guidance_scale=self.settings.lightx2v_guidance_scale,
        )

        self.components = LightX2VComponents(pipeline=pipe)
        self.is_loaded = True
        
        # Create temp directory for intermediate files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="lightx2v_")
        
        logger.info("LightX2V pipeline loaded successfully")



    async def edit_image(self, params: ImageEditParams) -> ImageEditResult:
        """Edit an image using the LightX2V pipeline.

        Args:
            params: Edit parameters including input image, prompt, dimensions

        Returns:
            ImageEditResult containing the edited image data
            
        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(
                "LightX2V models not loaded. Call load_models() first or set "
                "LIGHTX2V_DRY_RUN=false with models downloaded."
            )

        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Editing image with LightX2V",
            extra={
                "job_id": params.job_id,
                "width": params.width,
                "height": params.height,
                "seed": seed,
                "dry_run": self.dry_run,
            },
        )

        if self.dry_run:
            # Return a placeholder image for dry-run testing
            return await self._edit_dry_run(params, seed)

        # Run the synchronous generation in a thread pool
        loop = asyncio.get_event_loop()
        image_data, orig_w, orig_h, out_w, out_h = await loop.run_in_executor(
            None,
            lambda: self._edit_sync(params, seed),
        )

        return ImageEditResult(
            image_data=image_data,
            original_width=orig_w,
            original_height=orig_h,
            width=out_w,
            height=out_h,
            seed=seed,
        )

    def _edit_sync(
        self, params: ImageEditParams, seed: int
    ) -> tuple[bytes, int, int, int, int]:
        """Synchronous image editing (runs in thread pool).
        
        Args:
            params: Edit parameters
            seed: Random seed for generation
            
        Returns:
            Tuple of (image_bytes, orig_width, orig_height, output_width, output_height)
        """
        # Get original image dimensions
        input_image = Image.open(io.BytesIO(params.input_image_data))
        orig_width, orig_height = input_image.size
        
        # Cap dimensions at 2048x2048 (preserve aspect ratio)
        max_dim = 2048
        target_width, target_height = orig_width, orig_height
        if orig_width > max_dim or orig_height > max_dim:
            scale = min(max_dim / orig_width, max_dim / orig_height)
            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)
            logger.info(f"Capping resolution from {orig_width}x{orig_height} to {target_width}x{target_height}")

        # Save input image to temp file (LightX2V expects file paths)
        assert self._temp_dir is not None
        input_path = Path(self._temp_dir.name) / f"{params.job_id}_input.png"
        output_path = Path(self._temp_dir.name) / f"{params.job_id}_output.png"
        
        input_image.save(input_path, format="PNG")

        # Run inference
        self.components.pipeline.generate(
            seed=seed,
            image_path=str(input_path),
            prompt=params.prompt,
            negative_prompt="",
            save_result_path=str(output_path),
        )

        # Load output image
        output_image = Image.open(output_path)
        model_out_w, model_out_h = output_image.size
        
        # Resize output to match target dimensions (original or capped)
        if output_image.size != (target_width, target_height):
            logger.info(f"Resizing output from {model_out_w}x{model_out_h} to {target_width}x{target_height}")
            output_image = output_image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Clean up temp files
        try:
            input_path.unlink()
            output_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")

        return buffer.getvalue(), orig_width, orig_height, target_width, target_height

    async def _edit_dry_run(
        self, params: ImageEditParams, seed: int
    ) -> ImageEditResult:
        """Generate a placeholder image for dry-run testing.
        
        Creates a gradient image with text overlay showing the edit parameters.
        """
        # Simulate some processing time
        await asyncio.sleep(1.0)

        # Get original image dimensions
        input_image = Image.open(io.BytesIO(params.input_image_data))
        orig_width, orig_height = input_image.size

        # Create a placeholder overlay on the input image
        from PIL import ImageDraw, ImageFont

        # Create output image (copy of input with overlay)
        output_image = input_image.copy().convert("RGBA")
        overlay = Image.new("RGBA", output_image.size, (0, 0, 0, 128))
        output_image = Image.alpha_composite(output_image, overlay)
        output_image = output_image.convert("RGB")
        
        draw = ImageDraw.Draw(output_image)

        # Add text overlay
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        text_lines = [
            "LIGHTX2V DRY RUN",
            f"Job: {params.job_id[:8]}...",
            f"Seed: {seed}",
            f"Steps: {self.settings.lightx2v_infer_steps}",
            f"Prompt: {params.prompt[:40]}...",
        ]

        y_offset = orig_height // 3
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font) if font else (0, 0, len(line) * 6, 10)
            text_width = bbox[2] - bbox[0]
            x = (orig_width - text_width) // 2
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 25

        # Convert to bytes
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)

        return ImageEditResult(
            image_data=buffer.getvalue(),
            original_width=orig_width,
            original_height=orig_height,
            width=orig_width,
            height=orig_height,
            seed=seed,
        )



    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the generator.
        
        Returns:
            Dictionary containing generator status information
        """
        return {
            "generator_type": "LightX2VImageEditGenerator",
            "model": "Qwen-Image-Edit-2511",
            "is_loaded": self.is_loaded,
            "dry_run": self.dry_run,
            "model_path": self.settings.lightx2v_model_path,
            "lora_path": self.settings.lightx2v_lora_path,
            "lora_filename": self.settings.lightx2v_lora_filename,
            "device": self.settings.lightx2v_device,
            "attn_mode": self.settings.lightx2v_attn_mode,
            "infer_steps": self.settings.lightx2v_infer_steps,
            "guidance_scale": self.settings.lightx2v_guidance_scale,
            "cpu_offload": self.settings.lightx2v_cpu_offload,
            "text_encoder_offload": self.settings.lightx2v_text_encoder_offload,
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
            logger.info("LightX2V models not loaded, nothing to unload")
            return
        
        logger.info("Unloading LightX2V (Qwen-Image-Edit) models...")
        
        if self.components is not None:
            import gc
            try:
                import torch
                
                # Delete the pipeline
                if hasattr(self.components, 'pipeline') and self.components.pipeline is not None:
                    del self.components.pipeline
                
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
        logger.info("LightX2V models unloaded successfully")

    def __del__(self):
        """Cleanup temporary directory on deletion."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
