"""LightX2V Image Editing Service.

This module provides the LightX2VImageEditGenerator service for editing images
using Qwen-Image-Edit-2511 with 8-step distilled LORA. Key features:
- Instance pool for concurrent image editing
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
from typing import Any, Dict, List, Optional

from PIL import Image

from app.config import Settings
from app.services.interfaces import ImageEditor
from app.models.internal import ImageEditParams, ImageEditResult
from app.services.lightx2v_pool import LightX2VInstancePool, PooledInstance

logger = logging.getLogger(__name__)




class LightX2VImageEditGenerator(ImageEditor):
    """LightX2V image editing service using Qwen-Image-Edit-2511.
    
    This service handles loading the LightX2V pipeline with Qwen-Image-Edit-2511
    model and 8-step distilled LORA for fast image editing. Uses an instance
    pool for concurrent processing when batch editing.
    
    Attributes:
        settings: Application settings
        _pool: Instance pool for concurrent processing
        is_loaded: Whether models are loaded
        dry_run: Whether running in dry-run mode (workflow testing without models)
    """

    def __init__(self, settings: Settings, max_instances: int = 2):
        """Initialize the LightX2V generator.

        Args:
            settings: Application settings containing model paths and configuration
            max_instances: Maximum concurrent instances in the pool (mode-dependent)
        """
        self.settings = settings
        self._max_instances = max_instances
        self._pool: Optional[LightX2VInstancePool] = None
        self.is_loaded = False
        self.dry_run = settings.lightx2v_dry_run
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def load_models(self) -> None:
        """Load LightX2V pipeline(s) with LORA.
        
        This loads an instance pool for concurrent batch processing.
        If dry_run is True, this will skip actual model loading.
        """
        if self.dry_run:
            logger.info("LightX2V dry-run mode enabled - skipping model loading")
            logger.info(f"  Model path: {self.settings.lightx2v_model_path}")
            logger.info(f"  LORA path: {self.settings.lightx2v_lora_path}")
            logger.info(f"  Max instances: {self._max_instances}")
            logger.info(f"  Inference steps: {self.settings.lightx2v_infer_steps}")
            
            # Create dry-run pool
            self._pool = LightX2VInstancePool(
                settings=self.settings,
                max_instances=self._max_instances,
                dry_run=True
            )
            self._pool.load_all()
            self._temp_dir = tempfile.TemporaryDirectory(prefix="lightx2v_")
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

        # Create and load instance pool for concurrent processing
        logger.info(
            f"Loading LightX2V with {self._max_instances} concurrent instances "
            f"from {model_path}"
        )
        
        self._pool = LightX2VInstancePool(
            settings=self.settings,
            max_instances=self._max_instances,
            dry_run=False
        )
        self._pool.load_all()
        
        # Create temp directory for intermediate files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="lightx2v_")
        self.is_loaded = True
        
        logger.info(
            f"LightX2V pool loaded: {self._pool.size} instances ready "
            f"for concurrent processing"
        )



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

        # Acquire instance from pool (blocks until one is available)
        instance = await self._pool.acquire()
        try:
            # Run the synchronous generation in a thread pool using the acquired instance
            loop = asyncio.get_event_loop()
            image_data, orig_w, orig_h, out_w, out_h = await loop.run_in_executor(
                None,
                lambda: self._edit_with_instance(instance, params, seed),
            )

            return ImageEditResult(
                image_data=image_data,
                original_width=orig_w,
                original_height=orig_h,
                width=out_w,
                height=out_h,
                seed=seed,
            )
        finally:
            # Always release the instance back to the pool
            await self._pool.release(instance)

    async def edit_batch(
        self, 
        params_list: List[ImageEditParams]
    ) -> List[ImageEditResult]:
        """Edit multiple images concurrently using the instance pool.
        
        This method processes images in parallel using the instance pool.
        Concurrency is limited by the pool size (e.g., 4 instances = 4 concurrent).
        
        Args:
            params_list: List of edit parameters.
            
        Returns:
            List of edit results in same order as inputs.
            Failed jobs will still return results but with empty image_data.
            
        Raises:
            RuntimeError: If models are not loaded
        """
        # Early validation
        if not params_list:
            return []
        
        if len(params_list) == 1:
            # Fast path: single image uses standard method
            return [await self.edit_image(params_list[0])]
        
        if not self.is_loaded:
            raise RuntimeError(
                "LightX2V models not loaded. Call load_models() first or set "
                "LIGHTX2V_DRY_RUN=false with models downloaded."
            )

        # Validate that all images have the same dimensions
        first_w, first_h = params_list[0].width, params_list[0].height
        for i, params in enumerate(params_list[1:], start=1):
            if params.width != first_w or params.height != first_h:
                raise ValueError(
                    f"Batch items must have same dimensions. Item {i} has "
                    f"{params.width}x{params.height}, expected {first_w}x{first_h}"
                )
        
        batch_size = len(params_list)
        pool_size = self._pool.size if self._pool else 1
        logger.info(
            f"Starting LightX2V concurrent batch: {batch_size} images "
            f"with {pool_size} concurrent instances"
        )
        
        async def process_one(params: ImageEditParams, idx: int) -> ImageEditResult:
            """Process a single image using a pooled instance."""
            seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
            
            try:
                logger.debug(
                    f"Processing batch item {idx + 1}/{batch_size}: "
                    f"job_id={params.job_id}, seed={seed}"
                )
                
                if self.dry_run:
                    return await self._edit_dry_run(params, seed)
                
                # Acquire instance from pool (blocks until one is available)
                instance = await self._pool.acquire()
                try:
                    # Run synchronous generation in thread pool with this instance
                    loop = asyncio.get_event_loop()
                    image_data, orig_w, orig_h, out_w, out_h = await loop.run_in_executor(
                        None,
                        lambda: self._edit_with_instance(instance, params, seed),
                    )
                    
                    return ImageEditResult(
                        image_data=image_data,
                        original_width=orig_w,
                        original_height=orig_h,
                        width=out_w,
                        height=out_h,
                        seed=seed,
                    )
                finally:
                    # Always release the instance back to the pool
                    await self._pool.release(instance)
                    
            except Exception as e:
                logger.error(
                    f"Batch item {idx + 1}/{batch_size} failed "
                    f"(job_id={params.job_id}): {e}"
                )
                # Return placeholder result
                return ImageEditResult(
                    image_data=b"",  # Empty data indicates failure
                    original_width=params.width,
                    original_height=params.height,
                    width=params.width,
                    height=params.height,
                    seed=seed,
                )
        
        # Process all images concurrently - pool semaphore limits parallelism
        results = await asyncio.gather(*[
            process_one(params, idx)
            for idx, params in enumerate(params_list)
        ])
        
        # Count successes/failures
        successful = sum(1 for r in results if r.image_data)
        failed = len(results) - successful
        
        logger.info(
            f"LightX2V batch complete: {successful} succeeded, {failed} failed "
            f"out of {batch_size} total"
        )
        
        # If all jobs failed, raise an exception
        if successful == 0 and failed > 0:
            raise RuntimeError(f"All {failed} jobs in batch failed")
        
        return list(results)


    def _edit_with_instance(
        self, 
        instance: PooledInstance, 
        params: ImageEditParams, 
        seed: int
    ) -> tuple[bytes, int, int, int, int]:
        """Run image editing using a specific pooled pipeline instance.
        
        This is similar to _edit_sync but uses the pipeline from a pool instance
        instead of self.components.
        
        Args:
            instance: Pooled pipeline instance to use
            params: Edit parameters
            seed: Random seed for generation
            
        Returns:
            Tuple of (image_bytes, orig_width, orig_height, output_width, output_height)
        """
        # Get original image dimensions
        input_image = Image.open(io.BytesIO(params.input_image_data))
        orig_width, orig_height = input_image.size
        
        # Cap dimensions at 4096x4096 (preserve aspect ratio)
        max_dim = 4096
        target_width, target_height = orig_width, orig_height
        if orig_width > max_dim or orig_height > max_dim:
            scale = min(max_dim / orig_width, max_dim / orig_height)
            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)
            logger.info(f"Capping resolution from {orig_width}x{orig_height} to {target_width}x{target_height}")

        # Save input image to temp file in instance's temp dir
        input_path = Path(instance.temp_dir.name) / f"{params.job_id}_input.png"
        output_path = Path(instance.temp_dir.name) / f"{params.job_id}_output.png"
        
        input_image.save(input_path, format="PNG")

        # Run inference using this instance's pipeline
        # Model will generate at native resolution (or nearest 32px multiple)
        import torch
        with torch.inference_mode():
            instance.pipeline.generate(
                seed=seed,
                image_path=str(input_path),
                prompt=params.prompt,
                negative_prompt="",
                save_result_path=str(output_path),
            )

        # Load output image
        output_image = Image.open(output_path)
        model_out_w, model_out_h = output_image.size
        
        # Center-crop output to exact target dimensions if needed (like z-image)
        # This handles any 32px alignment padding the model may have added
        if model_out_w != target_width or model_out_h != target_height:
            logger.info(f"Center-cropping output from {model_out_w}x{model_out_h} to {target_width}x{target_height}")
            # Calculate crop box (center-aligned)
            left = (model_out_w - target_width) // 2
            top = (model_out_h - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            output_image = output_image.crop((left, top, right, bottom))

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
        
        # Unload the instance pool
        if self._pool is not None:
            self._pool.unload_all()
            self._pool = None
        
        self.is_loaded = False
        logger.info("LightX2V models unloaded successfully")

    def __del__(self):
        """Cleanup temporary directory on deletion."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
