"""Z-Image Turbo Generation Service using Diffusers.

This module provides the ZImageGenerator service for generating images using
the Z-Image-Turbo model via Hugging Face Diffusers. It supports:
- Model loading on startup via ZImagePipeline
- Native Diffusers LoRA loading/unloading
- Async wrapper for synchronous PyTorch inference
- Dry-run mode for testing without models
"""

import asyncio
import io
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from app.config import Settings
from app.services.interfaces import ImageGenerator
from app.models.internal import ImageGenerationParams, ImageGenerationResult

logger = logging.getLogger(__name__)


class ZImageGenerator(ImageGenerator):
    """Z-Image Turbo image generation service using Diffusers.
    
    This service uses the official ZImagePipeline from Diffusers for cleaner
    LoRA handling and better maintainability. Models are loaded on startup
    when MOCK_MODE=false.
    
    Attributes:
        settings: Application settings
        pipeline: ZImagePipeline instance (None if dry_run or not loaded)
        is_loaded: Whether models are loaded
        dry_run: Whether running in dry-run mode (workflow testing without models)
    """

    def __init__(self, settings: Settings):
        """Initialize the Z-Image generator.

        Args:
            settings: Application settings containing model paths and configuration
        """
        self.settings = settings
        self.pipeline: Optional[Any] = None  # ZImagePipeline
        self.is_loaded = False
        self.dry_run = settings.zimage_dry_run
        self._current_lora: Optional[str] = None
        self._lora_scale: Optional[float] = None

    def load_models(self) -> None:
        """Load Z-Image model via ZImagePipeline.
        
        This should be called during application startup when MOCK_MODE=false.
        If dry_run is True, this will skip actual model loading and log the
        configuration that would be used.
        """
        if self.dry_run:
            logger.info("Z-Image dry-run mode enabled - skipping model loading")
            logger.info(f"  Model path: {self.settings.zimage_model_path}")
            logger.info(f"  LoRA path: {self.settings.zimage_lora_path}")
            logger.info(f"  Device: {self.settings.zimage_device}")
            logger.info(f"  Dtype: {self.settings.zimage_dtype}")
            logger.info(f"  Compile: {self.settings.zimage_compile}")
            logger.info(f"  Attention backend: {self.settings.zimage_attention_backend}")
            self.is_loaded = True
            return

        try:
            import torch
            from diffusers import ZImagePipeline
        except ImportError as e:
            raise ImportError(
                "PyTorch and Diffusers are required for Z-Image generation. "
                "Install with: pip install torch diffusers>=0.33.0"
            ) from e

        model_path = Path(self.settings.zimage_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Z-Image model not found at {model_path.absolute()}. "
                f"Download with: huggingface-cli download Tongyi-MAI/Z-Image-Turbo "
                f"--local-dir {model_path}"
            )

        logger.info(f"Loading Z-Image models from {model_path}")

        # Determine dtype
        dtype = torch.bfloat16 if self.settings.zimage_dtype == "bfloat16" else torch.float16

        # Load via ZImagePipeline
        logger.info(f"Loading with dtype={dtype}, device={self.settings.zimage_device}")
        self.pipeline = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.settings.zimage_device)

        # Set attention backend if specified
        # Diffusers uses SDPA by default; options: "sdpa", "flash", "_flash_3"
        if self.settings.zimage_attention_backend and self.settings.zimage_attention_backend != "sdpa":
            try:
                self.pipeline.transformer.set_attention_backend(self.settings.zimage_attention_backend)
                logger.info(f"Attention backend set to: {self.settings.zimage_attention_backend}")
            except Exception as e:
                logger.warning(f"Failed to set attention backend: {e}")

        # Optionally compile for faster inference
        if self.settings.zimage_compile:
            logger.info("Compiling transformer for faster inference...")
            self.pipeline.transformer = torch.compile(self.pipeline.transformer)

        self.is_loaded = True
        logger.info("Z-Image models loaded successfully via ZImagePipeline")

    async def generate_image(self, params: ImageGenerationParams) -> ImageGenerationResult:
        """Generate an image from a text prompt.

        Args:
            params: Generation parameters including prompt, dimensions, seed

        Returns:
            ImageGenerationResult containing the generated image data
            
        Raises:
            RuntimeError: If models are not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Z-Image models not loaded. Call load_models() first or set "
                "ZIMAGE_DRY_RUN=false with models downloaded."
            )

        # Determine seed
        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        logger.info(
            f"Generating image",
            extra={
                "job_id": params.job_id,
                "width": params.width,
                "height": params.height,
                "seed": seed,
                "lora_name": params.lora_name,
                "dry_run": self.dry_run,
            },
        )

        # Handle LoRA switching
        if params.lora_name != self._current_lora:
            if params.lora_name is None:
                await self.unload_lora()
            else:
                # If we have a different LoRA loaded, unload it first
                if self._current_lora is not None:
                    await self.unload_lora()
                await self.load_lora(params.lora_name)

        if self.dry_run:
            # Return a placeholder image for dry-run testing
            return await self._generate_dry_run(params, seed)

        # Run the synchronous generation in a thread pool
        loop = asyncio.get_event_loop()
        image_data, width, height = await loop.run_in_executor(
            None,
            lambda: self._generate_sync(params, seed),
        )

        return ImageGenerationResult(
            image_data=image_data,
            width=width,
            height=height,
            seed=seed,
        )

    def _generate_sync(self, params: ImageGenerationParams, seed: int) -> tuple[bytes, int, int]:
        """Synchronous image generation (runs in thread pool).
        
        Args:
            params: Generation parameters
            seed: Random seed for generation
            
        Returns:
            Tuple of (image_bytes, width, height)
        """
        import torch

        generator = torch.Generator(self.settings.zimage_device).manual_seed(seed)

        # Calculate dimensions (round up to nearest 32 for model compatibility)
        target_width = params.width
        target_height = params.height
        
        gen_width = ((target_width + 31) // 32) * 32
        gen_height = ((target_height + 31) // 32) * 32
        
        logger.info(
            f"Generating at padded resolution",
            extra={
                "target": f"{target_width}x{target_height}",
                "padded": f"{gen_width}x{gen_height}",
            },
        )

        # Generate image using ZImagePipeline
        output = self.pipeline(
            prompt=params.prompt,
            height=gen_height,
            width=gen_width,
            num_inference_steps=params.num_inference_steps or 8,
            guidance_scale=0.0,  # Z-Image-Turbo uses 0 guidance
            generator=generator,
        )
        image = output.images[0]

        # Crop back to target dimensions if needed
        if gen_width != target_width or gen_height != target_height:
            left = (gen_width - target_width) // 2
            top = (gen_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            image = image.crop((left, top, right, bottom))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer.getvalue(), params.width, params.height

    async def _generate_dry_run(
        self, params: ImageGenerationParams, seed: int
    ) -> ImageGenerationResult:
        """Generate a placeholder image for dry-run testing.
        
        Creates a gradient image with text overlay showing the generation parameters.
        """
        # Simulate some processing time
        await asyncio.sleep(0.5)

        # Create a gradient placeholder image
        from PIL import Image, ImageDraw, ImageFont

        width, height = params.width, params.height
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)

        # Create a gradient background
        for y in range(height):
            r = int(50 + (y / height) * 100)
            g = int(100 + (y / height) * 50)
            b = int(150 - (y / height) * 50)
            for x in range(width):
                draw.point((x, y), fill=(r, g, b))

        # Add text overlay
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        text_lines = [
            "Z-IMAGE DRY RUN",
            f"Job: {params.job_id[:8]}...",
            f"Size: {width}x{height}",
            f"Seed: {seed}",
            f"Prompt: {params.prompt[:50]}...",
        ]

        y_offset = height // 3
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font) if font else (0, 0, len(line) * 6, 10)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 25

        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return ImageGenerationResult(
            image_data=buffer.getvalue(),
            width=width,
            height=height,
            seed=seed,
        )

    async def load_lora(self, lora_name: str, weight: float = 1.0) -> None:
        """Load a LoRA adapter using Diffusers native method.
        
        Args:
            lora_name: Name of the LoRA file (without .safetensors extension)
            weight: LoRA weight multiplier (0.0 to 1.0)
            
        Raises:
            FileNotFoundError: If LoRA file doesn't exist
            RuntimeError: If LoRA loading fails
        """
        if self.dry_run:
            logger.info(f"Dry-run: Would load LoRA '{lora_name}' with weight {weight}")
            self._current_lora = lora_name
            self._lora_scale = weight
            return

        lora_path = Path(self.settings.zimage_lora_path) / f"{lora_name}.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")

        logger.info(f"Loading LoRA '{lora_name}' with weight {weight} from {lora_path}")
        
        try:
            # Use Diffusers native LoRA loading - just 1 line!
            self.pipeline.load_lora_weights(str(lora_path))
            
            # Set adapter scale if not 1.0
            if weight != 1.0:
                self.pipeline.set_adapters(["default"], adapter_weights=[weight])
            
            self._current_lora = lora_name
            self._lora_scale = weight
            logger.info(f"Successfully loaded LoRA '{lora_name}'")
            
        except Exception as e:
            logger.error(f"Failed to load LoRA '{lora_name}': {e}")
            raise RuntimeError(f"Failed to load LoRA: {e}") from e

    async def unload_lora(self) -> None:
        """Unload the current LoRA adapter using Diffusers native method.
        
        Unlike the previous implementation, this does NOT require reloading
        the entire model - Diffusers handles this cleanly.
        """
        if self._current_lora is None:
            return
            
        if self.dry_run:
            logger.info(f"Dry-run: Would unload LoRA '{self._current_lora}'")
            self._current_lora = None
            self._lora_scale = None
            return
        
        logger.info(f"Unloading LoRA '{self._current_lora}'")
        
        try:
            # Diffusers native unload - no model reload needed!
            self.pipeline.unload_lora_weights()
            self._current_lora = None
            self._lora_scale = None
            logger.info("LoRA unloaded successfully")
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the generator.
        
        Returns:
            Dictionary containing generator status information
        """
        return {
            "is_loaded": self.is_loaded,
            "dry_run": self.dry_run,
            "model_path": self.settings.zimage_model_path,
            "device": self.settings.zimage_device,
            "dtype": self.settings.zimage_dtype,
            "current_lora": self._current_lora,
            "attention_backend": self.settings.zimage_attention_backend,
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
            logger.info("Z-Image models not loaded, nothing to unload")
            return
        
        logger.info("Unloading Z-Image Turbo models...")
        
        if self.pipeline is not None:
            import gc
            try:
                import torch
                
                # Delete the pipeline
                del self.pipeline
                self.pipeline = None
                self._current_lora = None
                self._lora_scale = None
                
                # Force garbage collection and clear CUDA cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except ImportError:
                self.pipeline = None
                gc.collect()
        
        self.is_loaded = False
        logger.info("Z-Image Turbo models unloaded successfully")

    # Legacy property for backwards compatibility with old tests
    @property
    def components(self) -> Optional[Any]:
        """Legacy accessor - returns None in new implementation.
        
        The new Diffusers-based implementation uses self.pipeline instead.
        This property is kept for backwards compatibility with tests.
        """
        return None
