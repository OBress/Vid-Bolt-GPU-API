"""Z-Image Turbo Generation Service.

This module provides the ZImageGenerator service for generating images using
the Z-Image-Turbo model. It supports:
- Model loading on startup
- Async wrapper for synchronous PyTorch inference
- LoRA loading/unloading
- Dry-run mode for testing without models
"""

import asyncio
import io
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from app.config import Settings
from app.services.interfaces import ImageGenerator
from app.models.internal import ImageGenerationParams, ImageGenerationResult

logger = logging.getLogger(__name__)


@dataclass
class ZImageComponents:
    """Container for loaded Z-Image model components."""

    transformer: Any
    vae: Any
    text_encoder: Any
    tokenizer: Any
    scheduler: Any


class ZImageGenerator(ImageGenerator):
    """Z-Image Turbo image generation service.
    
    This service handles loading the Z-Image model components and generating
    images from text prompts. Models are loaded on startup when MOCK_MODE=false.
    
    Attributes:
        settings: Application settings
        components: Loaded model components (None if dry_run or not loaded)
        is_loaded: Whether models are loaded
        dry_run: Whether running in dry-run mode (workflow testing without models)
    """

    def __init__(self, settings: Settings):
        """Initialize the Z-Image generator.

        Args:
            settings: Application settings containing model paths and configuration
        """
        self.settings = settings
        self.components: Optional[ZImageComponents] = None
        self.is_loaded = False
        self.dry_run = settings.zimage_dry_run
        self._generate_func: Optional[callable] = None
        self._current_lora: Optional[str] = None
        self._lora_scale: Optional[float] = None

    def load_models(self) -> None:
        """Load Z-Image model components.
        
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
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for Z-Image generation. "
                "Install with: pip install torch"
            ) from e

        model_path = Path(self.settings.zimage_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Z-Image model not found at {model_path.absolute()}. "
                f"Download with: huggingface-cli download Tongyi-MAI/Z-Image-Turbo "
                f"--local-dir {model_path}"
            )

        # Check for essential files
        transformer_dir = model_path / "transformer"
        if not transformer_dir.exists():
            raise FileNotFoundError(
                f"Transformer weights not found at {transformer_dir}. "
                f"Model may not be fully downloaded."
            )

        logger.info(f"Loading Z-Image models from {model_path}")

        # Set up the Z-Image source path
        zimage_src_path = Path(__file__).parent.parent.parent.parent / "Z-Image" / "src"
        if zimage_src_path.exists():
            sys.path.insert(0, str(zimage_src_path))
            logger.info(f"Added Z-Image source to path: {zimage_src_path}")
        else:
            logger.warning(f"Z-Image source not found at {zimage_src_path}")

        # Import Z-Image utilities
        try:
            from utils import load_from_local_dir, set_attention_backend
            from zimage import generate
        except ImportError as e:
            raise ImportError(
                f"Failed to import Z-Image modules. Ensure Z-Image repo is cloned "
                f"at the project root level. Error: {e}"
            ) from e

        # Determine dtype
        dtype = torch.bfloat16 if self.settings.zimage_dtype == "bfloat16" else torch.float16

        # Load model components
        logger.info(f"Loading with dtype={dtype}, device={self.settings.zimage_device}")
        components_dict = load_from_local_dir(
            model_dir=str(model_path),
            device=self.settings.zimage_device,
            dtype=dtype,
            verbose=True,
            compile=self.settings.zimage_compile,
        )

        # Set attention backend
        set_attention_backend(self.settings.zimage_attention_backend)
        logger.info(f"Attention backend set to: {self.settings.zimage_attention_backend}")

        self.components = ZImageComponents(
            transformer=components_dict["transformer"],
            vae=components_dict["vae"],
            text_encoder=components_dict["text_encoder"],
            tokenizer=components_dict["tokenizer"],
            scheduler=components_dict["scheduler"],
        )
        self._generate_func = generate
        self.is_loaded = True
        logger.info("Z-Image models loaded successfully")

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

        # Generate images
        images = self._generate_func(
            prompt=params.prompt,
            transformer=self.components.transformer,
            vae=self.components.vae,
            text_encoder=self.components.text_encoder,
            tokenizer=self.components.tokenizer,
            scheduler=self.components.scheduler,
            height=gen_height,
            width=gen_width,
            num_inference_steps=params.num_inference_steps or 8,
            guidance_scale=0.0,  # Z-Image-Turbo uses 0 guidance
            generator=generator,
        )

        # Crop back to target dimensions if needed
        image = images[0]
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
        """Load a LoRA adapter by merging weights into the transformer.
        
        This implementation loads LoRA weights from a safetensors file and
        merges them into the transformer model. Z-Image uses a Flux-based
        architecture, so we look for standard LoRA key patterns.
        
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
            from safetensors.torch import load_file
            import torch
            
            # Load LoRA weights from safetensors
            lora_weights = load_file(str(lora_path))
            
            if not lora_weights:
                raise RuntimeError(f"No weights found in LoRA file: {lora_path}")
            
            logger.info(f"Loaded LoRA with {len(lora_weights)} weight tensors")
            
            # Apply LoRA weights to transformer
            applied_count = self._apply_lora_to_transformer(lora_weights, weight)
            
            if applied_count == 0:
                logger.warning(
                    f"No LoRA weights were applied. The LoRA format may be incompatible "
                    f"with Z-Image. Found keys: {list(lora_weights.keys())[:5]}..."
                )
            else:
                logger.info(f"Successfully applied LoRA to {applied_count} layer pairs")
            
            self._current_lora = lora_name
            self._lora_scale = weight
            self._lora_applied_layers: list[str] = []  # Track for potential unmerge
            
        except ImportError as e:
            raise RuntimeError(
                f"safetensors is required for LoRA loading. Install with: pip install safetensors"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load LoRA '{lora_name}': {e}")
            raise RuntimeError(f"Failed to load LoRA: {e}") from e

    def _apply_lora_to_transformer(self, lora_weights: dict, scale: float) -> int:
        """Apply LoRA weights to the transformer model.
        
        Standard LoRA format uses low-rank decomposition: W' = W + scale * (B @ A)
        Keys typically follow patterns like:
        - "transformer.blocks.0.attn.to_q.lora_A.weight"
        - "transformer.blocks.0.attn.to_q.lora_B.weight"
        
        Args:
            lora_weights: Dictionary of LoRA weight tensors
            scale: Scale factor for the LoRA weights
            
        Returns:
            Number of layer pairs successfully applied
        """
        import torch
        
        if self.components is None or self.components.transformer is None:
            raise RuntimeError("Transformer not loaded")
        
        applied_count = 0
        processed_keys: set[str] = set()
        
        # Find all lora_A keys and match with lora_B
        for key in lora_weights.keys():
            if key in processed_keys:
                continue
                
            # Handle different LoRA key patterns
            if ".lora_A" in key or ".lora_down" in key:
                # Determine matching lora_B key
                if ".lora_A" in key:
                    lora_a_key = key
                    lora_b_key = key.replace(".lora_A", ".lora_B")
                    base_key = key.replace(".lora_A.weight", "").replace(".lora_A", "")
                else:
                    lora_a_key = key
                    lora_b_key = key.replace(".lora_down", ".lora_up")
                    base_key = key.replace(".lora_down.weight", "").replace(".lora_down", "")
                
                if lora_b_key not in lora_weights:
                    logger.debug(f"Skipping {lora_a_key}: no matching B weight found")
                    continue
                
                lora_a = lora_weights[lora_a_key]
                lora_b = lora_weights[lora_b_key]
                
                # Find target layer in transformer
                target_layer = self._find_layer_by_key(base_key)
                
                if target_layer is not None and hasattr(target_layer, 'weight'):
                    try:
                        # Compute delta: B @ A (low-rank update)
                        # lora_a shape: (rank, in_features) or (in_features, rank)
                        # lora_b shape: (out_features, rank) or (rank, out_features)
                        
                        # Handle different dimension orderings
                        if lora_a.dim() == 2 and lora_b.dim() == 2:
                            if lora_b.shape[1] == lora_a.shape[0]:
                                # B @ A format
                                delta = (lora_b @ lora_a) * scale
                            elif lora_a.shape[1] == lora_b.shape[0]:
                                # A @ B format (transposed)
                                delta = (lora_a @ lora_b).T * scale
                            else:
                                logger.debug(f"Incompatible shapes for {base_key}: A={lora_a.shape}, B={lora_b.shape}")
                                continue
                            
                            # Ensure delta matches target shape
                            if delta.shape == target_layer.weight.shape:
                                target_layer.weight.data += delta.to(
                                    device=target_layer.weight.device,
                                    dtype=target_layer.weight.dtype
                                )
                                applied_count += 1
                                processed_keys.add(lora_a_key)
                                processed_keys.add(lora_b_key)
                            else:
                                logger.debug(
                                    f"Shape mismatch for {base_key}: "
                                    f"delta={delta.shape}, target={target_layer.weight.shape}"
                                )
                        else:
                            logger.debug(f"Unexpected tensor dimensions for {base_key}")
                            
                    except Exception as e:
                        logger.debug(f"Failed to apply LoRA to {base_key}: {e}")
                else:
                    logger.debug(f"Layer not found for key: {base_key}")
        
        return applied_count

    def _find_layer_by_key(self, key: str) -> Any:
        """Find a layer in the transformer by its key path.
        
        Args:
            key: Dot-separated path to the layer (e.g., "blocks.0.attn.to_q")
            
        Returns:
            The layer module if found, None otherwise
        """
        # Clean up the key - remove common prefixes
        clean_key = key
        for prefix in ["transformer.", "model.", "lora_unet_", "lora_te_"]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        
        # Also handle diffusers-style keys with underscores instead of dots
        clean_key = clean_key.replace("_", ".")
        
        try:
            obj = self.components.transformer
            for part in clean_key.split('.'):
                if part.isdigit():
                    obj = obj[int(part)]
                elif hasattr(obj, part):
                    obj = getattr(obj, part)
                elif hasattr(obj, 'named_modules'):
                    # Try to find in named modules
                    found = False
                    for name, module in obj.named_modules():
                        if name == part or name.endswith('.' + part):
                            obj = module
                            found = True
                            break
                    if not found:
                        return None
                else:
                    return None
            return obj
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    async def unload_lora(self) -> None:
        """Unload the current LoRA adapter.
        
        Since LoRA weights are merged into the model, unloading requires
        reloading the base model weights. This ensures a clean state.
        """
        if self._current_lora is None:
            return
            
        if self.dry_run:
            logger.info(f"Dry-run: Would unload LoRA '{self._current_lora}'")
            self._current_lora = None
            self._lora_scale = None
            return
        
        logger.info(f"Unloading LoRA '{self._current_lora}' by reloading base model")
        
        # Since we merged LoRA into weights, we need to reload base model
        # to get clean weights. This is the safest approach.
        old_is_loaded = self.is_loaded
        
        try:
            self.unload_models()
            self._current_lora = None
            self._lora_scale = None
            
            if old_is_loaded:
                self.load_models()
                logger.info("Base model reloaded, LoRA unloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload base model after LoRA unload: {e}")
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
        
        if self.components is not None:
            import gc
            try:
                import torch
                
                # Delete individual components
                if hasattr(self.components, 'transformer') and self.components.transformer is not None:
                    del self.components.transformer
                if hasattr(self.components, 'vae') and self.components.vae is not None:
                    del self.components.vae
                if hasattr(self.components, 'text_encoder') and self.components.text_encoder is not None:
                    del self.components.text_encoder
                if hasattr(self.components, 'tokenizer') and self.components.tokenizer is not None:
                    del self.components.tokenizer
                if hasattr(self.components, 'scheduler') and self.components.scheduler is not None:
                    del self.components.scheduler
                
                # Clear the components container
                self.components = None
                self._generate_func = None
                self._current_lora = None
                
                # Force garbage collection and clear CUDA cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except ImportError:
                # torch not available (shouldn't happen if models were loaded)
                self.components = None
                self._generate_func = None
                gc.collect()
        
        self.is_loaded = False
        logger.info("Z-Image Turbo models unloaded successfully")
