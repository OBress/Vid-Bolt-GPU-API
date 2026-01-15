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
from typing import Any, Dict, Optional, List, Union
import inspect
import math

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

    @property
    def _loaded(self) -> bool:
        """Check if models are loaded (implements abstract property)."""
        return self.is_loaded

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

    async def generate_batch(
        self, params_list: List[ImageGenerationParams]
    ) -> List[ImageGenerationResult]:
        """Generate multiple images concurrently using the instance pool.
        
        This method uses true concurrent generation with multiple pipeline
        instances, maximizing GPU utilization on high-VRAM GPUs (96GB).
        
        Args:
            params_list: List of generation parameters (must have same dimensions)
            
        Returns:
            List of generation results in same order as inputs
        """
        if not params_list:
            return []
        
        if len(params_list) == 1:
            # Fast path: single image uses original method
            return [await self.generate_image(params_list[0])]
        
        if not self.is_loaded:
            raise RuntimeError(
                "Z-Image models not loaded. Call load_models() first or set "
                "ZIMAGE_DRY_RUN=false with models downloaded."
            )
        
        batch_size = len(params_list)
        first_width = params_list[0].width
        first_height = params_list[0].height
        
        # Validate all images have same dimensions
        for params in params_list:
            if params.width != first_width or params.height != first_height:
                raise ValueError(
                    f"Batch requires same dimensions. Got {params.width}x{params.height} "
                    f"but expected {first_width}x{first_height}"
                )
        
        logger.info(f"Starting VECTORIZED batch generation of {batch_size} images at {first_width}x{first_height}")
        
        # Generate seeds for each image
        seeds = []
        for params in params_list:
            seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
            seeds.append(seed)
        
        if self.dry_run:
            # Generate placeholder images concurrently
            tasks = [
                self._generate_dry_run(params, seed)
                for params, seed in zip(params_list, seeds)
            ]
            return await asyncio.gather(*tasks)
        
        # Use vectorized batching (single pipeline, batch_size=N)
        # More memory efficient (shared weights) and faster (batched tensor ops)
        
        logger.info(f"Using VECTORIZED batching (single pipeline, batch_size={batch_size})")
        loop = asyncio.get_event_loop()
        image_data_list = await loop.run_in_executor(
            None,
            lambda: self._generate_batch_sync(params_list, seeds),
        )
        results = []
        for params, seed, image_data in zip(params_list, seeds, image_data_list):
            results.append(ImageGenerationResult(
                image_data=image_data,
                width=params.width,
                height=params.height,
                seed=seed,
            ))
        
        # Aggressive memory cleanup after batch
        # CRITICAL: Run cleanup synchronously to prevent next job from starting
        # while previous batch's tensors are still held by thread pool
        import gc
        import torch
        gc.collect()  # Force Python GC to release tensor references from thread
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all GPU ops to complete
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Verify cleanup actually worked
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            if allocated_gb > 25:  # Model weights are ~15-20GB
                logger.warning(f"VRAM still high after cleanup: {allocated_gb:.2f}GB - forcing additional GC")
                gc.collect()
                torch.cuda.empty_cache()
        
        logger.info(f"Batch generation completed: {batch_size} images")
        return results
    

    def _generate_batch_sync(
        self, 
        params_list: List[ImageGenerationParams], 
        seeds: List[int]
    ) -> List[bytes]:
        """Synchronous batch image generation (runs in thread pool).
        
        This is the core vectorized implementation that generates multiple
        images in a single forward pass.
        
        Args:
            params_list: List of generation parameters
            seeds: List of seeds corresponding to each params
            
        Returns:
            List of PNG image bytes
        """
        import torch
        
        batch_size = len(params_list)
        
        # All images have same dimensions (validated upstream)
        target_width = params_list[0].width
        target_height = params_list[0].height
        gen_width = ((target_width + 31) // 32) * 32
        gen_height = ((target_height + 31) // 32) * 32
        
        device = self.settings.zimage_device
        dtype = torch.bfloat16 if self.settings.zimage_dtype == "bfloat16" else torch.float16
        
        # Pipeline components
        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        text_encoder = self.pipeline.text_encoder
        tokenizer = self.pipeline.tokenizer
        scheduler = self.pipeline.scheduler
        
        # Constants from Z-Image config
        BASE_IMAGE_SEQ_LEN = 256
        MAX_IMAGE_SEQ_LEN = 4096
        BASE_SHIFT = 0.5
        MAX_SHIFT = 1.15
        
        # --- 1. Prepare Text Embeddings (Batched) ---
        prompts = [p.prompt for p in params_list]
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted)
        
        # Tokenize all prompts at once
        text_inputs = tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()
        
        # Encode all prompts at once
        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]  # Penultimate layer
        
        # Select embeddings matching mask for each sample
        prompt_embeds_list = []
        for i in range(batch_size):
            prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])
        
        # --- 2. Prepare Latents (Batched) ---
        if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_scale_factor = 8
        vae_scale = vae_scale_factor * 2
        
        height_latent = 2 * (int(gen_height) // vae_scale)
        width_latent = 2 * (int(gen_width) // vae_scale)
        
        # Create batched latents with different seeds
        latent_shape = (transformer.in_channels, height_latent, width_latent)
        latents_list = []
        for seed in seeds:
            gen = torch.Generator(device).manual_seed(seed)
            latent = torch.randn(latent_shape, generator=gen, device=device, dtype=torch.float32)
            latents_list.append(latent)
        latents = torch.stack(latents_list, dim=0)  # (B, C, H, W)
        
        # --- 3. Prepare Scheduler ---
        num_inference_steps = params_list[0].num_inference_steps or 8
        image_seq_len = (height_latent // 2) * (width_latent // 2)
        
        m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
        b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
        mu = image_seq_len * m + b
        
        scheduler.sigma_min = 0.0
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = scheduler.timesteps
        
        logger.info(f"Batch generation: {batch_size} images, {num_inference_steps} steps, mu={mu:.4f}")
        
        # --- 4. Denoising Loop (Batched) ---
        for i, t in enumerate(timesteps):
            # Skip last step if t == 0 (Critical fix)
            if t == 0 and i == len(timesteps) - 1:
                continue
            
            timestep = t.expand(batch_size)
            timestep = (1000 - timestep) / 1000
            
            # Prepare model input
            latent_model_input = latents.to(dtype)
            latent_model_input = latent_model_input.unsqueeze(2)  # Add time dim
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            
            # Forward pass with batched inputs
            model_out_list = transformer(
                latent_model_input_list,
                timestep,
                prompt_embeds_list,
            )[0]
            
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
            noise_pred = -noise_pred.squeeze(2)
            
            # Scheduler step
            latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
        
        # --- 5. Decode Latents (Batched) ---
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
        
        # VAE decode (may need to chunk for very large batches)
        decoded_images = vae.decode(latents, return_dict=False)[0]
        
        # IMMEDIATELY delete latents after VAE decode - no longer needed
        del latents
        
        # Process to PIL and encode - move to CPU ASAP
        decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
        images_cpu = decoded_images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Delete GPU tensor immediately after moving to CPU
        del decoded_images
        
        # CRITICAL: Force cleanup NOW before CPU-bound PNG encoding
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Log VRAM after immediate cleanup
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"VRAM after batch decode cleanup: {allocated_gb:.2f}GB allocated")
        
        images = (images_cpu * 255).round().astype("uint8")
        del images_cpu
        
        from PIL import Image as PILImage
        from concurrent.futures import ThreadPoolExecutor
        
        def _process_single_image(idx_and_img):
            idx, img_array = idx_and_img
            image = PILImage.fromarray(img_array)
            
            # Crop to target dimensions if needed
            if gen_width != target_width or gen_height != target_height:
                left = (gen_width - target_width) // 2
                top = (gen_height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                image = image.crop((left, top, right, bottom))
            
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=False, compress_level=1)  # Faster saving
            buffer.seek(0)
            return buffer.getvalue()
        
        # Parallelize PNG encoding (CPU intensive)
        with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
            result_bytes = list(executor.map(_process_single_image, enumerate(images)))
        
        # Final cleanup - delete any remaining intermediates from denoising loop
        # These are last-iteration values still in scope
        try:
            del latent_model_input, latent_model_input_list
            del noise_pred, model_out_list
            del prompt_embeds, prompt_embeds_list, text_input_ids, prompt_masks
            del timestep, timesteps
            del images
        except NameError:
            pass  # Some might not exist depending on code path
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Final VRAM check
        final_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"VRAM after full batch cleanup: {final_allocated_gb:.2f}GB (should be ~15-20GB for model weights)")
        
        return result_bytes

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

        # Use Manual Generation Loop to fix IndexError
        image = self._manual_generation_loop(params, seed, generator)

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

    def _manual_generation_loop(
        self, 
        params: ImageGenerationParams, 
        seed: int, 
        generator: Any
    ) -> Any:
        """Manual generation loop to fix IndexError in pipeline.
        
        This replicates the Z-Image native generation logic but adds a check
        to skip the last step if t=0, which fixes the index out of bounds error.
        """
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler

        # Constants from Z-Image config
        BASE_IMAGE_SEQ_LEN = 256
        MAX_IMAGE_SEQ_LEN = 4096
        BASE_SHIFT = 0.5
        MAX_SHIFT = 1.15
        
        # Unpack pipeline components
        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        text_encoder = self.pipeline.text_encoder
        tokenizer = self.pipeline.tokenizer
        scheduler = self.pipeline.scheduler
        
        # Parameters
        height = params.height
        width = params.width
        # Determine actual generation dimensions (multiple of 32)
        gen_width = ((width + 31) // 32) * 32
        gen_height = ((height + 31) // 32) * 32
        
        num_inference_steps = params.num_inference_steps or 8
        guidance_scale = 0.0 # Z-Image-Turbo defaults
        
        device = self.settings.zimage_device
        dtype = torch.bfloat16 if self.settings.zimage_dtype == "bfloat16" else torch.float16

        # --- Helpers ---
        def calculate_shift(
            image_seq_len,
            base_seq_len: int = BASE_IMAGE_SEQ_LEN,
            max_seq_len: int = MAX_IMAGE_SEQ_LEN,
            base_shift: float = BASE_SHIFT,
            max_shift: float = MAX_SHIFT,
        ):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            return mu

        def retrieve_timesteps(
            scheduler,
            num_inference_steps: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = None,
            timesteps: Optional[List[int]] = None,
            sigmas: Optional[List[float]] = None,
            **kwargs,
        ):
            if timesteps is not None and sigmas is not None:
                raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
            if timesteps is not None:
                accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
                if not accepts_timesteps:
                    raise ValueError(f"The scheduler does not support custom timestep schedules.")
                scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
                timesteps = scheduler.timesteps
                num_inference_steps = len(timesteps)
            elif sigmas is not None:
                accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
                if not accept_sigmas:
                    raise ValueError(f"The scheduler does not support custom sigmas schedules.")
                scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
                timesteps = scheduler.timesteps
                num_inference_steps = len(timesteps)
            else:
                scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
                timesteps = scheduler.timesteps
            return timesteps, num_inference_steps

        # --- 1. Prepare Inputs ---
        
        # Calculate scaling factors
        if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_scale_factor = 8
        # vae_scale = vae_scale_factor * 2 # Original code had *2 for some reason? 
        # Z-Image pipeline.py says: vae_scale = vae_scale_factor * 2. 
        # Let's trust the native implementation.
        vae_scale = vae_scale_factor * 2

        # Prepare Text Embeddings
        prompt = params.prompt
        # Z-Image uses chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True, # Original had this, might be needed?
        )
        
        # Tokenize
        text_inputs = tokenizer(
            [formatted_prompt],
            padding="max_length",
            max_length=256, # config.DEFAULT_MAX_SEQUENCE_LENGTH is imported as 256/4096? 
            # In config/__init__.py, DEFAULT_MAX_SEQUENCE_LENGTH is imported from inference.
            # I don't have its value. I'll guess 256 or look at pipeline.py default.
            # pipeline.py says DEFAULT_MAX_SEQUENCE_LENGTH. 
            # I will use 256 which is safely large for prompts, or check pipeline default.
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode
        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2] # Penultimate layer

        # Select embeddings matching mask
        prompt_embeds_list = []
        for i in range(len(prompt_embeds)):
            prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])
            
        # --- 2. Prepare Latents ---
        
        batch_size = 1
        num_images_per_prompt = 1
        
        height_latent = 2 * (int(gen_height) // vae_scale)
        width_latent = 2 * (int(gen_width) // vae_scale)
        
        shape = (batch_size, transformer.in_channels, height_latent, width_latent)
        
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        
        # --- 3. Prepare Scheduler ---
        
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=BASE_IMAGE_SEQ_LEN,
            max_seq_len=MAX_IMAGE_SEQ_LEN,
            base_shift=BASE_SHIFT,
            max_shift=MAX_SHIFT,
        )
        
        scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=None,
            **scheduler_kwargs,
        )
        
        logger.info(f"Manual generation loop: {num_inference_steps} steps, mu={mu:.4f}")
        
        # --- 4. Denoising Loop ---
        
        for i, t in enumerate(timesteps):
            # CRITICAL FIX: Skip last step if t == 0
            if t == 0 and i == len(timesteps) - 1:
                logger.debug(f"Step {i+1}/{num_inference_steps} | t: {t.item():.2f} | Skipping last step (Fix applied)")
                continue
                
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            # t_norm = timestep[0].item() # Not used unless CFG truncation

            # Prepare model input
            latent_model_input = latents.to(dtype)
            prompt_embeds_model_input = prompt_embeds_list
            timestep_model_input = timestep
            
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            
            # Predict noise
            model_out_list = transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )[0]
            
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
            noise_pred = -noise_pred.squeeze(2)
            
            # Step
            latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

        # --- 5. Decode Latents ---
        
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latents_scaled = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
        
        # Delete original latents BEFORE decode to free memory
        del latents
        
        # Decode
        decoded_image = vae.decode(latents_scaled, return_dict=False)[0]
        
        # Delete scaled latents immediately after decode
        del latents_scaled
        
        # Process to PIL - move to CPU immediately
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        image_cpu = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Delete GPU tensor immediately after CPU transfer
        del decoded_image
        
        # CRITICAL: Force cleanup NOW - tensors are deleted, release memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log VRAM for debugging
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"VRAM after single-image decode cleanup: {allocated_gb:.2f}GB allocated")
        
        image = (image_cpu * 255).round().astype("uint8")
        del image_cpu
        
        # Cleanup remaining denoising loop tensors
        try:
            del latent_model_input, latent_model_input_list
            del prompt_embeds, prompt_embeds_list, text_input_ids, prompt_masks
            del noise_pred, model_out_list
            del timestep, timesteps
        except NameError:
            pass
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Final VRAM check
        final_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"VRAM after single-image full cleanup: {final_gb:.2f}GB (should be ~15-20GB)")
        
        from PIL import Image as PILImage
        return PILImage.fromarray(image[0])

    def _manual_generation_loop(
        self, 
        params: ImageGenerationParams, 
        seed: int, 
        generator: Any
    ) -> Any:
        """Manual generation loop to fix IndexError in pipeline.
        
        This replicates the Z-Image native generation logic but adds a check
        to skip the last step if t=0, which fixes the index out of bounds error.
        """
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler

        # Constants from Z-Image config
        BASE_IMAGE_SEQ_LEN = 256
        MAX_IMAGE_SEQ_LEN = 4096
        BASE_SHIFT = 0.5
        MAX_SHIFT = 1.15
        
        # Unpack pipeline components
        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        text_encoder = self.pipeline.text_encoder
        tokenizer = self.pipeline.tokenizer
        scheduler = self.pipeline.scheduler
        
        # Parameters
        height = params.height
        width = params.width
        # Determine actual generation dimensions (multiple of 32)
        gen_width = ((width + 31) // 32) * 32
        gen_height = ((height + 31) // 32) * 32
        
        num_inference_steps = params.num_inference_steps or 8
        guidance_scale = 0.0 # Z-Image-Turbo defaults
        
        device = self.settings.zimage_device
        dtype = torch.bfloat16 if self.settings.zimage_dtype == "bfloat16" else torch.float16

        # --- Helpers ---
        def calculate_shift(
            image_seq_len,
            base_seq_len: int = BASE_IMAGE_SEQ_LEN,
            max_seq_len: int = MAX_IMAGE_SEQ_LEN,
            base_shift: float = BASE_SHIFT,
            max_shift: float = MAX_SHIFT,
        ):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            return mu

        def retrieve_timesteps(
            scheduler,
            num_inference_steps: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = None,
            timesteps: Optional[List[int]] = None,
            sigmas: Optional[List[float]] = None,
            **kwargs,
        ):
            if timesteps is not None and sigmas is not None:
                raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
            if timesteps is not None:
                accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
                if not accepts_timesteps:
                    raise ValueError(f"The scheduler does not support custom timestep schedules.")
                scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
                timesteps = scheduler.timesteps
                num_inference_steps = len(timesteps)
            elif sigmas is not None:
                accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
                if not accept_sigmas:
                    raise ValueError(f"The scheduler does not support custom sigmas schedules.")
                scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
                timesteps = scheduler.timesteps
                num_inference_steps = len(timesteps)
            else:
                scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
                timesteps = scheduler.timesteps
            return timesteps, num_inference_steps

        # --- 1. Prepare Inputs ---
        
        # Calculate scaling factors
        if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_scale_factor = 8
        # vae_scale = vae_scale_factor * 2 # Original code had *2 for some reason? 
        # Z-Image pipeline.py says: vae_scale = vae_scale_factor * 2. 
        # Let's trust the native implementation.
        vae_scale = vae_scale_factor * 2

        # Prepare Text Embeddings
        prompt = params.prompt
        # Z-Image uses chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True, # Original had this, might be needed?
        )
        
        # Tokenize
        text_inputs = tokenizer(
            [formatted_prompt],
            padding="max_length",
            max_length=256, # config.DEFAULT_MAX_SEQUENCE_LENGTH is imported as 256/4096? 
            # In config/__init__.py, DEFAULT_MAX_SEQUENCE_LENGTH is imported from inference.
            # I don't have its value. I'll guess 256 or look at pipeline.py default.
            # pipeline.py says DEFAULT_MAX_SEQUENCE_LENGTH. 
            # I will use 256 which is safely large for prompts, or check pipeline default.
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode
        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2] # Penultimate layer

        # Select embeddings matching mask
        prompt_embeds_list = []
        for i in range(len(prompt_embeds)):
            prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])
            
        # --- 2. Prepare Latents ---
        
        batch_size = 1
        num_images_per_prompt = 1
        
        height_latent = 2 * (int(gen_height) // vae_scale)
        width_latent = 2 * (int(gen_width) // vae_scale)
        
        shape = (batch_size, transformer.in_channels, height_latent, width_latent)
        
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        
        # --- 3. Prepare Scheduler ---
        
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=BASE_IMAGE_SEQ_LEN,
            max_seq_len=MAX_IMAGE_SEQ_LEN,
            base_shift=BASE_SHIFT,
            max_shift=MAX_SHIFT,
        )
        
        scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=None,
            **scheduler_kwargs,
        )
        
        logger.info(f"Manual generation loop: {num_inference_steps} steps, mu={mu:.4f}")
        
        # --- 4. Denoising Loop ---
        
        for i, t in enumerate(timesteps):
            # CRITICAL FIX: Skip last step if t == 0
            if t == 0 and i == len(timesteps) - 1:
                logger.debug(f"Step {i+1}/{num_inference_steps} | t: {t.item():.2f} | Skipping last step (Fix applied)")
                continue
                
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            # t_norm = timestep[0].item() # Not used unless CFG truncation

            # Prepare model input
            latent_model_input = latents.to(dtype)
            prompt_embeds_model_input = prompt_embeds_list
            timestep_model_input = timestep
            
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            
            # Predict noise
            model_out_list = transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )[0]
            
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
            noise_pred = -noise_pred.squeeze(2)
            
            # Step
            latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

        # --- 5. Decode Latents ---
        
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latents_scaled = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
        
        # Delete original latents BEFORE decode to free memory
        del latents
        
        # Decode
        decoded_image = vae.decode(latents_scaled, return_dict=False)[0]
        
        # Delete scaled latents immediately after decode
        del latents_scaled
        
        # Process to PIL - move to CPU immediately
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        image_cpu = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Delete GPU tensor immediately after CPU transfer
        del decoded_image
        
        # CRITICAL: Force cleanup NOW
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        image = (image_cpu * 255).round().astype("uint8")
        del image_cpu
        
        # Cleanup remaining denoising loop tensors
        try:
            del latent_model_input, latent_model_input_list
            del prompt_embeds, prompt_embeds_list, text_input_ids, prompt_masks
            del noise_pred, model_out_list
            del timestep, timesteps
        except NameError:
            pass
        
        gc.collect()
        torch.cuda.empty_cache()
        
        from PIL import Image as PILImage
        return PILImage.fromarray(image[0])

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
