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
import threading
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
from app.services.ltx2_concurrent import LTX2ConcurrencyController
from app.services import vram_estimator

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

    Uses two pipelines for optimal quality and speed:
    - DistilledPipeline: For I2V with single keyframe (fast 8+4 step schedule)
    - KeyframeInterpolationPipeline: For 2-keyframe requests using guiding latents
      for smoother transitions (uses 8-step distilled schedule for speed)
    
    Key features:
    - Automatic pipeline selection based on keyframe count
    - ~40GB VRAM usage (both pipelines share model weights)
    - Synchronized audio generation alongside video
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
        
        # Concurrent generation support
        self._concurrent_controller: LTX2ConcurrencyController | None = None
        self._concurrent_enabled = getattr(settings, 'ltx2_concurrent_enabled', True)
        
        # Thread lock to serialize text encoder access during concurrent generation.
        # The Gemma LLM has internal mutable state (KV caches) that is not thread-safe.
        self._text_encode_lock = threading.Lock()

    @property
    def _loaded(self) -> bool:
        """Compatibility property for model_manager which expects _loaded."""
        return self.is_loaded

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
                f"LTX-2.3 checkpoint not found at {checkpoint_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2.3-fp8 "
                f"ltx-2.3-22b-dev-fp8.safetensors --local-dir {checkpoint_path.parent}"
            )

        spatial_upsampler_path = Path(self.settings.ltx2_spatial_upsampler_path)
        if not spatial_upsampler_path.exists():
            raise FileNotFoundError(
                f"LTX-2.3 spatial upsampler not found at {spatial_upsampler_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2.3 "
                f"ltx-2.3-spatial-upscaler-x2-1.0.safetensors --local-dir {spatial_upsampler_path.parent}"
            )

        gemma_root = Path(self.settings.ltx2_gemma_root)
        if not gemma_root.exists():
            raise FileNotFoundError(
                f"Gemma text encoder not found at {gemma_root.absolute()}. "
                f"Download with: huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized "
                f"--local-dir {gemma_root}"
            )

        distilled_lora_path = Path(self.settings.ltx2_distilled_lora_path)
        if not distilled_lora_path.exists():
            raise FileNotFoundError(
                f"LTX-2.3 distilled LoRA not found at {distilled_lora_path.absolute()}. "
                f"Download with: huggingface-cli download Lightricks/LTX-2.3 "
                f"ltx-2.3-22b-distilled-lora-384.safetensors --local-dir {distilled_lora_path.parent}"
            )

        logger.info(f"Loading LTX-2.3 pipelines from {checkpoint_path}")

        try:
            import torch
            from ltx_pipelines.distilled import DistilledPipeline
            from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
            from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
            from ltx_core.quantization import QuantizationPolicy
            from ltx_core.quantization.fp8_cast import UPCAST_DURING_INFERENCE
            from ltx_core.loader.sd_ops import SDOps
        except ImportError as e:
            raise ImportError(
                "LTX-2 packages are required. Install with: "
                "pip install -e path/to/LTX-2/packages/ltx-pipelines"
            ) from e

        # Initialize device for all pipelines
        device = torch.device(self.settings.ltx2_device)

        # Prepare distilled LoRA for the dev model
        # This converts the dev model into distilled-equivalent at inference time
        distilled_lora = LoraPathStrengthAndSDOps(
            path=str(distilled_lora_path.absolute()),
            strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
        logger.info(f"Distilled LoRA: {distilled_lora_path.name} (strength=1.0)")

        # FP8 quantization for pre-quantized checkpoint + LoRA fusion:
        # 1. SD ops: dequantize FP8→bf16 using per-tensor scales at load time
        # 2. LoRA fusion runs in bf16 (standard path, correct shapes)
        # 3. Module ops: downcast fused bf16 weights back to FP8, patch forward for upcast
        # Result: ~19GB VRAM with correct values

        from ltx_core.loader.sd_ops import SDOps, KeyValueOperationResult
        from ltx_core.loader.module_ops import ModuleOps
        from ltx_core.quantization.fp8_cast import _replace_fwd_with_upcast
        from ltx_core.model.transformer.model import LTXModel

        # Mutable closure to handle key ordering in safetensors
        # (.weight sorts before .weight_scale, so weight arrives first)
        _pending_fp8 = {}

        def _dequant_fp8_with_scale(key: str, value: torch.Tensor) -> list:
            if key.endswith(".weight_scale"):
                # Scale arrived — check if weight is already pending
                weight_key = key[:-len("_scale")]  # "x.weight_scale" → "x.weight"
                if weight_key in _pending_fp8:
                    fp8_w = _pending_fp8.pop(weight_key)
                    dequant = fp8_w.to(torch.bfloat16) * value
                    return [KeyValueOperationResult(weight_key, dequant)]
                else:
                    _pending_fp8[key] = value  # Store scale for later
                    return []

            if key.endswith(".input_scale"):
                return []  # Drop input_scale keys entirely

            if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn:
                # FP8 weight — check if scale already arrived
                scale_key = key + "_scale"
                if scale_key in _pending_fp8:
                    scale = _pending_fp8.pop(scale_key)
                    dequant = value.to(torch.bfloat16) * scale
                    return [KeyValueOperationResult(key, dequant)]
                else:
                    _pending_fp8[key] = value  # Store weight, wait for scale
                    return []

            return [KeyValueOperationResult(key, value)]

        fp8_dequant_sd_ops = SDOps("fp8_dequant_to_bf16").with_kv_operation(
            _dequant_fp8_with_scale, key_prefix="", key_suffix=""
        )

        # Module ops: after load_state_dict, downcast bf16→FP8 and patch forward
        def _downcast_and_patch_forward(model: torch.nn.Module) -> torch.nn.Module:
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16:
                    m.weight = torch.nn.Parameter(
                        m.weight.data.to(torch.float8_e4m3fn),
                        requires_grad=False,
                    )
                    _replace_fwd_with_upcast(m, with_stochastic_rounding=True)
            return model

        FP8_DOWNCAST_UPCAST = ModuleOps(
            name="fp8_downcast_and_upcast_inference",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=_downcast_and_patch_forward,
        )

        fp8_policy = QuantizationPolicy(
            sd_ops=fp8_dequant_sd_ops,
            module_ops=(FP8_DOWNCAST_UPCAST,),
        )
        logger.info("Using FP8 quantization (dequant→fuse→recast, ~19GB VRAM)")

        # DistilledPipeline for I2V generation (1 keyframe)
        # Uses latent replacement conditioning - exact keyframe preservation
        logger.info("Loading DistilledPipeline for single-keyframe I2V generation...")
        try:
            distilled_pipeline = DistilledPipeline(
                distilled_checkpoint_path=str(checkpoint_path.absolute()),
                spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
                gemma_root=str(gemma_root.absolute()),
                loras=[distilled_lora],  # Apply distilled LoRA to dev model
                device=device,
                quantization=fp8_policy,
            )
        except Exception:
            logger.exception("Failed to initialize DistilledPipeline")
            raise
        logger.info("DistilledPipeline loaded successfully")
        
        # KeyframeInterpolationPipeline for 2-keyframe requests
        # Uses guiding latents conditioning - smoother transitions between keyframes
        # The new API shares models automatically via ModelLedger registry
        logger.info("Initializing KeyframeInterpolationPipeline...")
        try:
            keyframe_pipeline = KeyframeInterpolationPipeline(
                checkpoint_path=str(checkpoint_path.absolute()),
                distilled_lora=[distilled_lora],  # Apply distilled LoRA to dev model
                spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
                gemma_root=str(gemma_root.absolute()),
                loras=[],  # No extra LoRAs beyond distilled
                device=device,
                quantization=fp8_policy,
            )
        except Exception:
            logger.exception("Failed to initialize KeyframeInterpolationPipeline")
            raise
        logger.info("KeyframeInterpolationPipeline initialized")

        self.components = LTX2Components(
            distilled_pipeline=distilled_pipeline,
            keyframe_pipeline=keyframe_pipeline,
        )
        self.is_loaded = True

        # Create temp directory for intermediate files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ltx2_")

        logger.info("LTX-2 pipelines loaded, running warmup...")
        
        # Warmup: Run minimal inference to force GPU tensor transfer and kernel JIT
        self._run_warmup(device)
        
        logger.info("LTX-2 pipelines loaded and warmed up successfully")

    def _run_warmup(self, device: "torch.device") -> None:
        """Force models to materialize on GPU and patch model_ledger for caching.
        
        The LTX-2 model_ledger creates NEW model instances on each call to
        text_encoder(), transformer(), etc. This causes race conditions when
        multiple threads try to lazy-load models simultaneously.
        
        This warmup:
        1. Triggers the first load of each model (serialized, no race)
        2. Patches the model_ledger to cache and reuse these loaded instances
        
        Args:
            device: The target GPU device
        """
        import torch
        from PIL import Image
        from ltx_core.model.video_vae import TilingConfig
        
        logger.info("Running LTX-2 warmup inference...")
        
        try:
            # Create minimal 64x64 black image (smallest valid size for LTX-2 is 64)
            warmup_image = Image.new('RGB', (256, 256), color='black')
            warmup_path = Path(self._temp_dir.name) / "warmup_frame.png"
            warmup_image.save(warmup_path, format="PNG")
            
            # Run warmup with DistilledPipeline
            # This forces all model weights to GPU and compiles CUDA kernels
            with torch.no_grad():
                tiling_config = TilingConfig.default()
                
                # Warmup DistilledPipeline
                from ltx_pipelines.utils.args import ImageConditioningInput
                video_chunks, audio = self.components.distilled_pipeline(
                    prompt="warmup",
                    seed=42,
                    height=256,  # Small but divisible by 64
                    width=256,
                    num_frames=9,  # Minimum valid: 1 + 8*1 = 9
                    frame_rate=24.0,
                    images=[ImageConditioningInput(path=str(warmup_path), frame_idx=0, strength=1.0, crf=0)],  # crf=0: skip lossy compression
                    tiling_config=tiling_config,
                    enhance_prompt=False,
                )
                
                # Consume the generator to ensure all ops run
                if hasattr(video_chunks, '__iter__') and not isinstance(video_chunks, torch.Tensor):
                    for _ in video_chunks:
                        pass
            
            # Cleanup warmup file
            warmup_path.unlink(missing_ok=True)
            
            # CRITICAL: Patch the model_ledger to cache text encoders
            # After warmup, the text_encoder has been loaded once. We now patch
            # the model_ledger.text_encoder() method to return a cached instance
            # instead of creating new ones (which causes race conditions).
            self._patch_model_ledger_caching()
            
            # Clear CUDA cache to free warmup memory (not model weights)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("LTX-2 warmup complete - GPU tensors loaded and kernels compiled")
            
        except Exception as e:
            logger.warning(f"LTX-2 warmup failed (non-fatal): {e}", exc_info=True)
            # Warmup failure is non-fatal - first real inference will just be slow

    def _patch_model_ledger_caching(self) -> None:
        """Patch model_ledger to cache and reuse ALL model instances.
        
        The upstream LTX-2 model_ledger creates NEW model instances on each call
        (see model_ledger.py docstring: "Models are not cached"). This means every
        pipeline.__call__ loads a fresh transformer (~19GB), video_encoder, etc.
        
        With concurrent generation, N calls = N copies of the transformer in VRAM.
        Even with sequential calls, the old copy must be freed before a new one loads.
        
        We build each model ONCE during warmup and patch model_ledger to return
        the cached instances, so all calls (single or concurrent) share one set of
        weights in VRAM.
        """
        distilled = self.components.distilled_pipeline
        ledger = distilled.model_ledger
        
        logger.info("Caching all model_ledger models for concurrent access...")
        
        try:
            # Build ALL heavy models once and cache them
            cached_text_encoder = ledger.text_encoder()
            cached_embeddings_processor = ledger.gemma_embeddings_processor()
            cached_transformer = ledger.transformer()
            cached_video_encoder = ledger.video_encoder()
            cached_video_decoder = ledger.video_decoder()
            cached_spatial_upsampler = ledger.spatial_upsampler()
            cached_audio_decoder = ledger.audio_decoder()
            cached_vocoder = ledger.vocoder()
            
            logger.info(
                "  Built and cached: text_encoder, embeddings_processor, "
                "transformer, video_encoder, video_decoder, spatial_upsampler, "
                "audio_decoder, vocoder"
            )
            
            # Create cached factory functions
            def _text_encoder(): return cached_text_encoder
            def _embeddings_processor(): return cached_embeddings_processor
            def _transformer(): return cached_transformer
            def _video_encoder(): return cached_video_encoder
            def _video_decoder(): return cached_video_decoder
            def _spatial_upsampler(): return cached_spatial_upsampler
            def _audio_decoder(): return cached_audio_decoder
            def _vocoder(): return cached_vocoder
            
            # Patch distilled pipeline's model_ledger
            ledger.text_encoder = _text_encoder
            ledger.gemma_embeddings_processor = _embeddings_processor
            ledger.transformer = _transformer
            ledger.video_encoder = _video_encoder
            ledger.video_decoder = _video_decoder
            ledger.spatial_upsampler = _spatial_upsampler
            ledger.audio_decoder = _audio_decoder
            ledger.vocoder = _vocoder
            
            # Patch keyframe pipeline's model_ledgers too
            keyframe = self.components.keyframe_pipeline
            if hasattr(keyframe, 'stage_1_model_ledger'):
                for kf_ledger in [keyframe.stage_1_model_ledger]:
                    kf_ledger.text_encoder = _text_encoder
                    kf_ledger.gemma_embeddings_processor = _embeddings_processor
                    kf_ledger.transformer = _transformer
                    kf_ledger.video_encoder = _video_encoder
                    kf_ledger.video_decoder = _video_decoder
                    kf_ledger.spatial_upsampler = _spatial_upsampler
                    kf_ledger.audio_decoder = _audio_decoder
                    kf_ledger.vocoder = _vocoder
                if hasattr(keyframe, 'stage_2_model_ledger'):
                    kf_ledger2 = keyframe.stage_2_model_ledger
                    kf_ledger2.text_encoder = _text_encoder
                    kf_ledger2.gemma_embeddings_processor = _embeddings_processor
                    kf_ledger2.transformer = _transformer
                    kf_ledger2.video_encoder = _video_encoder
                    kf_ledger2.video_decoder = _video_decoder
                    kf_ledger2.spatial_upsampler = _spatial_upsampler
                    kf_ledger2.audio_decoder = _audio_decoder
                    kf_ledger2.vocoder = _vocoder
                logger.info("  Patched DistilledPipeline + KeyframeInterpolationPipeline model_ledgers")
            else:
                logger.info("  Patched DistilledPipeline model_ledger")
            
            logger.info("All models cached — concurrent calls now share VRAM weights")
            
        except Exception as e:
            logger.warning(f"Failed to patch model_ledger caching (non-fatal): {e}")
            import traceback
            traceback.print_exc()



    # ========================================================================
    # I2V (Image-to-Video) Generation
    # ========================================================================

    async def generate_video(
        self, 
        params: LTX2VideoParams, 
        pre_encoded_context: Any | None = None,
    ) -> LTX2VideoResult:
        """Generate a video from a single start frame (I2V mode).

        This is a convenience method that converts I2V parameters to keyframe
        format internally (start frame at index 0, optional end frame at last index).

        Args:
            params: I2V generation parameters
            pre_encoded_context: Optional pre-encoded prompt embeddings (for concurrent batch)

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
            (params.start_frame_data, 0, 1.0)
        ]
        
        # Add end frame if provided
        if params.end_frame_data is not None:
            # KeyframeInterpolationPipeline uses pixel indices (not latent indices)
            # The end frame should be at the last generated frame
            pixel_idx = num_frames - 1
            keyframes.append((params.end_frame_data, pixel_idx, 1.0))

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
        return await self.generate_keyframe_video(keyframe_params, pre_encoded_context=pre_encoded_context)

    # ========================================================================
    # Batch Video Generation (Sequential Warm-Model)
    # ========================================================================

    async def generate_batch(
        self, 
        params_list: list[LTX2VideoParams]
    ) -> list[LTX2VideoResult]:
        """Generate multiple videos sequentially with warm model.
        
        This method processes videos one-at-a-time but keeps the model loaded
        between generations, eliminating the ~30s model load time per video.
        This provides significant throughput improvement for batch workloads.
        
        Args:
            params_list: List of video generation parameters
            
        Returns:
            List of VideoGenerationResults in the same order as params_list
            
        Raises:
            RuntimeError: If models are not loaded
        """
        if not params_list:
            return []
        
        if not self.is_loaded:
            raise RuntimeError(
                "LTX-2 models not loaded. Call load_models() first or set "
                "LTX2_DRY_RUN=true for testing."
            )
        
        # Single item - use fast path
        if len(params_list) == 1:
            return [await self.generate_video(params_list[0])]
        
        logger.info(
            f"Starting sequential batch of {len(params_list)} videos "
            f"(warm model mode)"
        )
        
        # Sequential processing with warm model
        results: list[LTX2VideoResult] = []
        for i, params in enumerate(params_list):
            logger.info(
                f"Generating video {i+1}/{len(params_list)} "
                f"(job_id={params.job_id}, {params.width}x{params.height}, "
                f"{params.duration_seconds}s)"
            )
            try:
                result = await self.generate_video(params)
                results.append(result)
            except Exception as e:
                logger.error(f"Video {i+1}/{len(params_list)} failed: {e}")
                # Create empty result to maintain ordering
                results.append(LTX2VideoResult(
                    video_data=b"",
                    width=params.width,
                    height=params.height,
                    duration_seconds=params.duration_seconds,
                    frame_rate=params.frame_rate,
                    has_audio=False,
                    seed=params.seed or 0,
                ))
        
        logger.info(
            f"Batch complete: {sum(1 for r in results if r.video_data)} / "
            f"{len(results)} videos succeeded"
        )
        return results

    # ========================================================================
    # Concurrent Batch Video Generation (Shared Pipeline)
    # ========================================================================

    def _pre_encode_prompt(
        self,
        prompt: str,
        model_ledger: Any,
        enhance_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        negative_prompt: str | None = None,
    ) -> Any:
        """Pre-encode a prompt with the text encoder, serialized by thread lock.
        
        The Gemma LLM has internal mutable state (KV caches, attention buffers)
        that is not thread-safe. This method acquires the text encode lock before
        calling encode_prompts, ensuring only one thread accesses the shared
        text encoder at a time.
        
        Args:
            prompt: Text prompt to encode
            model_ledger: ModelLedger instance for text encoder access
            enhance_prompt: Whether to enhance the prompt
            enhance_prompt_image: Optional image path for prompt enhancement
            enhance_prompt_seed: Seed for prompt enhancement
            negative_prompt: Optional negative prompt (for keyframe interpolation)
            
        Returns:
            Pre-encoded context(s) — single EmbeddingsProcessorOutput for distilled,
            or tuple of (positive, negative) for keyframe interpolation.
        """
        from ltx_pipelines.utils.helpers import encode_prompts
        
        with self._text_encode_lock:
            if negative_prompt is not None:
                # Keyframe interpolation needs both positive and negative contexts
                return encode_prompts(
                    [prompt, negative_prompt],
                    model_ledger,
                    enhance_first_prompt=enhance_prompt,
                    enhance_prompt_image=enhance_prompt_image,
                    enhance_prompt_seed=enhance_prompt_seed,
                )
            else:
                # Distilled pipeline needs only positive context
                (ctx,) = encode_prompts(
                    [prompt],
                    model_ledger,
                    enhance_first_prompt=enhance_prompt,
                    enhance_prompt_image=enhance_prompt_image,
                )
                return ctx

    async def generate_concurrent_batch(
        self,
        params_list: list[LTX2VideoParams],
    ) -> list[LTX2VideoResult]:
        """Generate multiple videos concurrently using shared pipeline.
        
        Uses a two-phase approach for thread safety:
        1. Pre-encode all prompts sequentially (text encoder is not thread-safe)
        2. Run diffusion concurrently (each thread creates its own model instances)
        
        Falls back to sequential processing if concurrent generation fails.
        
        Args:
            params_list: List of video generation parameters
            
        Returns:
            List of VideoGenerationResults in the same order as params_list
            
        Raises:
            RuntimeError: If models are not loaded
        """
        if not params_list:
            return []
        
        if not self.is_loaded:
            raise RuntimeError(
                "LTX-2 models not loaded. Call load_models() first or set "
                "LTX2_DRY_RUN=true for testing."
            )
        
        # Single item - use fast path
        if len(params_list) == 1:
            return [await self.generate_video(params_list[0])]
        
        # Check if concurrent is disabled - fall back to sequential
        if not self._concurrent_enabled:
            logger.info("Concurrent generation disabled, using sequential batch")
            return await self.generate_batch(params_list)
        
        # Calculate dynamic concurrency based on longest video duration
        max_duration = max(p.duration_seconds for p in params_list)
        max_width = max(p.width for p in params_list)
        max_height = max(p.height for p in params_list)
        fps = params_list[0].frame_rate
        
        max_concurrent = vram_estimator.calculate_ltx2_max_concurrent(
            duration_seconds=max_duration,
            width=max_width,
            height=max_height,
            fps=fps,
        )
        
        logger.info(
            f"Starting concurrent batch of {len(params_list)} videos "
            f"(max_concurrent={max_concurrent}, longest={max_duration:.1f}s)"
        )
        
        # Phase 1: Pre-encode all prompts sequentially (thread-safe)
        # This serializes access to the shared Gemma text encoder whose internal
        # KV caches are not safe for concurrent use.
        pre_encoded_contexts = []
        loop = asyncio.get_event_loop()
        
        for i, params in enumerate(params_list):
            logger.debug(f"Pre-encoding prompt {i+1}/{len(params_list)} for job {params.job_id}")
            
            # Determine which pipeline will be used to select the right model_ledger
            # and encoding strategy (with or without negative prompt)
            requested_frames = math.ceil(params.duration_seconds * params.frame_rate) + 1
            num_frames = round_up_to_valid_frames(requested_frames)
            has_end_frame = params.end_frame_data is not None
            num_keyframes = 2 if has_end_frame else 1
            
            if num_keyframes == 1:
                # Will use DistilledPipeline — encode single prompt
                model_ledger = self.components.distilled_pipeline.model_ledger
                ctx = await loop.run_in_executor(
                    None,
                    lambda p=params, ml=model_ledger: self._pre_encode_prompt(
                        prompt=p.prompt,
                        model_ledger=ml,
                        enhance_prompt=p.enhance_prompt,
                    ),
                )
            else:
                # Will use KeyframeInterpolationPipeline — encode prompt + negative
                model_ledger = self.components.keyframe_pipeline.stage_1_model_ledger
                ctx = await loop.run_in_executor(
                    None,
                    lambda p=params, ml=model_ledger: self._pre_encode_prompt(
                        prompt=p.prompt,
                        model_ledger=ml,
                        enhance_prompt=p.enhance_prompt,
                        negative_prompt=p.negative_prompt or "",
                        enhance_prompt_seed=p.seed or 42,
                    ),
                )
            
            pre_encoded_contexts.append(ctx)
        
        logger.info(f"Pre-encoded {len(pre_encoded_contexts)} prompts, starting concurrent diffusion")
        
        # Phase 2: Run diffusion concurrently (each thread gets its own models)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_one(
            idx: int, params: LTX2VideoParams, pre_ctx: Any
        ) -> LTX2VideoResult:
            """Generate a single video with semaphore-controlled concurrency."""
            async with semaphore:
                logger.debug(
                    f"Starting video {idx+1}/{len(params_list)} "
                    f"(job_id={params.job_id}, {params.duration_seconds}s)"
                )
                try:
                    result = await self.generate_video(params, pre_encoded_context=pre_ctx)
                    logger.debug(f"Completed video {idx+1}/{len(params_list)}")
                    return result
                except Exception as e:
                    logger.error(
                        f"Video {idx+1}/{len(params_list)} failed: {e}", 
                        exc_info=True
                    )
                    # Return empty result to maintain ordering
                    return LTX2VideoResult(
                        video_data=b"",
                        width=params.width,
                        height=params.height,
                        duration_seconds=params.duration_seconds,
                        frame_rate=params.frame_rate,
                        has_audio=False,
                        seed=params.seed or 0,
                    )
        
        # Run all generations concurrently (semaphore limits parallelism)
        try:
            results = await asyncio.gather(*[
                generate_one(i, p, ctx) 
                for i, (p, ctx) in enumerate(zip(params_list, pre_encoded_contexts))
            ])
        except Exception as e:
            logger.error(
                f"Concurrent batch failed: {e}. Falling back to sequential.",
                exc_info=True
            )
            # Fallback to sequential processing
            return await self.generate_batch(params_list)
        
        succeeded = sum(1 for r in results if r.video_data)
        logger.info(
            f"Concurrent batch complete: {succeeded}/{len(results)} videos succeeded"
        )
        
        return list(results)

    @property
    def concurrent_enabled(self) -> bool:
        """Whether concurrent video generation is enabled."""
        return self._concurrent_enabled

    # ========================================================================
    # Keyframe Interpolation Generation
    # ========================================================================

    async def generate_keyframe_video(
        self, 
        params: KeyframeInterpolationParams,
        pre_encoded_context: Any | None = None,
    ) -> LTX2VideoResult:
        """Generate a video by interpolating between keyframes.

        Args:
            params: Generation parameters including keyframes, prompt, dimensions
            pre_encoded_context: Optional pre-encoded prompt embeddings (for concurrent batch)

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
                target_height=target_height,
                pre_encoded_context=pre_encoded_context,
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
        pre_encoded_context: Any | None = None,
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

        # Wrap entire generation in inference_mode to match official LTX-2 CLI pattern
        # This ensures encode_video (which iterates the video generator) runs in the same context
        with torch.no_grad():
            return self._generate_sync_inner(
                params, num_frames, seed, target_width, target_height,
                TilingConfig, get_video_chunks_number, encode_video,
                pre_encoded_context=pre_encoded_context,
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
        pre_encoded_context: Any | None = None,
    ) -> tuple[bytes, bool]:
        """Inner implementation of _generate_sync - runs inside inference_mode context.
        
        When pre_encoded_context is provided (from concurrent batch), it is passed
        directly to the pipeline, skipping the text encoder entirely.
        """

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
            from ltx_pipelines.utils.args import ImageConditioningInput
            images.append(ImageConditioningInput(path=str(image_path), frame_idx=frame_idx, strength=strength, crf=0))  # crf=0: skip lossy compression

        
        # Configure tiling for video decoding
        tiling_config = TilingConfig.default()

        # Validate keyframe count
        num_keyframes = len(params.keyframes)
        if num_keyframes > 2:
            raise ValueError(
                f"Too many keyframes ({num_keyframes}). Video generation supports "
                f"maximum 2 keyframes: 1 start frame with optional end frame."
            )
        
        if num_keyframes < 1:
            raise ValueError("At least 1 keyframe (start frame) is required for video generation.")
        
        # Select pipeline based on keyframe count:
        # - 1 keyframe: DistilledPipeline (latent replacement for exact preservation)
        # - 2 keyframes: KeyframeInterpolationPipeline (guiding latents for smooth transitions)
        if num_keyframes == 1:
            # Single keyframe: use DistilledPipeline (fastest, exact preservation)
            logger.info("Using DistilledPipeline for single-keyframe I2V")
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
                pre_encoded_context=pre_encoded_context,
            )
        else:
            # Two keyframes: use KeyframeInterpolationPipeline with distilled schedule
            # This uses guiding latents for smoother transitions between start/end frames
            from ltx_core.components.guiders import MultiModalGuiderParams
            logger.info("Using KeyframeInterpolationPipeline for 2-keyframe interpolation")
            video_chunks, audio = self.components.keyframe_pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                seed=seed,
                height=params.height,
                width=params.width,
                num_frames=num_frames,
                frame_rate=params.frame_rate,
                num_inference_steps=8,
                video_guider_params=MultiModalGuiderParams(cfg_scale=1.0),
                audio_guider_params=MultiModalGuiderParams(cfg_scale=1.0),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=params.enhance_prompt,
                pre_encoded_contexts=pre_encoded_context,
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
        # For single-keyframe (I2V): trim to exact duration (e.g., 120 frames for 5s @ 24fps)
        # For multi-keyframe (interpolation): preserve all frames to include the end keyframe
        num_keyframes = len(params.keyframes)
        if num_keyframes > 1:
            # Preserve all generated frames to include the end keyframe
            target_frames_exact = num_frames
        else:
            # Standard I2V: trim to exact requested duration
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
            )

        # Encode to MP4 with audio
        output_path = Path(self._temp_dir.name) / f"{params.job_id}_output.mp4"
        
        # When passing a single tensor, video_chunks_number is 1
        video_chunks_number = 1

        encode_video(
            video=video_tensor,
            fps=params.frame_rate,
            audio=audio,
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
        audio,
        target_duration: float,
    ):
        """Trim Audio object's waveform to target duration.
        
        Args:
            audio: ltx_core.types.Audio dataclass with .waveform and .sampling_rate
            target_duration: target duration in seconds
        Returns:
            Audio object with trimmed waveform
        """
        from dataclasses import replace as dc_replace
        target_samples = int(target_duration * audio.sampling_rate)
        waveform = audio.waveform
        
        # Waveform shape: (Samples,) or (Samples, Channels)
        if waveform.shape[0] > target_samples:
            logger.info(f"Trimming audio waveform from {waveform.shape[0]} to {target_samples} samples")
            waveform = waveform[:target_samples]
            return dc_replace(audio, waveform=waveform)
            
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
            "model": "LTX-2.3-22b-dev-fp8 + Distilled LoRA",
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
