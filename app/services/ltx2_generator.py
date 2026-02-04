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
from app.services.ltx2_concurrent import LTX2ConcurrencyController
from app.services import vram_estimator

logger = logging.getLogger(__name__)


# ============================================================================
# SageAttention 2.2.0: Blackwell-optimized FP8 attention (NON-GLOBAL)
# Uses a context manager to patch SDPA only during LTX-2 inference
# This prevents SageAttention from affecting Z-Image or Qwen-Image-Edit
# ============================================================================
_sage_attn = None
_original_sdpa = None
_sage_available = False
_sage_call_count = 0
_fallback_call_count = 0

try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda as _sage_attn
    import torch.nn.functional as F
    _original_sdpa = F.scaled_dot_product_attention
    _sage_available = True
    logger.info("SageAttention 2.2.0 available (FP8 CUDA backend, Blackwell-optimized)")
except ImportError:
    logger.info("SageAttention not installed - using PyTorch SDPA")


def _make_sage_sdpa():
    """Create a SageAttention-patched SDPA function."""
    def _patched_sdpa(query, key, value, attn_mask=None, *args, **kwargs):
        global _sage_call_count, _fallback_call_count
        # Fall back to original SDPA when masks are used
        # (SageAttention CUDA backend doesn't support attention masks)
        if attn_mask is not None:
            _fallback_call_count += 1
            return _original_sdpa(query, key, value, attn_mask, *args, **kwargs)

        # SageAttention expects tensor_layout="HND" which matches LTX-2's
        # [Batch, Heads, SeqLen, Dim] layout after PytorchAttention reshape
        _sage_call_count += 1
        is_causal = kwargs.get('is_causal', False)
        # Blackwell-optimized parameters (sm120/sm121):
        # - qk_quant_gran="per_warp": Faster quantization granularity on Blackwell
        # - pv_accum_dtype="fp32+fp16": SageAttention2++ mode for speed/accuracy balance
        # - smooth_k=True: Improves accuracy with minimal overhead
        return _sage_attn(
            query, key, value,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran="per_warp",
            pv_accum_dtype="fp32+fp16",
            smooth_k=True,
        )
    return _patched_sdpa


class sage_attention_context:
    """Context manager to temporarily enable SageAttention for LTX-2 only.
    
    Usage:
        with sage_attention_context():
            # SageAttention is active here
            result = pipeline(...)
        # Original SDPA is restored here
    
    This prevents SageAttention from leaking into Z-Image or Qwen-Image-Edit.
    """
    
    def __enter__(self):
        if not _sage_available:
            return self
        
        import torch.nn.functional as F
        import torch.nn.functional
        
        # Patch SDPA with SageAttention
        self._patched = _make_sage_sdpa()
        F.scaled_dot_product_attention = self._patched
        torch.nn.functional.scaled_dot_product_attention = self._patched
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _sage_available:
            return False
        
        import torch.nn.functional as F
        import torch.nn.functional
        
        # Restore original SDPA
        F.scaled_dot_product_attention = _original_sdpa
        torch.nn.functional.scaled_dot_product_attention = _original_sdpa
        return False


def log_sage_stats():
    """Log SageAttention usage statistics."""
    if not _sage_available:
        return {"sage_calls": 0, "fallback_calls": 0, "disabled": True}
    logger.info(f"SageAttention stats: {_sage_call_count} SageAttn calls, {_fallback_call_count} fallback calls")
    return {"sage_calls": _sage_call_count, "fallback_calls": _fallback_call_count}


# ============================================================================
# TeaCache: Step-Level Caching for 1.4-2.1x Speedup
# Based on official ali-vilab/TeaCache implementation for LTX-Video
# https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/teacache_ltx.py
#
# Key insight: When timestep embeddings are similar between steps, the transformer
# output changes minimally. We can reuse the previous residual instead of recomputing.
# ============================================================================

import numpy as np

# Polynomial coefficients for LTX-Video rescaling (from official TeaCache)
# These rescale the relative L1 distance to better predict output similarity
_TEACACHE_COEFFICIENTS_LTXV = [
    2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 
    7.92487521e-01, 9.69274326e-03
]


class teacache_context:
    """Context manager to enable TeaCache for LTX-2 inference.
    
    Patches LTXModel.forward to implement timestep-embedding-aware caching.
    Follows the official ali-vilab/TeaCache implementation pattern.
    
    Args:
        thresh: Relative L1 distance threshold. Lower = more accuracy, less speedup.
                Recommended: 0.03 for 1.6x, 0.05 for 2.1x speedup
        enabled: Whether to enable TeaCache
    
    Usage:
        with teacache_context(thresh=0.05, enabled=True):
            result = pipeline(...)
    """
    
    def __init__(self, thresh: float = 0.05, enabled: bool = True):
        self.thresh = thresh
        self.enabled = enabled
        self._original_forward = None
        self._ltx_model_class = None
    
    def __enter__(self):
        if not self.enabled:
            return self
        
        try:
            from ltx_core.model.transformer.model import LTXModel
            self._ltx_model_class = LTXModel
        except ImportError:
            logger.warning("Could not import LTXModel, TeaCache disabled")
            return self
        
        # Save original forward method
        self._original_forward = LTXModel.forward
        
        # Create patched forward with TeaCache logic
        original_forward = self._original_forward
        thresh = self.thresh
        coefficients = _TEACACHE_COEFFICIENTS_LTXV
        
        def teacache_forward(
            model_self, 
            video, 
            audio, 
            perturbations
        ):
            """TeaCache-enhanced forward pass.
            
            Implements the official TeaCache algorithm (ali-vilab/TeaCache):
            1. Apply norm1 to hidden states before computing modulated input
            2. Compare with previous modulated input using rescaled L1 distance
            3. If accumulated distance < threshold, add cached residual (which includes norm+scale-shift)
            4. Otherwise, compute transformer blocks, apply norm+scale-shift, then cache residual
            
            CRITICAL: The residual is computed AFTER norm_out and scale-shift, not before.
            This matches the official implementation exactly.
            """
            import torch
            from ltx_core.utils import rms_norm
            
            # Initialize TeaCache state on the model instance if needed
            # OR reset if input shape changed (handles multi-stage pipelines and new generations)
            current_shape = video_args.x.shape if video_args is not None else None
            
            if not hasattr(model_self, '_teacache_enabled'):
                # First time initialization
                model_self._teacache_enabled = True
                model_self._teacache_cnt = 0
                model_self._teacache_accumulated_rel_l1_distance = 0.0
                model_self._teacache_previous_modulated_input = None
                model_self._teacache_previous_residual = None
                model_self._teacache_previous_residual_audio = None
                model_self._teacache_skip_count = 0
                model_self._teacache_compute_count = 0
                model_self._teacache_last_shape = current_shape
            elif current_shape != model_self._teacache_last_shape:
                # Shape changed (new generation or new pipeline stage) - reset cached tensors
                # Keep skip/compute counts for stats, but reset caching state
                model_self._teacache_cnt = 0
                model_self._teacache_accumulated_rel_l1_distance = 0.0
                model_self._teacache_previous_modulated_input = None
                model_self._teacache_previous_residual = None
                model_self._teacache_previous_residual_audio = None
                model_self._teacache_last_shape = current_shape
            
            if not model_self._teacache_enabled:
                return original_forward(model_self, video, audio, perturbations)
            
            # Handle non-video case
            if not model_self.model_type.is_video_enabled() or video is None:
                return original_forward(model_self, video, audio, perturbations)
            
            # Get the preprocessed video args (contains hidden_states and timestep info)
            video_args = model_self.video_args_preprocessor.prepare(video)
            audio_args = model_self.audio_args_preprocessor.prepare(audio) if audio is not None else None
            
            # ===== TeaCache Decision Logic =====
            # Compute modulated input from first transformer block
            hidden_states = video_args.x
            timestep = video_args.timesteps  # This is the embedded timestep (TransformerArgs.timesteps)
            batch_size = hidden_states.shape[0]
            
            # Get first block for modulation calculation
            first_block = model_self.transformer_blocks[0]
            
            # CRITICAL FIX #1: Apply rms_norm BEFORE modulation (matches official implementation)
            # Official: inp = self.transformer_blocks[0].norm1(inp)
            normed_hidden_states = rms_norm(hidden_states, eps=first_block.norm_eps)
            
            # Get first block's scale_shift_table for modulation
            scale_shift_table = first_block.scale_shift_table
            num_ada_params = scale_shift_table.shape[0]
            
            # Compute AdaLN modulation: scale_shift_table + timestep
            ada_values = (
                scale_shift_table[None, None].to(device=timestep.device, dtype=timestep.dtype)
                + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)
            )
            shift_msa, scale_msa = ada_values[:, :, 0], ada_values[:, :, 1]
            
            # Modulated input = normed_hidden_states * (1 + scale) + shift
            modulated_inp = normed_hidden_states * (1 + scale_msa) + shift_msa
            
            # Determine if we should compute or skip
            # Use threshold-based approach only (no fixed step counting)
            # This handles multi-stage pipelines like distilled (8+3 steps)
            cnt = model_self._teacache_cnt
            
            if cnt == 0:
                # Always compute first step after shape change/reset
                should_calc = True
                model_self._teacache_accumulated_rel_l1_distance = 0.0
            elif model_self._teacache_previous_modulated_input is None:
                # First time, need to compute
                should_calc = True
                model_self._teacache_accumulated_rel_l1_distance = 0.0
            else:
                # Compare with previous modulated input
                prev = model_self._teacache_previous_modulated_input
                rel_l1_dist = (modulated_inp - prev).abs().mean() / (prev.abs().mean() + 1e-8)
                
                # Apply polynomial rescaling (matches official implementation)
                rescale_func = np.poly1d(coefficients)
                rescaled_dist = rescale_func(rel_l1_dist.cpu().item())
                
                model_self._teacache_accumulated_rel_l1_distance += rescaled_dist
                
                if model_self._teacache_accumulated_rel_l1_distance < thresh:
                    should_calc = False
                else:
                    should_calc = True
                    model_self._teacache_accumulated_rel_l1_distance = 0.0
            
            # Update state
            model_self._teacache_previous_modulated_input = modulated_inp.detach().clone()
            model_self._teacache_cnt += 1
            
            # ===== Execute or Skip =====
            if not should_calc and model_self._teacache_previous_residual is not None:
                # SKIP PATH: Add cached residual directly
                # The residual already includes norm_out + scale-shift, so we only apply proj_out
                model_self._teacache_skip_count += 1
                
                # Add cached residual to original hidden states
                # Official: hidden_states += self.previous_residual
                skipped_video_hidden = video_args.x + model_self._teacache_previous_residual
                
                # Only apply proj_out (norm+scale-shift already baked into residual)
                vx = model_self.proj_out(skipped_video_hidden)
                
                # Audio handling (if present)
                ax = None
                if audio_args is not None and model_self._teacache_previous_residual_audio is not None:
                    skipped_audio_hidden = audio_args.x + model_self._teacache_previous_residual_audio
                    ax = model_self.audio_proj_out(skipped_audio_hidden)
                
                return vx, ax
            else:
                # COMPUTE PATH: Full forward through transformer blocks
                model_self._teacache_compute_count += 1
                
                # Store original hidden states for residual calculation
                original_video_x = video_args.x.clone()
                original_audio_x = audio_args.x.clone() if audio_args is not None else None
                
                # Process transformer blocks - this is the expensive part we skip when caching
                video_out, audio_out = model_self._process_transformer_blocks(
                    video=video_args,
                    audio=audio_args,
                    perturbations=perturbations,
                )
                
                # CRITICAL FIX #2: Compute residual AFTER norm_out and scale-shift
                # This matches official: self.previous_residual = hidden_states - ori_hidden_states
                # where hidden_states has been through norm_out and scale-shift
                
                vx = None
                if video_out is not None:
                    # Apply norm_out and scale-shift
                    scale_shift_values = (
                        model_self.scale_shift_table[None, None].to(
                            device=video_out.x.device, dtype=video_out.x.dtype
                        ) + video_out.embedded_timestep[:, :, None]
                    )
                    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
                    
                    normed_video = model_self.norm_out(video_out.x)
                    processed_video = normed_video * (1 + scale) + shift
                    
                    # Cache residual AFTER norm+scale-shift (official pattern)
                    model_self._teacache_previous_residual = processed_video - original_video_x
                    
                    # Apply proj_out
                    vx = model_self.proj_out(processed_video)
                
                ax = None
                if audio_out is not None and original_audio_x is not None:
                    # Apply norm_out and scale-shift for audio
                    audio_scale_shift_values = (
                        model_self.audio_scale_shift_table[None, None].to(
                            device=audio_out.x.device, dtype=audio_out.x.dtype
                        ) + audio_out.embedded_timestep[:, :, None]
                    )
                    audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
                    
                    normed_audio = model_self.audio_norm_out(audio_out.x)
                    processed_audio = normed_audio * (1 + audio_scale) + audio_shift
                    
                    # Cache audio residual AFTER norm+scale-shift
                    model_self._teacache_previous_residual_audio = processed_audio - original_audio_x
                    
                    # Apply audio proj_out
                    ax = model_self.audio_proj_out(processed_audio)
                
                return vx, ax
        
        # Apply patch
        LTXModel.forward = teacache_forward
        logger.info(f"TeaCache enabled with threshold {self.thresh}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or self._original_forward is None:
            return False
        
        # Log session completion for debugging
        logger.info(f"TeaCache session completed (thresh={self.thresh})")
        
        # Restore original forward
        if self._ltx_model_class is not None:
            self._ltx_model_class.forward = self._original_forward
            logger.info("TeaCache: Restored original LTXModel.forward")
        
        return False


def log_teacache_stats() -> dict:
    """Log and return TeaCache usage statistics.
    
    Note: With the new implementation, TeaCache stats are stored per-model-instance
    and are logged automatically when the teacache_context exits.
    This function is kept for backward compatibility but returns empty stats.
    """
    # Stats are now per-model-instance and logged in teacache_context.__exit__
    return {"skip_count": 0, "compute_count": 0, "skip_rate": 0.0, "note": "See INFO logs for stats"}


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

        # Initialize device for all pipelines
        device = torch.device(self.settings.ltx2_device)

        # DistilledPipeline for I2V generation (1 keyframe)
        # Uses latent replacement conditioning - exact keyframe preservation
        logger.info("Loading DistilledPipeline for single-keyframe I2V generation...")
        try:
            distilled_pipeline = DistilledPipeline(
                checkpoint_path=str(checkpoint_path.absolute()),
                spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
                gemma_root=str(gemma_root.absolute()),
                loras=[],  # No extra LoRAs
                device=device,
                fp8transformer=self.settings.ltx2_fp8_enabled,
            )
        except Exception:
            logger.exception("Failed to initialize DistilledPipeline")
            raise
        logger.info("DistilledPipeline loaded successfully")
        
        # KeyframeInterpolationPipeline for 2-keyframe requests
        # Uses guiding latents conditioning - smoother transitions between keyframes
        # Share all components from DistilledPipeline to save ~60% VRAM
        logger.info("Initializing KeyframeInterpolationPipeline with shared components...")
        try:
            keyframe_pipeline = KeyframeInterpolationPipeline(
                checkpoint_path=str(checkpoint_path.absolute()),
                distilled_lora=[],  # Already using distilled checkpoint
                spatial_upsampler_path=str(spatial_upsampler_path.absolute()),
                gemma_root=str(gemma_root.absolute()),
                loras=[],  # No extra LoRAs
                device=device,
                fp8transformer=self.settings.ltx2_fp8_enabled,
                # Share all components from DistilledPipeline (VRAM optimization)
                shared_text_encoder=distilled_pipeline.text_encoder,
                shared_video_encoder=distilled_pipeline.video_encoder,
                shared_transformer=distilled_pipeline.transformer,
                shared_spatial_upsampler=distilled_pipeline.spatial_upsampler,
                shared_video_decoder=distilled_pipeline.video_decoder,
                shared_audio_decoder=distilled_pipeline.audio_decoder,
                shared_vocoder=distilled_pipeline.vocoder,
            )
        except Exception:
            logger.exception("Failed to initialize KeyframeInterpolationPipeline")
            raise
        logger.info("KeyframeInterpolationPipeline initialized with shared components")

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
        
        # Log SageAttention usage stats from warmup
        log_sage_stats()

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
            # Use sage_attention_context to enable SageAttention during warmup
            with torch.no_grad(), sage_attention_context():
                tiling_config = TilingConfig.default()
                
                # Warmup DistilledPipeline
                video_chunks, audio = self.components.distilled_pipeline(
                    prompt="warmup",
                    seed=42,
                    height=256,  # Small but divisible by 64
                    width=256,
                    num_frames=9,  # Minimum valid: 1 + 8*1 = 9
                    frame_rate=24.0,
                    images=[(str(warmup_path), 0, 1.0)],
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
            logger.warning(f"LTX-2 warmup failed (non-fatal): {e}")
            # Warmup failure is non-fatal - first real inference will just be slow

    def _patch_model_ledger_caching(self) -> None:
        """Patch model_ledger to cache and reuse model instances.
        
        The upstream LTX-2 model_ledger creates new model instances on each call.
        This patches the text_encoder() method to return the already-loaded instance
        from DistilledPipeline, preventing race conditions during concurrent video
        generation AND avoiding loading the text encoder twice.
        """
        distilled = self.components.distilled_pipeline
        
        # Use the already-loaded text encoder from DistilledPipeline.__init__
        # This avoids loading it twice (which wastes ~18GB VRAM!)
        logger.info("Patching text encoder for thread-safe concurrent access...")
        
        try:
            # The text encoder is already loaded at distilled.text_encoder
            # We just patch model_ledger to return it instead of loading a new one
            cached_text_encoder = distilled.text_encoder
            
            def cached_text_encoder_fn():
                return cached_text_encoder
            
            distilled.model_ledger.text_encoder = cached_text_encoder_fn
            logger.info("  Patched model_ledger.text_encoder() to reuse existing instance")
            logger.info("Text encoder caching enabled - concurrent generation is now thread-safe")
            
        except Exception as e:
            logger.warning(f"Failed to patch model_ledger caching (non-fatal): {e}")
            # If patching fails, concurrent generation may still have race conditions
            # but single-threaded generation will still work



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
        return await self.generate_keyframe_video(keyframe_params)

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

    async def generate_concurrent_batch(
        self,
        params_list: list[LTX2VideoParams],
    ) -> list[LTX2VideoResult]:
        """Generate multiple videos concurrently using shared pipeline.
        
        This method leverages the stateless nature of LTX-2's DistilledPipeline
        to run multiple video generations in parallel. Concurrency is dynamically
        limited based on video duration and available VRAM.
        
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
        
        # Create semaphore for this batch
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_one(idx: int, params: LTX2VideoParams) -> LTX2VideoResult:
            """Generate a single video with semaphore-controlled concurrency."""
            async with semaphore:
                logger.debug(
                    f"Starting video {idx+1}/{len(params_list)} "
                    f"(job_id={params.job_id}, {params.duration_seconds}s)"
                )
                try:
                    result = await self.generate_video(params)
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
                generate_one(i, p) for i, p in enumerate(params_list)
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

        # Get TeaCache settings from config
        teacache_enabled = self.settings.ltx2_teacache_enabled
        teacache_thresh = self.settings.ltx2_teacache_thresh

        # Wrap entire generation in inference_mode to match official LTX-2 CLI pattern
        # This ensures encode_video (which iterates the video generator) runs in the same context
        # Also wrap with sage_attention_context to enable SageAttention ONLY during LTX-2 inference
        # And teacache_context for step-skipping acceleration (1.4-1.7x speedup)
        with torch.no_grad(), sage_attention_context(), teacache_context(
            thresh=teacache_thresh, 
            enabled=teacache_enabled
        ):
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
            )
        else:
            # Two keyframes: use KeyframeInterpolationPipeline with distilled schedule
            # This uses guiding latents for smoother transitions between start/end frames
            logger.info("Using KeyframeInterpolationPipeline for 2-keyframe interpolation (8-step distilled mode)")
            video_chunks, audio = self.components.keyframe_pipeline(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                seed=seed,
                height=params.height,
                width=params.width,
                num_frames=num_frames,
                frame_rate=params.frame_rate,
                num_inference_steps=8,  # Placeholder, ignored when use_distilled_schedule=True
                cfg_guidance_scale=1.0,  # Placeholder, ignored when use_distilled_schedule=True
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=params.enhance_prompt,
                use_distilled_schedule=True,  # Enable 8-step fast mode
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
