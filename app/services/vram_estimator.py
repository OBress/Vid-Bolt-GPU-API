"""VRAM Estimation Utilities.

This module provides functions to estimate VRAM usage for various generation
tasks and query available GPU memory. Used by JobManager for dynamic batch sizing.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)

# VRAM estimation constants (in GB)
# These are conservative estimates calibrated for RTX PRO 6000

# Base costs (model weights already loaded)
ZIMAGE_BASE_ACTIVATION_GB = 0.5  # Base activation memory overhead

# Per-pixel scaling factors (GB per megapixel)
# 1 Megapixel = 1,000,000 pixels
ZIMAGE_GB_PER_MEGAPIXEL = 1.2  # ~1.2 GB per megapixel for Z-Image Turbo

# Safety margins
VRAM_SAFETY_MARGIN_GB = 4.0  # Reserve for OS, display, fragmentation
MIN_FREE_VRAM_GB = 2.0  # Minimum free VRAM to attempt any generation

# Batch limits (absolute caps regardless of VRAM)
MAX_BATCH_SIZE_ZIMAGE = 64  # Never exceed this even with infinite VRAM

# =============================================================================
# LightX2V (Qwen-Image-Edit-2511) VRAM Estimation Constants
# =============================================================================
# LightX2V is an image-to-image editing model with Qwen25-VL text encoder.
# Higher per-image cost than Z-Image due to vision-language conditioning.

# Base model costs (in GB)
LIGHTX2V_BASE_MODEL_FULL_GB = 28.0      # DiT + Qwen25-VL + VAE without offload
LIGHTX2V_BASE_MODEL_OFFLOAD_GB = 12.0   # With CPU offload enabled (default config)

# Per-image activation costs (in GB)
# Higher than Z-Image because each edit requires:
# 1. Vision encoding of input image via Qwen25-VL
# 2. VAE encoding/decoding of input/output
# 3. Conditioning latent storage
LIGHTX2V_BASE_ACTIVATION_GB = 1.0       # Base overhead per image
LIGHTX2V_GB_PER_MEGAPIXEL = 2.5         # ~2.5 GB per megapixel (input + output)
LIGHTX2V_CONDITIONING_OVERHEAD_GB = 0.5 # Vision-language conditioning per image

# Batch limit for LightX2V (conservative due to sequential processing)
MAX_BATCH_SIZE_LIGHTX2V = 16  # Cap even with high VRAM for memory stability

# =============================================================================
# LTX-2 (Video Generation) VRAM Estimation Constants
# =============================================================================
# LTX-2 is a 19B parameter video generation model using FP8 distilled checkpoint.
# Uses two-stage pipeline: Stage 1 at half resolution, Stage 2 at full resolution.

# Base model costs (in GB) - FP8 distilled model
LTX2_BASE_MODEL_FP8_GB = 20.0           # FP8 DiT + Gemma text encoder in VRAM
LTX2_BASE_MODEL_FP16_GB = 40.0          # FP16 model (not used in current config)

# Per-video activation costs (in GB)
# Video generation requires storing:
# 1. Latent tensors for all frames (both stages)
# 2. Conditioning tensors (text + image)
# 3. Intermediate activations during denoising
LTX2_BASE_ACTIVATION_GB = 5.0           # Base overhead per video
LTX2_GB_PER_MEGAPIXEL = 0.015           # Per megapixel scaling (at full res)
LTX2_GB_PER_FRAME = 0.08                # Per frame scaling (at 1080p baseline)

# Batch limit for LTX-2 sequential processing (fallback mode)
MAX_BATCH_SIZE_LTX2 = 1  # Sequential batching when concurrent disabled

# =============================================================================
# LTX-2 Concurrent Video Generation Constants
# =============================================================================
# The LTX-2 DistilledPipeline.__call__ is stateless - all generation state
# is local variables, model weights are read-only. This enables safe concurrent
# video generation with a shared pipeline.

LTX2_CONCURRENT_OVERHEAD_GB = 2.0      # Additional overhead per concurrent slot
LTX2_MAX_CONCURRENT_VIDEOS = 4         # Absolute cap regardless of VRAM
LTX2_CONCURRENT_VRAM_BUDGET_GB = 72.0  # Available VRAM after base model + safety
LTX2_CONCURRENT_SAFETY_FACTOR = 0.9    # Use 90% of calculated capacity for safety


@dataclass
class VRAMInfo:
    """Container for VRAM status."""
    
    free_gb: float
    total_gb: float
    used_gb: float
    
    @property
    def available_for_inference_gb(self) -> float:
        """Available VRAM after applying safety margin."""
        return max(0.0, self.free_gb - VRAM_SAFETY_MARGIN_GB)


def get_vram_info() -> VRAMInfo:
    """Get current VRAM usage information.
    
    Returns:
        VRAMInfo with current GPU memory statistics
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, returning zero VRAM info")
            return VRAMInfo(free_gb=0.0, total_gb=0.0, used_gb=0.0)
        
        # Get memory info in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        
        # Convert to GB
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        used_gb = total_gb - free_gb
        
        return VRAMInfo(free_gb=free_gb, total_gb=total_gb, used_gb=used_gb)
        
    except ImportError:
        logger.warning("PyTorch not available, returning zero VRAM info")
        return VRAMInfo(free_gb=0.0, total_gb=0.0, used_gb=0.0)
    except Exception as e:
        logger.error(f"Failed to get VRAM info: {e}")
        return VRAMInfo(free_gb=0.0, total_gb=0.0, used_gb=0.0)


def estimate_zimage_vram_per_image(width: int, height: int) -> float:
    """Estimate VRAM usage for a single Z-Image generation.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Estimated VRAM usage in GB
    """
    # Calculate megapixels
    megapixels = (width * height) / 1_000_000
    
    # Estimate: base overhead + scaled by resolution
    estimated_gb = ZIMAGE_BASE_ACTIVATION_GB + (megapixels * ZIMAGE_GB_PER_MEGAPIXEL)
    
    return estimated_gb


def calculate_max_batch_size(
    width: int, 
    height: int, 
    available_vram_gb: float | None = None
) -> int:
    """Calculate maximum batch size for Z-Image generation.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        available_vram_gb: Override for available VRAM (None = auto-detect)
        
    Returns:
        Maximum safe batch size (minimum 1)
    """
    # Get available VRAM
    if available_vram_gb is None:
        vram_info = get_vram_info()
        available_vram_gb = vram_info.available_for_inference_gb
    
    # Check minimum threshold
    if available_vram_gb < MIN_FREE_VRAM_GB:
        logger.warning(f"Low VRAM ({available_vram_gb:.1f}GB), limiting to batch size 1")
        return 1
    
    # Estimate per-image cost
    vram_per_image = estimate_zimage_vram_per_image(width, height)
    
    if vram_per_image <= 0:
        return 1
    
    # Calculate max batch
    max_batch = int(available_vram_gb / vram_per_image)
    
    # Apply absolute cap
    max_batch = min(max_batch, MAX_BATCH_SIZE_ZIMAGE)
    
    # Ensure at least 1
    max_batch = max(1, max_batch)
    
    logger.debug(
        f"Batch size calculation: {available_vram_gb:.1f}GB available, "
        f"{vram_per_image:.2f}GB per image ({width}x{height}), "
        f"max batch = {max_batch}"
    )
    
    return max_batch


def log_vram_status() -> None:
    """Log current VRAM status for debugging."""
    vram_info = get_vram_info()
    logger.info(
        f"VRAM Status: {vram_info.used_gb:.1f}GB / {vram_info.total_gb:.1f}GB used, "
        f"{vram_info.free_gb:.1f}GB free, "
        f"{vram_info.available_for_inference_gb:.1f}GB available for inference"
    )


# =============================================================================
# LightX2V VRAM Estimation Functions
# =============================================================================

def estimate_lightx2v_vram_per_image(width: int, height: int) -> float:
    """Estimate VRAM usage for a single LightX2V image edit.
    
    LightX2V requires more VRAM per image than Z-Image because:
    1. Vision-language encoder (Qwen25-VL) processes the input image
    2. VAE encodes input and decodes output
    3. Conditioning latents must be stored throughout generation
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Estimated VRAM usage in GB for one image edit
    """
    # Calculate megapixels
    megapixels = (width * height) / 1_000_000
    
    # Estimate: base overhead + per-megapixel scaling + conditioning
    estimated_gb = (
        LIGHTX2V_BASE_ACTIVATION_GB +
        (megapixels * LIGHTX2V_GB_PER_MEGAPIXEL) +
        LIGHTX2V_CONDITIONING_OVERHEAD_GB
    )
    
    return estimated_gb


def calculate_lightx2v_max_batch_size(
    width: int,
    height: int,
    available_vram_gb: float | None = None,
    cpu_offload: bool = True,
    other_models_loaded: bool = False
) -> int:
    """Calculate maximum batch size for LightX2V image editing.
    
    This function is VRAM-mode aware:
    - In IMAGE_EDITING mode (LightX2V only): More VRAM available for batching
    - In ALL mode (multiple models): Less VRAM available, smaller batches
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        available_vram_gb: Override for available VRAM (None = auto-detect)
        cpu_offload: Whether CPU offload is enabled (affects base model cost)
        other_models_loaded: Whether Z-Image/LTX-2 are also loaded (ALL mode)
        
    Returns:
        Maximum safe batch size (minimum 1)
    """
    # Get available VRAM
    if available_vram_gb is None:
        vram_info = get_vram_info()
        available_vram_gb = vram_info.available_for_inference_gb
    
    # Account for base model VRAM if calculating from scratch
    # (If LightX2V is already loaded, this is already in used_gb,
    # so available_for_inference_gb already accounts for it)
    
    # Reduce available VRAM if other models are loaded (ALL mode)
    # This is an approximation - in practice, model_manager handles mode switching
    if other_models_loaded:
        # Reserve VRAM for Z-Image (~4GB) and LTX-2 (~8GB) model weights
        # that may still be partially in VRAM
        available_vram_gb = max(0.0, available_vram_gb - 8.0)
        logger.debug(f"ALL mode detected, reduced available VRAM to {available_vram_gb:.1f}GB")
    
    # Check minimum threshold
    if available_vram_gb < MIN_FREE_VRAM_GB:
        logger.warning(
            f"Low VRAM ({available_vram_gb:.1f}GB), limiting LightX2V to batch size 1"
        )
        return 1
    
    # Estimate per-image cost
    vram_per_image = estimate_lightx2v_vram_per_image(width, height)
    
    if vram_per_image <= 0:
        logger.warning("Invalid VRAM estimation, defaulting to batch size 1")
        return 1
    
    # Calculate max batch
    max_batch = int(available_vram_gb / vram_per_image)
    
    # Apply absolute cap (more conservative than Z-Image due to sequential nature)
    max_batch = min(max_batch, MAX_BATCH_SIZE_LIGHTX2V)
    
    # Ensure at least 1
    max_batch = max(1, max_batch)
    
    logger.debug(
        f"LightX2V batch size: {available_vram_gb:.1f}GB available, "
        f"{vram_per_image:.2f}GB per image ({width}x{height}), "
        f"max batch = {max_batch}"
    )
    
    return max_batch


def get_lightx2v_base_vram(cpu_offload: bool = True) -> float:
    """Get the base VRAM footprint for LightX2V model.
    
    Args:
        cpu_offload: Whether CPU offload is enabled
        
    Returns:
        Base VRAM usage in GB for model weights
    """
    if cpu_offload:
        return LIGHTX2V_BASE_MODEL_OFFLOAD_GB
    return LIGHTX2V_BASE_MODEL_FULL_GB


def calculate_lightx2v_optimal_pool_size(
    available_vram_gb: float | None = None,
    max_instances: int = 8,
    cpu_offload: bool = True
) -> int:
    """Calculate optimal number of LightX2V pipeline instances.
    
    Determines how many concurrent LightX2V instances can fit in VRAM
    for parallel image editing.
    
    Args:
        available_vram_gb: Override for available VRAM (None = auto-detect)
        max_instances: Absolute cap on instance count
        cpu_offload: Whether CPU offload is enabled
        
    Returns:
        Optimal number of instances (minimum 1)
    """
    # Get available VRAM
    if available_vram_gb is None:
        vram_info = get_vram_info()
        available_vram_gb = vram_info.available_for_inference_gb
    
    # Calculate VRAM per instance
    base_model_gb = get_lightx2v_base_vram(cpu_offload)
    activation_overhead_gb = 4.0  # Per-image activation memory
    vram_per_instance = base_model_gb + activation_overhead_gb
    
    # Calculate optimal count
    optimal = int(available_vram_gb / vram_per_instance)
    
    # Apply bounds
    optimal = max(1, min(optimal, max_instances))
    
    logger.debug(
        f"LightX2V pool sizing: {available_vram_gb:.1f}GB available, "
        f"{vram_per_instance:.1f}GB per instance, optimal = {optimal}"
    )
    
    return optimal


# =============================================================================
# LTX-2 VRAM Estimation Functions
# =============================================================================

def estimate_ltx2_vram_per_video(
    width: int, 
    height: int, 
    num_frames: int
) -> float:
    """Estimate VRAM usage for a single LTX-2 video generation.
    
    LTX-2 uses a two-stage pipeline:
    1. Stage 1: Generate at half resolution
    2. Stage 2: Upsample 2x with distilled LoRA refinement
    
    VRAM scales with resolution (megapixels) and frame count.
    
    Args:
        width: Video width in pixels (final output)
        height: Video height in pixels (final output)
        num_frames: Number of frames to generate
        
    Returns:
        Estimated VRAM usage in GB for one video
    """
    # Calculate megapixels at full resolution
    megapixels = (width * height) / 1_000_000
    
    # Estimate: base overhead + per-megapixel + per-frame
    # The per-frame scaling is normalized to 1080p
    frame_scale = num_frames * (megapixels / 2.07)  # 2.07 MP = 1920x1080
    
    estimated_gb = (
        LTX2_BASE_ACTIVATION_GB +
        (megapixels * LTX2_GB_PER_MEGAPIXEL * num_frames) +
        (frame_scale * LTX2_GB_PER_FRAME)
    )
    
    return estimated_gb


def calculate_ltx2_max_batch_size(
    width: int = 1920,
    height: int = 1080,
    num_frames: int = 97,
    available_vram_gb: float | None = None,
) -> int:
    """Calculate maximum batch size for LTX-2 video generation.
    
    Currently always returns 1 for sequential batching. This function
    exists for API consistency and future extensibility if parallel
    batching becomes feasible.
    
    Args:
        width: Video width in pixels
        height: Video height in pixels
        num_frames: Number of frames
        available_vram_gb: Override for available VRAM (None = auto-detect)
        
    Returns:
        Maximum batch size (currently always 1)
    """
    # Sequential batching only - parallel video batching is not supported
    # due to VRAM constraints and architecture limitations
    logger.debug(
        f"LTX-2 batch size: sequential only (max={MAX_BATCH_SIZE_LTX2}), "
        f"resolution={width}x{height}, frames={num_frames}"
    )
    return MAX_BATCH_SIZE_LTX2


def get_ltx2_base_vram(fp8_enabled: bool = True) -> float:
    """Get the base VRAM footprint for LTX-2 model.
    
    Args:
        fp8_enabled: Whether FP8 quantization is enabled
        
    Returns:
        Base VRAM usage in GB for model weights
    """
    if fp8_enabled:
        return LTX2_BASE_MODEL_FP8_GB
    return LTX2_BASE_MODEL_FP16_GB


# =============================================================================
# LTX-2 Concurrent Video Generation Functions
# =============================================================================

def estimate_ltx2_concurrent_vram(
    duration_seconds: float,
    width: int = 1920,
    height: int = 1080,
    fps: float = 24.0,
) -> float:
    """Estimate VRAM usage for a single concurrent LTX-2 video generation.
    
    This calculates the activation memory needed for one video during
    concurrent generation with a shared pipeline.
    
    Args:
        duration_seconds: Video duration in seconds
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        
    Returns:
        Estimated VRAM usage in GB for one concurrent video
    """
    # Calculate frame count
    num_frames = int(duration_seconds * fps) + 1
    
    # Calculate megapixels at full resolution
    megapixels = (width * height) / 1_000_000
    
    # Normalize frame scaling to 1080p baseline (2.07 MP)
    frame_scale = num_frames * (megapixels / 2.07)
    
    # Base activation + per-megapixel + per-frame + concurrent overhead
    estimated_gb = (
        LTX2_BASE_ACTIVATION_GB +
        (megapixels * LTX2_GB_PER_MEGAPIXEL * num_frames) +
        (frame_scale * LTX2_GB_PER_FRAME) +
        LTX2_CONCURRENT_OVERHEAD_GB
    )
    
    return estimated_gb


def calculate_ltx2_max_concurrent(
    duration_seconds: float,
    width: int = 1920,
    height: int = 1080,
    fps: float = 24.0,
    available_vram_gb: float | None = None,
) -> int:
    """Calculate maximum concurrent LTX-2 video generations.
    
    Determines how many videos of the given duration can be generated
    concurrently based on available VRAM. Uses conservative estimates
    with safety margins.
    
    Args:
        duration_seconds: Video duration in seconds (use longest in batch)
        width: Video width in pixels
        height: Video height in pixels  
        fps: Frames per second
        available_vram_gb: Override for available VRAM (None = use default budget)
        
    Returns:
        Maximum safe concurrent video count (1 to LTX2_MAX_CONCURRENT_VIDEOS)
    """
    # Use configured budget if not specified
    if available_vram_gb is None:
        available_vram_gb = LTX2_CONCURRENT_VRAM_BUDGET_GB
    
    # Check minimum threshold
    if available_vram_gb < MIN_FREE_VRAM_GB:
        logger.warning(
            f"Low VRAM ({available_vram_gb:.1f}GB), limiting to 1 concurrent video"
        )
        return 1
    
    # Estimate VRAM per video at this duration
    vram_per_video = estimate_ltx2_concurrent_vram(
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        fps=fps,
    )
    
    if vram_per_video <= 0:
        logger.warning("Invalid VRAM estimation, defaulting to 1 concurrent")
        return 1
    
    # Calculate max concurrent with safety factor
    raw_max = available_vram_gb / vram_per_video
    safe_max = int(raw_max * LTX2_CONCURRENT_SAFETY_FACTOR)
    
    # Apply bounds
    max_concurrent = max(1, min(safe_max, LTX2_MAX_CONCURRENT_VIDEOS))
    
    logger.debug(
        f"LTX-2 concurrent: {available_vram_gb:.1f}GB available, "
        f"{vram_per_video:.1f}GB per {duration_seconds}s video, "
        f"max concurrent = {max_concurrent}"
    )
    
    return max_concurrent


def calculate_ltx2_batch_concurrency(
    durations: list[float],
    width: int = 1920,
    height: int = 1080,
    fps: float = 24.0,
    available_vram_gb: float | None = None,
) -> int:
    """Calculate optimal concurrency for a batch of videos with varying durations.
    
    Uses the LONGEST duration in the batch to ensure all videos can complete
    without OOM. This is conservative but safe.
    
    Args:
        durations: List of video durations in seconds
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        available_vram_gb: Override for available VRAM
        
    Returns:
        Maximum concurrent videos for this batch
    """
    if not durations:
        return 1
    
    # Use longest duration for conservative estimation
    max_duration = max(durations)
    
    max_concurrent = calculate_ltx2_max_concurrent(
        duration_seconds=max_duration,
        width=width,
        height=height,
        fps=fps,
        available_vram_gb=available_vram_gb,
    )
    
    logger.info(
        f"LTX-2 batch concurrency: {len(durations)} videos, "
        f"max duration = {max_duration:.1f}s, max concurrent = {max_concurrent}"
    )
    
    return max_concurrent

