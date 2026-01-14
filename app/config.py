"""Application configuration.

Environment variables (from .env):
- MOCK_MODE: Enable mock mode for testing (default: true)
- API_KEY: API authentication key (required)
- LOG_LEVEL: Logging level (default: INFO)
- CORS_ALLOWED_ORIGINS: Comma-separated allowed origins (default: http://localhost:3000)
- PORT: Server port (default: 8000) - used by uvicorn, not this config

All model paths, inference parameters, and hardware settings are hardcoded
as sensible defaults below. Override them by editing this file directly
if needed for your specific deployment.
"""

from functools import lru_cache
from typing import Literal, Optional

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Hardcoded Model Configuration (not in .env)
# =============================================================================

class ModelPaths:
    """Hardcoded model paths relative to project root."""
    
    # Z-Image Turbo (text-to-image)
    ZIMAGE_MODEL = "models/z-image-turbo"
    ZIMAGE_LORA = "models/loras/z-image"
    
    # LightX2V / Qwen-Image-Edit (image editing)
    LIGHTX2V_MODEL = "models/qwen-image-edit-2511"
    LIGHTX2V_LORA = "models/loras/qwen-image-edit-2511"
    LIGHTX2V_LORA_FILE = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"
    
    # LTX-2 (video generation)
    LTX2_CHECKPOINT = "models/ltx-2/ltx-2-19b-distilled-fp8.safetensors"
    LTX2_SPATIAL_UPSAMPLER = "models/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    LTX2_GEMMA_ROOT = "models/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized"
    LTX2_DISTILLED_LORA = "models/ltx-2/ltx-2-19b-distilled-lora-384.safetensors"
    
    # Stream-DiffVSR (video upscaling)
    STREAM_DIFFVSR_MODEL_ID = "Jamichsu/Stream-DiffVSR"


class InferenceConfig:
    """Hardcoded inference parameters."""
    
    # Device settings
    DEVICE = "cuda"
    DTYPE: Literal["bfloat16", "float16"] = "bfloat16"
    
    # Concurrency limits
    MAX_CONCURRENT_IMAGE_GENERATIONS = 2  # Across Z-Image + Qwen-Image-Edit
    MAX_CONCURRENT_VIDEO_GENERATIONS = 1  # LTX-2 + Stream-DiffVSR workflow
    
    # Z-Image settings (uses Diffusers ZImagePipeline)
    ZIMAGE_COMPILE = False
    # Diffusers uses SDPA by default; options: "sdpa", "flash", "_flash_3"
    ZIMAGE_ATTENTION_BACKEND = "sdpa"
    # Z-Image instance counts for concurrent pool (each instance ~8GB + 3GB activation)
    ZIMAGE_MAX_INSTANCES_ALL = 2         # Conservative when sharing VRAM with LightX2V + LTX-2
    ZIMAGE_MAX_INSTANCES_DEDICATED = 8   # Full utilization when Z-Image-only mode (96GB GPU)
    
    # LightX2V settings
    LIGHTX2V_ATTN_MODE = "torch_sdpa"
    LIGHTX2V_INFER_STEPS = 8
    LIGHTX2V_GUIDANCE_SCALE = 1.0
    LIGHTX2V_LORA_STRENGTH = 1.0
    LIGHTX2V_RESIZE_MODE = "adaptive"
    LIGHTX2V_CPU_OFFLOAD = False
    LIGHTX2V_TEXT_ENCODER_OFFLOAD = True
    # LightX2V instance counts per mode (each instance ~16GB VRAM)
    LIGHTX2V_MAX_INSTANCES_ALL = 1       # Conservative when sharing VRAM with Z-Image + LTX-2
    LIGHTX2V_MAX_INSTANCES_DEDICATED = 5  # Full utilization when LightX2V-only mode
    
    # LTX-2 settings (Optimized for Distilled Model)
    # Distilled model uses 8 predefined sigmas (Stage 1: 8 steps, Stage 2: 4 steps)
    # See: LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
    LTX2_FP8_ENABLED = True  # FP8 for faster inference, ~20GB VRAM
    LTX2_NUM_INFERENCE_STEPS = 8  # Distilled model uses 8 predefined sigma values
    LTX2_CFG_GUIDANCE_SCALE = 1.0  # Distilled LoRA works best without CFG (1.0)
    LTX2_DEFAULT_FRAME_RATE = 24.0
    
    # LTX-2 Concurrent Generation Settings
    # Enables parallel video generation using shared pipeline (stateless architecture)
    LTX2_CONCURRENT_ENABLED = True  # Enable concurrent video generation
    LTX2_MAX_CONCURRENT_VIDEOS = 4  # Absolute cap on concurrent videos
    LTX2_CONCURRENT_VRAM_BUDGET_GB = 72.0  # VRAM available for activations (after base model)
    
    # Job timeouts (seconds)
    IMAGE_JOB_TIMEOUT = 120      # 2 minutes for image jobs
    VIDEO_JOB_TIMEOUT = 600      # 10 minutes for video jobs
    
    # Limits
    MAX_IMAGE_SIZE_MB = 10
    MAX_VIDEO_DURATION_SECONDS = 10


# =============================================================================
# Environment-based Settings (from .env)
# =============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    Only runtime/deployment settings are loaded from .env:
    - mock_mode: Run with mock generators (no GPU needed)
    - api_key: Authentication key
    - log_level: Logging verbosity
    - cors_allowed_origins: CORS configuration
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Runtime Settings (from .env)
    mock_mode: bool = True
    api_key: str = ""
    log_level: str = "INFO"
    cors_allowed_origins: str = "http://localhost:3000"
    
    # Model mode (default startup mode)
    default_model_mode: Literal["image", "video"] = "image"
    
    # External service tokens (optional)
    github_access_token: str = ""
    hf_token: str = ""

    # Test Overrides (hidden from .env, used for testing real generators without weights)
    zimage_dry_run_override: Optional[bool] = None
    lightx2v_dry_run_override: Optional[bool] = None
    ltx2_dry_run_override: Optional[bool] = None

    # ==========================================================================
    # Computed properties exposing hardcoded config
    # ==========================================================================
    
    # --- Model Paths ---
    @property
    def zimage_model_path(self) -> str:
        return ModelPaths.ZIMAGE_MODEL
    
    @property
    def zimage_lora_path(self) -> str:
        return ModelPaths.ZIMAGE_LORA
    
    @property
    def lightx2v_model_path(self) -> str:
        return ModelPaths.LIGHTX2V_MODEL
    
    @property
    def lightx2v_lora_path(self) -> str:
        return ModelPaths.LIGHTX2V_LORA
    
    @property
    def lightx2v_lora_filename(self) -> str:
        return ModelPaths.LIGHTX2V_LORA_FILE
    
    @property
    def ltx2_checkpoint_path(self) -> str:
        return ModelPaths.LTX2_CHECKPOINT
    
    @property
    def ltx2_spatial_upsampler_path(self) -> str:
        return ModelPaths.LTX2_SPATIAL_UPSAMPLER
    
    @property
    def ltx2_gemma_root(self) -> str:
        return ModelPaths.LTX2_GEMMA_ROOT
    
    @property
    def ltx2_distilled_lora_path(self) -> str:
        return ModelPaths.LTX2_DISTILLED_LORA
    
    @property
    def stream_diffvsr_model_id(self) -> str:
        return ModelPaths.STREAM_DIFFVSR_MODEL_ID
    
    # --- Device & Dtype ---
    @property
    def zimage_device(self) -> str:
        return InferenceConfig.DEVICE
    
    @property
    def zimage_dtype(self) -> Literal["bfloat16", "float16"]:
        return InferenceConfig.DTYPE
    
    @property
    def lightx2v_device(self) -> str:
        return InferenceConfig.DEVICE
    
    @property
    def ltx2_device(self) -> str:
        return InferenceConfig.DEVICE
    
    @property
    def stream_diffvsr_device(self) -> str:
        return InferenceConfig.DEVICE
    
    # --- Z-Image ---
    @property
    def zimage_compile(self) -> bool:
        return InferenceConfig.ZIMAGE_COMPILE
    
    @property
    def zimage_attention_backend(self) -> str:
        return InferenceConfig.ZIMAGE_ATTENTION_BACKEND
    
    @property
    def zimage_dry_run(self) -> bool:
        """In mock_mode, all generators run in dry-run mode."""
        if self.zimage_dry_run_override is not None:
            return self.zimage_dry_run_override
        return self.mock_mode
    
    @property
    def zimage_max_instances_all(self) -> int:
        """Max Z-Image instances when in ALL mode (sharing VRAM)."""
        return InferenceConfig.ZIMAGE_MAX_INSTANCES_ALL
    
    @property
    def zimage_max_instances_dedicated(self) -> int:
        """Max Z-Image instances when in dedicated IMAGE_GENERATION mode."""
        return InferenceConfig.ZIMAGE_MAX_INSTANCES_DEDICATED
    
    # --- LightX2V ---
    @property
    def lightx2v_attn_mode(self) -> str:
        return InferenceConfig.LIGHTX2V_ATTN_MODE
    
    @property
    def lightx2v_infer_steps(self) -> int:
        return InferenceConfig.LIGHTX2V_INFER_STEPS
    
    @property
    def lightx2v_guidance_scale(self) -> float:
        return InferenceConfig.LIGHTX2V_GUIDANCE_SCALE
    
    @property
    def lightx2v_lora_strength(self) -> float:
        return InferenceConfig.LIGHTX2V_LORA_STRENGTH
    
    @property
    def lightx2v_resize_mode(self) -> str:
        return InferenceConfig.LIGHTX2V_RESIZE_MODE
    
    @property
    def lightx2v_cpu_offload(self) -> bool:
        return InferenceConfig.LIGHTX2V_CPU_OFFLOAD
    
    @property
    def lightx2v_text_encoder_offload(self) -> bool:
        return InferenceConfig.LIGHTX2V_TEXT_ENCODER_OFFLOAD
    
    @property
    def lightx2v_dry_run(self) -> bool:
        """In mock_mode, all generators run in dry-run mode."""
        if self.lightx2v_dry_run_override is not None:
            return self.lightx2v_dry_run_override
        return self.mock_mode
    
    @property
    def lightx2v_max_instances_all(self) -> int:
        """Max LightX2V instances when in ALL mode (sharing VRAM)."""
        return InferenceConfig.LIGHTX2V_MAX_INSTANCES_ALL
    
    @property
    def lightx2v_max_instances_dedicated(self) -> int:
        """Max LightX2V instances when in dedicated IMAGE_EDITING mode."""
        return InferenceConfig.LIGHTX2V_MAX_INSTANCES_DEDICATED
    
    # --- LTX-2 ---
    @property
    def ltx2_fp8_enabled(self) -> bool:
        return InferenceConfig.LTX2_FP8_ENABLED
    
    @property
    def ltx2_num_inference_steps(self) -> int:
        return InferenceConfig.LTX2_NUM_INFERENCE_STEPS
    
    @property
    def ltx2_cfg_guidance_scale(self) -> float:
        return InferenceConfig.LTX2_CFG_GUIDANCE_SCALE
    
    @property
    def ltx2_default_frame_rate(self) -> float:
        return InferenceConfig.LTX2_DEFAULT_FRAME_RATE
    
    @property
    def ltx2_dry_run(self) -> bool:
        """In mock_mode, all generators run in dry-run mode."""
        if self.ltx2_dry_run_override is not None:
            return self.ltx2_dry_run_override
        return self.mock_mode
    
    @property
    def ltx2_concurrent_enabled(self) -> bool:
        """Whether concurrent video generation is enabled."""
        return InferenceConfig.LTX2_CONCURRENT_ENABLED
    
    @property
    def ltx2_max_concurrent_videos(self) -> int:
        """Maximum concurrent video generations."""
        return InferenceConfig.LTX2_MAX_CONCURRENT_VIDEOS
    
    @property
    def ltx2_concurrent_vram_budget_gb(self) -> float:
        """VRAM budget for concurrent video activations."""
        return InferenceConfig.LTX2_CONCURRENT_VRAM_BUDGET_GB
    
    # --- Limits ---
    @property
    def max_image_size_mb(self) -> int:
        return InferenceConfig.MAX_IMAGE_SIZE_MB
    
    @property
    def max_video_duration_seconds(self) -> int:
        return InferenceConfig.MAX_VIDEO_DURATION_SECONDS
    
    @property
    def max_image_size_bytes(self) -> int:
        """Get max image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024
    
    # --- Concurrency ---
    @property
    def max_concurrent_image_generations(self) -> int:
        """Max concurrent image generations (Z-Image + Qwen-Image-Edit combined)."""
        return InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS
    
    @property
    def max_concurrent_video_generations(self) -> int:
        """Max concurrent video generations (LTX-2 + upscaling workflow)."""
        return InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
