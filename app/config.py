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
    LIGHTX2V_MODEL_FP8 = "models/qwen-image-edit-2511-fp8"  # FP8 quantized (~19GB vs ~38GB)
    LIGHTX2V_LORA = "models/loras/qwen-image-edit-2511"
    LIGHTX2V_LORA_FILE = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"
    
    # Multiple Angles LoRA (camera control for Qwen-Image-Edit)
    LIGHTX2V_MULTIPLE_ANGLES_LORA = "models/loras/qwen-image-edit-multiple-angles"
    LIGHTX2V_MULTIPLE_ANGLES_LORA_FILE = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
    
    # LTX-2 (video generation)
    LTX2_CHECKPOINT = "models/ltx-2/ltx-2-19b-distilled-fp8.safetensors"
    LTX2_SPATIAL_UPSAMPLER = "models/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    LTX2_GEMMA_ROOT = "models/ltx-2/gemma-3-12b-it-qat"
    LTX2_DISTILLED_LORA = "models/ltx-2/ltx-2-19b-distilled-lora-384.safetensors"
    
    # Stream-DiffVSR (video upscaling)
    STREAM_DIFFVSR_MODEL_ID = "Jamichsu/Stream-DiffVSR"
    
    # ACE-Step 1.5 (music generation)
    ACESTEP_MODEL = "models/ace-step-1.5"
    
    # AudioGen (sound effect generation)
    AUDIOGEN_MODEL = "models/audiogen-medium"


class InferenceConfig:
    """Hardcoded inference parameters."""
    
    # Device settings
    DEVICE = "cuda"
    DTYPE: Literal["bfloat16", "float16"] = "bfloat16"
    
    # Concurrency limits
    MAX_CONCURRENT_IMAGE_GENERATIONS = 2  # Across Z-Image + Qwen-Image-Edit
    MAX_CONCURRENT_VIDEO_GENERATIONS = 3  # LTX-2 (QAT text encoder allows 3 concurrent in video-only mode)
    
    # Z-Image settings (uses Diffusers ZImagePipeline)
    ZIMAGE_COMPILE = False  # Disabled: torch.compile inductor fails with dynamic shapes
    # Diffusers uses SDPA by default; options: "sdpa", "flash", "_flash_3"
    ZIMAGE_ATTENTION_BACKEND = "sdpa"
    
    # LightX2V settings
    LIGHTX2V_ATTN_MODE = "torch_sdpa"
    LIGHTX2V_INFER_STEPS = 8
    LIGHTX2V_GUIDANCE_SCALE = 1.0
    LIGHTX2V_LORA_STRENGTH = 1.0
    LIGHTX2V_RESIZE_MODE = "adaptive"
    LIGHTX2V_CPU_OFFLOAD = False
    LIGHTX2V_TEXT_ENCODER_OFFLOAD = True
    LIGHTX2V_FP8_ENABLED = True  # Use FP8 quantized model (~19GB vs ~38GB)
    LIGHTX2V_FP8_QUANT_SCHEME = "fp8-sgl"  # FP8 single-element scaling
    
    # Dynamic LoRA settings (for Multiple Angles LoRA, etc.)
    LIGHTX2V_DEFAULT_LORA_STRENGTH = 0.9  # Recommended for Multiple Angles LoRA
    LIGHTX2V_DYNAMIC_LORA_ENABLED = True  # Enable per-request LoRA switching
    # LightX2V instance counts per mode (FP8: ~19GB per instance)
    LIGHTX2V_MAX_INSTANCES_ALL = 1       # Conservative when sharing VRAM with Z-Image + LTX-2
    LIGHTX2V_MAX_INSTANCES_DEDICATED = 2  # 2 concurrent instances (2 x ~19GB = ~38GB)
    
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
    LTX2_MAX_CONCURRENT_VIDEOS = 3  # 3 concurrent videos in video-only mode (QAT text encoder)
    LTX2_CONCURRENT_VRAM_BUDGET_GB = 72.0  # VRAM available for activations (after base model)
    
    # ACE-Step (music) settings
    ACESTEP_DEFAULT_DURATION = 30.0   # seconds
    ACESTEP_MAX_DURATION = 600.0      # 10 minutes max
    ACESTEP_SAMPLE_RATE = 48000       # 48kHz output (ACE-Step 1.5 native)
    
    # AudioGen (sound effects) settings
    AUDIOGEN_DEFAULT_DURATION = 5.0   # seconds
    AUDIOGEN_MAX_DURATION = 30.0      # 30 seconds max
    AUDIOGEN_SAMPLE_RATE = 16000      # 16kHz output (AudioGen native)
    
    # Job timeouts (seconds)
    IMAGE_JOB_TIMEOUT = 300      # 5 minutes for image batch jobs (large batches at 1920x1080)
    VIDEO_JOB_TIMEOUT = 600      # 10 minutes for video jobs
    AUDIO_JOB_TIMEOUT = 600      # 10 minutes for audio jobs
    
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
    audio_dry_run_override: Optional[bool] = None

    # Optimization settings
    ltx2_use_fp8_text_encoder: bool = True  # Use FP8 quantized text encoder

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
    
    @property
    def acestep_model_path(self) -> str:
        return ModelPaths.ACESTEP_MODEL
    
    @property
    def audiogen_model_path(self) -> str:
        return ModelPaths.AUDIOGEN_MODEL
    
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
    def lightx2v_fp8_enabled(self) -> bool:
        """Whether to use FP8 quantized model (~19GB vs ~38GB)."""
        return InferenceConfig.LIGHTX2V_FP8_ENABLED
    
    @property
    def lightx2v_fp8_model_path(self) -> str:
        """Path to FP8 quantized model."""
        return ModelPaths.LIGHTX2V_MODEL_FP8
    
    @property
    def lightx2v_fp8_quant_scheme(self) -> str:
        """FP8 quantization scheme."""
        return InferenceConfig.LIGHTX2V_FP8_QUANT_SCHEME
    
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
    
    # --- LightX2V Multiple Angles LoRA ---
    @property
    def lightx2v_multiple_angles_lora_path(self) -> str:
        """Directory containing the Multiple Angles LoRA."""
        return ModelPaths.LIGHTX2V_MULTIPLE_ANGLES_LORA
    
    @property
    def lightx2v_multiple_angles_lora_filename(self) -> str:
        """Filename of the Multiple Angles LoRA safetensors."""
        return ModelPaths.LIGHTX2V_MULTIPLE_ANGLES_LORA_FILE
    
    @property
    def lightx2v_default_dynamic_lora_strength(self) -> float:
        """Default strength when applying dynamic LoRAs (e.g., Multiple Angles)."""
        return InferenceConfig.LIGHTX2V_DEFAULT_LORA_STRENGTH
    
    @property
    def lightx2v_dynamic_lora_enabled(self) -> bool:
        """Whether dynamic per-request LoRA switching is enabled."""
        return InferenceConfig.LIGHTX2V_DYNAMIC_LORA_ENABLED
    
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
    
    # --- Audio Generation ---
    @property
    def acestep_default_duration(self) -> float:
        return InferenceConfig.ACESTEP_DEFAULT_DURATION
    
    @property
    def acestep_max_duration(self) -> float:
        return InferenceConfig.ACESTEP_MAX_DURATION
    
    @property
    def acestep_sample_rate(self) -> int:
        return InferenceConfig.ACESTEP_SAMPLE_RATE
    
    @property
    def audiogen_default_duration(self) -> float:
        return InferenceConfig.AUDIOGEN_DEFAULT_DURATION
    
    @property
    def audiogen_max_duration(self) -> float:
        return InferenceConfig.AUDIOGEN_MAX_DURATION
    
    @property
    def audiogen_sample_rate(self) -> int:
        return InferenceConfig.AUDIOGEN_SAMPLE_RATE
    
    @property
    def audio_job_timeout(self) -> int:
        return InferenceConfig.AUDIO_JOB_TIMEOUT
    
    @property
    def audio_dry_run(self) -> bool:
        """In mock_mode, all generators run in dry-run mode."""
        if self.audio_dry_run_override is not None:
            return self.audio_dry_run_override
        return self.mock_mode


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
