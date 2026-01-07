"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Mock Mode
    mock_mode: bool = True

    # API Authentication
    api_key: str = ""

    # ComfyUI Settings (future use)
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188

    # Limits
    max_image_size_mb: int = 10
    max_video_duration_seconds: int = 8

    # Logging
    log_level: str = "INFO"

    # CORS - comma-separated list of allowed origins
    cors_allowed_origins: str = "http://localhost:3000"

    # Z-Image Settings
    zimage_model_path: str = "models/z-image-turbo"
    zimage_lora_path: str = "models/loras"
    zimage_device: str = "cuda"
    zimage_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    zimage_compile: bool = False  # torch.compile for faster inference after warmup
    zimage_attention_backend: str = "_native_flash"  # flash, _flash_3, sdpa, _native_flash
    zimage_dry_run: bool = False  # Test workflow without loading models

    # LightX2V Settings (Qwen-Image-Edit-2511)
    lightx2v_model_path: str = "models/qwen-image-edit-2511"
    lightx2v_lora_path: str = "models/loras/qwen-image-edit-2511"
    lightx2v_lora_filename: str = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"
    lightx2v_device: str = "cuda"
    lightx2v_attn_mode: str = "flash_attn3"  # flash_attn2, flash_attn3, sage_attn2
    lightx2v_infer_steps: int = 8  # 8-step with LORA, 40 for base
    lightx2v_guidance_scale: float = 1.0  # CFG disabled with distill LORA
    lightx2v_resize_mode: str = "adaptive"
    lightx2v_dry_run: bool = False  # Test workflow without loading models
    lightx2v_cpu_offload: bool = False  # Enable for lower VRAM usage
    lightx2v_text_encoder_offload: bool = True  # Offload text encoder to CPU

    # LTX-2 Video Generation Settings
    ltx2_checkpoint_path: str = "models/ltx-2/ltx-2-19b-dev.safetensors"
    ltx2_distilled_lora_path: str = "models/ltx-2/ltx-2-19b-distilled-lora-384.safetensors"
    ltx2_spatial_upsampler_path: str = "models/ltx-2/ltx-2-spatial-upsampler-x2-1.0.safetensors"
    ltx2_gemma_root: str = "models/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized"
    ltx2_device: str = "cuda"
    ltx2_fp8_enabled: bool = False  # Enable FP8 for lower VRAM usage (~16GB instead of 24GB)
    ltx2_dry_run: bool = False  # Test workflow without loading models
    ltx2_num_inference_steps: int = 40  # Stage 1 denoising steps
    ltx2_cfg_guidance_scale: float = 4.0  # CFG scale for stage 1
    ltx2_default_frame_rate: float = 24.0  # Default FPS

    @property
    def max_image_size_bytes(self) -> int:
        """Get max image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

