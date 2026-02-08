"""Internal data models for generation services.

These dataclasses define the internal API between the API layer and the Model layer.
They are distinct from the Pydantic models used for the public HTTP API.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

@dataclass
class ImageGenerationParams:
    """Parameters for image generation."""
    job_id: str
    prompt: str
    width: int
    height: int
    seed: Optional[int]
    num_inference_steps: int
    lora_name: Optional[str] = None

@dataclass
class ImageGenerationResult:
    """Result of image generation."""
    image_data: bytes
    width: int
    height: int
    seed: int

@dataclass
class ImageEditParams:
    """Parameters for image editing."""
    job_id: str
    input_image_data: bytes
    prompt: str
    width: int
    height: int
    mask_data: Optional[bytes]
    seed: Optional[int]
    # Dynamic LoRA support (e.g., Multiple Angles LoRA)
    lora_name: Optional[str] = None  # e.g., "multiple-angles"
    lora_strength: Optional[float] = None  # 0.0-1.0, default from config

@dataclass
class ImageEditResult:
    """Result of image editing."""
    image_data: bytes
    original_width: int
    original_height: int
    width: int
    height: int
    seed: int

@dataclass
class VideoGenerationParams:
    """Parameters for standard I2V video generation."""
    job_id: str
    prompt: str
    negative_prompt: str
    start_frame_data: bytes
    end_frame_data: Optional[bytes]
    duration_seconds: float
    frame_rate: float
    width: int
    height: int
    seed: Optional[int]
    enhance_prompt: bool = False

@dataclass
class VideoGenerationResult:
    """Result of video generation."""
    video_data: bytes
    width: int
    height: int
    duration_seconds: float
    frame_rate: float
    has_audio: bool
    seed: int
    upscale_info: Optional[dict[str, Any]] = None

@dataclass
class KeyframeInterpolationParams:
    """Parameters for keyframe interpolation video generation."""
    job_id: str
    prompt: str
    negative_prompt: str
    keyframes: List[Tuple[bytes, int, float]]  # (image_data, frame_idx, strength)
    duration_seconds: float
    frame_rate: float
    width: int
    height: int
    seed: Optional[int]
    enhance_prompt: bool = False

@dataclass
class UpscaleParams:
    """Parameters for video upscaling."""
    job_id: str
    video_data: bytes
    preserve_audio: bool = True

@dataclass
class UpscaleResult:
    """Result of video upscaling."""
    video_data: bytes
    original_width: int
    original_height: int
    upscaled_width: int
    upscaled_height: int
    frame_count: int
    processing_time_seconds: float
    was_upscaled: bool


# --- Audio Generation ---

@dataclass
class MusicGenerationParams:
    """Parameters for music generation (ACE-Step 1.5)."""
    job_id: str
    prompt: str  # Style/genre description (maps to ACE-Step "caption")
    lyrics: Optional[str]  # Optional lyrics for vocal generation
    duration_seconds: float
    seed: Optional[int]
    # ACE-Step 1.5 metadata (optional — auto-detected via LM if omitted)
    bpm: Optional[int] = None  # Tempo (30-300)
    key_scale: Optional[str] = None  # e.g. "C Major", "Am"
    time_signature: Optional[str] = None  # "2","3","4","6" for 2/4, 3/4, 4/4, 6/8
    vocal_language: Optional[str] = None  # ISO 639-1 code, e.g. "en", "zh", "ja"


@dataclass
class MusicGenerationResult:
    """Result of music generation."""
    audio_data: bytes
    duration_seconds: float
    sample_rate: int
    seed: int

