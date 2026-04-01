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


# --- Segmentation ---

@dataclass
class ImageSegmentationParams:
    """Parameters for image segmentation (SAM 3)."""
    job_id: str
    input_image_data: bytes
    text_prompt: Optional[str] = None         # e.g., "all cars", "person in red"
    point_prompts: Optional[List[Tuple[int, int]]] = None  # [(x, y)] click coords
    box_prompts: Optional[List[Tuple[int, int, int, int]]] = None  # [(x1,y1,x2,y2)] all positive
    box_prompts_labeled: Optional[List[Tuple[Tuple[int, int, int, int], bool]]] = None  # [((x1,y1,x2,y2), label)]
    confidence_threshold: float = 0.5
    max_objects: int = 100

@dataclass
class ImageSegmentationResult:
    """Result of image segmentation."""
    masks_data: bytes             # JSON-encoded list of base64 PNG masks
    boxes: List[Tuple[int, int, int, int]]  # Bounding boxes per object
    scores: List[float]           # Confidence scores per object
    object_count: int
    width: int
    height: int

@dataclass
class VideoSegmentationParams:
    """Parameters for video segmentation/tracking (SAM 3)."""
    job_id: str
    input_video_data: bytes       # MP4 video bytes
    text_prompt: Optional[str] = None         # Text concept to track
    point_prompts: Optional[List[List[float]]] = None  # [[x, y]] pixel coords
    point_labels: Optional[List[int]] = None  # 1 = positive, 0 = negative
    box_prompts: Optional[List[List[float]]] = None  # [[x, y, w, h]] bounding boxes
    box_labels: Optional[List[int]] = None    # 1 = positive, 0 = negative
    prompt_frame_index: int = 0   # Frame to apply prompts on
    propagation_direction: str = "forward"  # "forward", "backward", "both"
    confidence_threshold: float = 0.5
    output_format: str = "masks_json"  # "masks_json" or "overlay_video"
    max_frames: int = 300         # Max frames to process

@dataclass
class VideoSegmentationResult:
    """Result of video segmentation."""
    result_data: bytes            # JSON masks or overlay video MP4
    output_format: str
    frame_count: int
    object_count: int
    tracked_ids: List[int]        # Unique IDs for tracked objects

