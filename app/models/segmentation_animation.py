"""Pydantic models for segmentation animation API endpoint."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from app.models.segmentation import BoxPrompt, ObjectPrompt, SEGMENTATION_OPERATION_TYPES


class AnimationConfig(BaseModel):
    """Animation configuration for an operation.
    
    Controls how an effect's parameters change over time.
    All numeric parameters in the parent operation can be animated
    by specifying their start and end values.
    """
    mode: Literal["transition", "draw", "pulse", "reveal", "loop", "stagger"] = Field(
        "transition",
        description=(
            "Animation mode. "
            "'transition': interpolate start→end. "
            "'draw': progressive contour tracing (for outline/bounding_box). "
            "'pulse': oscillate min→max→min. "
            "'reveal': directional wipe. "
            "'loop': continuous start→end→start. "
            "'stagger': per-object delay offset."
        ),
    )
    start: Optional[dict] = Field(
        None,
        description="Parameter values at the start of animation (e.g. {\"strength\": 0})",
    )
    end: Optional[dict] = Field(
        None,
        description="Parameter values at the end of animation (e.g. {\"strength\": 25})",
    )
    easing: str = Field(
        "ease_out",
        description=(
            "Easing function name. Options: "
            "'linear', 'ease_in', 'ease_out', 'ease_in_out', "
            "'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic', "
            "'ease_out_back', 'ease_out_elastic', 'ease_out_bounce'"
        ),
    )
    delay: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Seconds to wait before animation starts",
    )
    duration: Optional[float] = Field(
        None,
        ge=0.1,
        le=10.0,
        description="Animation duration in seconds (defaults to total video duration)",
    )
    cycles: int = Field(
        1,
        ge=1,
        le=20,
        description="Number of oscillation cycles (used by 'pulse' and 'loop' modes)",
    )
    direction: Optional[Literal["left", "right", "top", "bottom", "radial", "clockwise"]] = Field(
        None,
        description="Reveal direction (used by 'reveal' mode only)",
    )
    stagger_delay: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Seconds between each object's animation start (used by 'stagger' mode)",
    )


class AnimateSegmentRequest(BaseModel):
    """Request body for animated image segmentation (Image→Video).
    
    Takes a single image, segments it using SAM 3, then generates a video
    with animated visual effects applied to the segmented regions.
    """
    job_id: str = Field(
        ...,
        description="Unique job identifier (UUID format recommended)",
    )
    input_image_url: str = Field(
        ...,
        description="URL of the image to segment and animate (PNG/JPEG/WebP)",
    )
    text_prompt: Optional[str] = Field(
        None,
        description="Text describing objects to segment (e.g., 'person', 'all cars')",
    )
    object_prompts: Optional[List[ObjectPrompt]] = Field(
        None,
        description="Named object prompts for per-object animation control. Each object label can be reused in "
                    "select operations via object_label.",
    )
    point_prompts: Optional[List[List[int]]] = Field(
        None,
        description="List of [x, y] click coordinates to prompt specific objects",
    )
    box_prompts: Optional[List[List[int]]] = Field(
        None,
        description="List of [x1, y1, x2, y2] bounding boxes (all positive)",
    )
    box_prompts_labeled: Optional[List[BoxPrompt]] = Field(
        None,
        description="List of box prompts with positive/negative labels",
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include an object (0.0-1.0)",
    )
    max_objects: int = Field(
        100,
        ge=1,
        le=500,
        description="Maximum number of objects to segment",
    )
    duration_seconds: float = Field(
        3.0,
        ge=0.5,
        le=10.0,
        description="Total animation duration in seconds",
    )
    fps: int = Field(
        30,
        ge=8,
        le=60,
        description="Frames per second of the output video",
    )
    operations: List[dict] = Field(
        ...,
        description=(
            "Ordered list of visual operations with optional animation configs. "
            "Each operation is a dict with 'type', params, and optional 'animation' key. "
            f"Supported types: {SEGMENTATION_OPERATION_TYPES}."
        ),
    )
    save_url: str = Field(
        ...,
        description="Pre-signed URL to upload the MP4 result",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="URL to receive webhook notification on completion",
    )
    webhook_secret: Optional[str] = Field(
        None,
        description="Secret for webhook HMAC signature",
    )
    item_id: Optional[str] = Field(
        None,
        description="Client-side identifier for this item",
    )
