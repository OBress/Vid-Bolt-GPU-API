"""Pydantic models for segmentation API endpoints."""

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class BoxPrompt(BaseModel):
    """A box prompt with optional positive/negative label."""
    box: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box in [x1, y1, x2, y2] pixel coordinates",
    )
    label: bool = Field(
        True,
        description="True = include this region (positive), False = exclude this region (negative)",
    )


class ImageSegmentRequest(BaseModel):
    """Request body for image segmentation."""
    job_id: str = Field(
        ...,
        description="Unique job identifier (UUID format recommended)",
    )
    input_image_url: str = Field(
        ...,
        description="URL of the image to segment (PNG/JPEG/WebP)",
    )
    text_prompt: Optional[str] = Field(
        None,
        description="Text describing objects to segment (e.g., 'all cars', 'person in red')",
    )
    point_prompts: Optional[List[List[int]]] = Field(
        None,
        description="List of [x, y] click coordinates to prompt specific objects",
    )
    box_prompts: Optional[List[List[int]]] = Field(
        None,
        description="List of [x1, y1, x2, y2] bounding boxes (simple format, all positive)",
    )
    box_prompts_labeled: Optional[List[BoxPrompt]] = Field(
        None,
        description="List of box prompts with positive/negative labels for include/exclude regions",
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
    output_type: Literal["masks_json", "image"] = Field(
        "masks_json",
        description="'masks_json' returns raw mask data, 'image' applies operations and returns processed image",
    )
    operations: Optional[List[dict]] = Field(
        None,
        description="Ordered list of visual operations to apply (only used when output_type='image'). "
                    "Each operation is a dict with 'type' and params. Types: select, blur, pixelate, redact, "
                    "color_overlay, color_grade, opacity, replace_color, remove_background, replace_background, "
                    "greenscreen, outline, text_label, bounding_box, spotlight, bokeh, glow, shadow, vignette",
    )
    save_url: str = Field(
        ...,
        description="Pre-signed URL to upload the result (JSON for masks_json, PNG/JPEG for image)",
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


class VideoSegmentRequest(BaseModel):
    """Request body for video segmentation/tracking."""
    job_id: str = Field(
        ...,
        description="Unique job identifier (UUID format recommended)",
    )
    input_video_url: str = Field(
        ...,
        description="URL of the video to segment (MP4)",
    )
    text_prompt: Optional[str] = Field(
        None,
        description="Text describing objects to track (e.g., 'yellow school bus')",
    )
    point_prompts: Optional[List[List[float]]] = Field(
        None,
        description="List of [x, y] pixel coordinates for point prompts on the initial frame",
    )
    point_labels: Optional[List[int]] = Field(
        None,
        description="Labels for each point prompt: 1 = positive (include), 0 = negative (exclude)",
    )
    box_prompts: Optional[List[List[float]]] = Field(
        None,
        description="List of [x, y, w, h] bounding boxes for initial frame prompts",
    )
    box_labels: Optional[List[int]] = Field(
        None,
        description="Labels for each box prompt: 1 = positive, 0 = negative",
    )
    prompt_frame_index: int = Field(
        0,
        ge=0,
        description="Frame index to apply prompts on (default: first frame)",
    )
    propagation_direction: Literal["forward", "backward", "both"] = Field(
        "forward",
        description="Direction to propagate tracking: 'forward', 'backward', or 'both'",
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for object detection",
    )
    output_format: Literal["masks_json", "video"] = Field(
        "masks_json",
        description="'masks_json' returns raw mask data, 'video' applies operations and returns processed MP4",
    )
    operations: Optional[List[dict]] = Field(
        None,
        description="Ordered list of visual operations to apply per-frame (only used when output_format='video'). "
                    "Same operation types as image segmentation.",
    )
    max_frames: int = Field(
        300,
        ge=1,
        le=1000,
        description="Maximum number of frames to process",
    )
    save_url: str = Field(
        ...,
        description="Pre-signed URL to upload the result (JSON for masks_json, MP4 for video)",
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
