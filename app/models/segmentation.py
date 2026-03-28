"""Pydantic models for segmentation API endpoints."""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


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
        description="List of [x1, y1, x2, y2] bounding boxes to prompt specific regions",
    )
    max_objects: int = Field(
        100,
        ge=1,
        le=500,
        description="Maximum number of objects to segment",
    )
    save_url: str = Field(
        ...,
        description="Pre-signed URL to upload the segmentation result JSON",
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
    text_prompt: str = Field(
        ...,
        description="Text describing objects to track (e.g., 'yellow school bus')",
    )
    output_format: str = Field(
        "masks_json",
        description="Output format: 'masks_json' for per-frame mask data",
    )
    max_frames: int = Field(
        300,
        ge=1,
        le=1000,
        description="Maximum number of frames to process",
    )
    save_url: str = Field(
        ...,
        description="Pre-signed URL to upload the segmentation result",
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
