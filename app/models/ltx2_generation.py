"""LTX-2 video generation request and response models.

This module contains Pydantic models for both I2V (image-to-video) generation
and keyframe interpolation endpoints using the LTX-2 model.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.common import AspectRatio


def round_up_to_valid_frames(requested_frames: int) -> int:
    """Round up to the nearest valid LTX-2 frame count (8k+1 pattern).
    
    LTX-2 requires frame counts that follow the pattern: frames = 8k + 1
    where k is a non-negative integer (e.g., 9, 17, 25, 33, ..., 121, 241).
    
    Args:
        requested_frames: The requested number of frames
        
    Returns:
        The nearest valid frame count >= requested_frames
    """
    if requested_frames < 9:
        return 9
    remainder = (requested_frames - 1) % 8
    if remainder == 0:
        return requested_frames
    return requested_frames + (8 - remainder)


# ============================================================================
# Standard I2V (Image-to-Video) Request/Response
# ============================================================================

class LTX2GenerateRequest(BaseModel):
    """Request body for standard LTX-2 image-to-video generation."""

    job_id: str = Field(..., description="Unique job identifier")
    input_image_url: str = Field(..., description="URL of the start frame image")
    prompt: str = Field(
        ..., 
        description="Description of the video content/motion", 
        max_length=2000
    )
    negative_prompt: str = Field(
        default="",
        description="What should not appear in the video",
        max_length=1000
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=10.0,
        description="Target video duration in seconds (will be trimmed to exact length)"
    )
    frame_rate: float = Field(
        default=24.0,
        ge=8.0,
        le=60.0,
        description="Frame rate (FPS)"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the generated video"
    )
    width: int | None = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target width (overrides aspect_ratio default). Use 1920 for 1080p."
    )
    height: int | None = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target height (overrides aspect_ratio default). Use 1080 for 1080p."
    )
    end_image_url: str | None = Field(
        default=None,
        description="Optional URL of the end frame image for interpolation"
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    enhance_prompt: bool = Field(
        default=False,
        description="Auto-enhance the prompt for better results"
    )
    save_url: str = Field(..., description="Presigned URL (PUT) for direct storage upload")
    webhook_url: str = Field(
        ...,
        description="REQUIRED: URL to POST when generation completes (success or failure)",
    )
    item_id: str | None = Field(
        default=None,
        description="Client identifier for this item (returned in webhook, defaults to job_id)",
    )
    webhook_secret: str | None = Field(
        default=None,
        description="Secret for signing webhook payload (HMAC-SHA256)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440003",
                    "input_image_url": "https://example.com/start.png",
                    "prompt": "Gentle waves on the beach, cinematic motion",
                    "duration_seconds": 5.0,
                    "frame_rate": 24.0,
                    "save_url": "https://example.com/upload/video.mp4",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                }
            ]
        }
    }


class LTX2GenerateResponse(BaseModel):
    """Response for successful LTX-2 video generation."""

    status: Literal["completed"] = "completed"
    generation_time: float = Field(..., description="Processing time in seconds")
    save_url: str = Field(..., description="The URL where the video was saved")
    duration_seconds: float = Field(..., description="Final video duration in seconds")
    has_audio: bool = Field(default=True, description="Whether the video contains audio")
    upscale_info: dict[str, Any] | None = Field(
        default=None,
        description="Upscaling metadata if video was upscaled (720p→1080p)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "generation_time": 45.2,
                    "save_url": "https://example.com/upload/video.mp4",
                    "duration_seconds": 5.0,
                    "has_audio": True,
                    "upscale_info": {
                        "original_resolution": "1280x720",
                        "upscaled_resolution": "1920x1080",
                        "upscale_time_seconds": 8.5,
                        "was_upscaled": True
                    }
                }
            ]
        }
    }


# ============================================================================
# Keyframe Interpolation Request/Response
# ============================================================================

class KeyframeImage(BaseModel):
    """A keyframe image with its target frame index and conditioning strength."""

    image_url: str = Field(..., description="URL of the keyframe image")
    frame_index: int = Field(..., ge=0, description="Target frame index (0-indexed)")
    strength: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Conditioning strength"
    )


class KeyframeInterpolateRequest(BaseModel):
    """Request body for keyframe interpolation video generation."""

    job_id: str = Field(..., description="Unique job identifier")
    prompt: str = Field(
        ..., 
        description="Description of the video content", 
        max_length=2000
    )
    negative_prompt: str = Field(
        default="",
        description="What should not appear in the video",
        max_length=1000
    )
    keyframes: list[KeyframeImage] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Keyframe images with frame indices (at least 1 required)"
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=10.0,
        description="Target video duration in seconds (frames rounded up, then trimmed)"
    )
    frame_rate: float = Field(
        default=24.0,
        ge=8.0,
        le=60.0,
        description="Frame rate (FPS)"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the generated video"
    )
    width: int | None = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target width (overrides aspect_ratio default). Use 1920 for 1080p."
    )
    height: int | None = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target height (overrides aspect_ratio default). Use 1080 for 1080p."
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    enhance_prompt: bool = Field(
        default=False,
        description="Auto-enhance the prompt for better results"
    )
    save_url: str = Field(..., description="Presigned URL (PUT) for direct storage upload")
    webhook_url: str = Field(
        ...,
        description="REQUIRED: URL to POST when interpolation completes (success or failure)",
    )
    item_id: str | None = Field(
        default=None,
        description="Client identifier for this item (returned in webhook, defaults to job_id)",
    )
    webhook_secret: str | None = Field(
        default=None,
        description="Secret for signing webhook payload (HMAC-SHA256)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440004",
                    "prompt": "A person walking from left to right, cinematic lighting",
                    "keyframes": [
                        {"image_url": "https://example.com/start.png", "frame_index": 0, "strength": 1.0},
                        {"image_url": "https://example.com/end.png", "frame_index": 120, "strength": 1.0}
                    ],
                    "duration_seconds": 5.0,
                    "frame_rate": 24.0,
                    "save_url": "https://example.com/upload/video.mp4",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                }
            ]
        }
    }


class KeyframeInterpolateResponse(BaseModel):
    """Response for successful keyframe interpolation."""

    status: Literal["completed"] = "completed"
    generation_time: float = Field(..., description="Processing time in seconds")
    save_url: str = Field(..., description="The URL where the video was saved")
    duration_seconds: float = Field(..., description="Final video duration in seconds")
    has_audio: bool = Field(default=True, description="Whether the video contains audio")
    upscale_info: dict[str, Any] | None = Field(
        default=None,
        description="Upscaling metadata if video was upscaled (720p→1080p)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "generation_time": 45.2,
                    "save_url": "https://example.com/upload/video.mp4",
                    "duration_seconds": 5.0,
                    "has_audio": True,
                    "upscale_info": {
                        "original_resolution": "1280x720",
                        "upscaled_resolution": "1920x1080",
                        "upscale_time_seconds": 8.5,
                        "was_upscaled": True
                    }
                }
            ]
        }
    }
