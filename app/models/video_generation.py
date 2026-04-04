"""Video generation request and response models."""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.common import AspectRatio


class VideoGenerateRequest(BaseModel):
    """Request body for video generation."""

    job_id: str = Field(..., description="Unique job identifier")
    input_image_url: str = Field(..., description="URL of the first frame image")
    prompt: str = Field(..., description="Description of motion/action", max_length=10000)
    duration_seconds: float = Field(default=4.0, description="Video duration in seconds", ge=1.0, le=8.0)
    fps: int = Field(default=24, description="Frames per second (8, 12, 16, 24, or 30)")
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the generated video",
    )
    width: int | None = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target width (overrides aspect_ratio default). Must be divisible by 8."
    )
    height: int | None = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target height (overrides aspect_ratio default). Must be divisible by 8."
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    end_image_url: str | None = Field(default=None, description="Optional URL of the end frame image")
    save_url: str = Field(..., description="Presigned URL (PUT) for direct storage upload")
    webhook_url: str | None = Field(
        default=None,
        description="Optional: URL to POST when generation completes (success or failure). If not provided, use polling.",
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
                    "job_id": "550e8400-e29b-41d4-a716-446655440002",
                    "input_image_url": "https://example.com/start.png",
                    "prompt": "Gentle waves on the beach",
                    "save_url": "https://example.com/upload/video.mp4",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                }
            ]
        }
    }


class VideoGenerateResponse(BaseModel):
    """Response for successful video generation."""

    status: Literal["completed"] = "completed"
    generation_time: float = Field(..., description="Processing time in seconds")
    save_url: str = Field(..., description="The URL where the video was saved")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "generation_time": 8.5,
                    "save_url": "https://example.com/upload/video.mp4",
                }
            ]
        }
    }
