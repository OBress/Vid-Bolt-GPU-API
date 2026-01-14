"""Batch video generation request models."""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.common import AspectRatio


class BatchVideoGenerateItem(BaseModel):
    """Single item in a video generation batch.
    
    Mirrors LTX2GenerateRequest fields but without job_id (auto-generated).
    """
    item_id: str = Field(
        ...,
        description="REQUIRED: Client identifier for this item (returned in webhook)",
        min_length=1,
        max_length=100,
    )
    input_image_url: str = Field(
        ...,
        description="URL of the start frame image",
    )
    prompt: str = Field(
        ...,
        description="Description of the video content/motion",
        max_length=2000,
    )
    negative_prompt: str = Field(
        default="",
        description="What should not appear in the video",
        max_length=1000,
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=10.0,
        description="Target video duration in seconds",
    )
    frame_rate: float = Field(
        default=24.0,
        ge=8.0,
        le=60.0,
        description="Frame rate (FPS)",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the generated video",
    )
    width: Optional[int] = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target width (overrides aspect_ratio default)",
    )
    height: Optional[int] = Field(
        default=None,
        ge=512,
        le=1920,
        description="Target height (overrides aspect_ratio default)",
    )
    end_image_url: Optional[str] = Field(
        default=None,
        description="Optional URL of the end frame image for interpolation",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    enhance_prompt: bool = Field(
        default=False,
        description="Auto-enhance the prompt for better results",
    )
    save_url: str = Field(
        ...,
        description="Presigned URL (PUT) for direct storage upload",
    )


class BatchVideoGenerateRequest(BaseModel):
    """Request body for batch video generation."""
    batch_id: str = Field(
        ...,
        description="Unique batch identifier (UUID) provided by the caller",
        min_length=1,
    )
    webhook_url: str = Field(
        ...,
        description="REQUIRED: URL to POST when each item completes (success or failure)",
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        description="Secret for signing webhook payloads (HMAC-SHA256)",
    )
    items: List[BatchVideoGenerateItem] = Field(
        ...,
        description="List of video generation requests (max 100)",
        min_length=1,
        max_length=100,  # Videos are heavier, lower limit
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_id": "batch-550e8400-e29b-41d4-a716-446655440002",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                    "items": [
                        {
                            "item_id": "video_scene_001",
                            "input_image_url": "https://example.com/start1.png",
                            "prompt": "Gentle waves on the beach, cinematic motion",
                            "duration_seconds": 5.0,
                            "save_url": "https://example.com/upload/1.mp4"
                        },
                        {
                            "item_id": "video_scene_002",
                            "input_image_url": "https://example.com/start2.png",
                            "prompt": "Clouds moving across the sky, timelapse",
                            "duration_seconds": 3.0,
                            "save_url": "https://example.com/upload/2.mp4"
                        }
                    ]
                }
            ]
        }
    }
