"""Batch image generation request models."""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.common import AspectRatio


class BatchImageGenerateItem(BaseModel):
    """Single item in an image generation batch.
    
    Mirrors ImageGenerateRequest fields but without job_id (auto-generated).
    """
    item_id: str = Field(
        ...,
        description="REQUIRED: Client identifier for this item (returned in webhook)",
        min_length=1,
        max_length=100,
    )
    prompt: str = Field(
        ...,
        description="Text prompt describing the image to generate",
        min_length=1,
        max_length=10000,
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the generated image. Ignored if width/height are provided.",
    )
    width: Optional[int] = Field(
        default=None,
        description="Custom width in pixels",
        ge=256,
        le=2048,
    )
    height: Optional[int] = Field(
        default=None,
        description="Custom height in pixels",
        ge=256,
        le=2048,
    )
    num_inference_steps: int = Field(
        default=20,
        description="Number of diffusion steps",
        ge=1,
        le=50,
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation",
    )
    lora_name: Optional[str] = Field(
        default=None,
        description="Name of the LoRA style to apply (or 'none' for no LoRA)",
    )
    save_url: str = Field(
        ...,
        description="Presigned URL (PUT) for direct storage upload",
    )

    @model_validator(mode="after")
    def validate_dimensions(self):
        """Ensure both width and height are provided together."""
        if self.width is not None and self.height is None:
            raise ValueError("Height must be provided if width is provided")
        if self.width is None and self.height is not None:
            raise ValueError("Width must be provided if height is provided")
        return self


class BatchImageGenerateRequest(BaseModel):
    """Request body for batch image generation."""
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
    items: List[BatchImageGenerateItem] = Field(
        ...,
        description="List of image generation requests (max 500)",
        min_length=1,
        max_length=500,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_id": "batch-550e8400-e29b-41d4-a716-446655440000",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                    "items": [
                        {
                            "item_id": "scene_001_image",
                            "prompt": "A beautiful sunset over mountains",
                            "aspect_ratio": "16:9",
                            "save_url": "https://example.com/upload/1.png"
                        },
                        {
                            "item_id": "scene_002_image",
                            "prompt": "A cat sitting on a windowsill",
                            "aspect_ratio": "1:1",
                            "save_url": "https://example.com/upload/2.png"
                        }
                    ]
                }
            ]
        }
    }
