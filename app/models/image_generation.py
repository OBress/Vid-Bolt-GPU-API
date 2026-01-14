"""Image generation request and response models."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.common import AspectRatio


class ImageGenerateRequest(BaseModel):
    """Request body for image generation."""

    job_id: str = Field(
        ...,
        description="Unique job identifier (UUID) provided by the caller",
        min_length=1,
    )
    prompt: str = Field(
        ...,
        description="Text prompt describing the image to generate",
        min_length=1,
        max_length=2000,
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
        20,
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
    webhook_url: str = Field(
        ...,
        description="REQUIRED: URL to POST when generation completes (success or failure)",
    )
    item_id: Optional[str] = Field(
        default=None,
        description="Client identifier for this item (returned in webhook, defaults to job_id)",
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        description="Secret for signing webhook payload (HMAC-SHA256)",
    )

    @model_validator(mode="after")
    def validate_dimensions(self):
        """Enable custom width and height if one is provided, ensure the other is also provided."""
        if self.width is not None and self.height is None:
             raise ValueError("Height must be provided if width is provided")
        if self.width is None and self.height is not None:
             raise ValueError("Width must be provided if height is provided")
        return self




    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "prompt": "A beautiful sunset over mountains with vibrant colors",
                    "aspect_ratio": "16:9",
                    "num_inference_steps": 20,
                    "save_url": "https://example.com/upload/image.png",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                }
            ]
        }
    }


class ImageGenerateResponse(BaseModel):
    """Response for successful image generation."""

    status: Literal["completed"] = "completed"
    generation_time: float = Field(..., description="Generation time in seconds")
    save_url: str = Field(..., description="The URL where the image was saved")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "generation_time": 2.5,
                    "save_url": "https://example.com/upload/image.png",
                }
            ]
        }
    }
