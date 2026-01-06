"""Image generation request and response models."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
        description="Aspect ratio of the generated image",
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
    save_url: str = Field(
        ...,
        description="Presigned URL (PUT) for direct storage upload",
    )




    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "prompt": "A beautiful sunset over mountains with vibrant colors",
                    "aspect_ratio": "16:9",
                    "num_inference_steps": 20,
                    "save_url": "https://example.com/upload/image.png",
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
