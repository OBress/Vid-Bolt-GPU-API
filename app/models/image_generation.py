"""Image generation request and response models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
    negative_prompt: str | None = Field(
        None,
        description="Text describing what to avoid in the image",
        max_length=1000,
    )
    width: int = Field(
        1280,
        description="Image width in pixels",
        ge=512,
        le=2048,
    )
    height: int = Field(
        720,
        description="Image height in pixels",
        ge=512,
        le=2048,
    )
    seed: int | None = Field(
        None,
        description="Random seed for reproducibility (random if not provided)",
    )
    num_inference_steps: int = Field(
        20,
        description="Number of diffusion steps",
        ge=1,
        le=50,
    )
    output_url: str | None = Field(
        None,
        description="Optional presigned URL (PUT) for direct storage upload",
    )

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions_multiple(cls, v: int) -> int:
        """Ensure dimensions are multiples of 8 for compatibility."""
        if v % 8 != 0:
            # Round to nearest multiple of 8
            return round(v / 8) * 8
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "prompt": "A beautiful sunset over mountains with vibrant colors",
                    "negative_prompt": "blurry, low quality",
                    "width": 1280,
                    "height": 720,
                    "seed": 42,
                    "num_inference_steps": 20,
                    "output_url": None,
                }
            ]
        }
    }


class ImageGenerateResponse(BaseModel):
    """Response for successful image generation."""

    status: Literal["completed"] = "completed"
    r2_key: str | None = Field(None, description="Storage key in R2 bucket (if uploaded to default storage)")
    r2_url: str = Field(..., description="Public CDN URL or providing output_url for the image")
    width: int = Field(..., description="Generated image width")
    height: int = Field(..., description="Generated image height")
    seed: int = Field(..., description="Seed used for generation")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "r2_key": "outputs/images/550e8400-e29b-41d4-a716-446655440000.png",
                    "r2_url": "https://cdn.vid-bolt.com/outputs/images/550e8400-e29b-41d4-a716-446655440000.png",
                    "width": 1280,
                    "height": 720,
                    "seed": 42,
                    "generation_time_ms": 2500,
                }
            ]
        }
    }
