"""Batch image editing request models."""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.common import AspectRatio


class BatchImageEditItem(BaseModel):
    """Single item in an image editing batch.
    
    Mirrors ImageEditRequest fields but without job_id (auto-generated).
    """
    input_image_url: str = Field(
        ...,
        description="URL of the input image to edit",
    )
    prompt: str = Field(
        ...,
        description="Description of the edit",
        max_length=2000,
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the edited image",
    )
    mask_image_url: Optional[str] = Field(
        default=None,
        description="URL of the mask image for inpainting",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    save_url: str = Field(
        ...,
        description="Presigned URL (PUT) for direct storage upload",
    )


class BatchImageEditRequest(BaseModel):
    """Request body for batch image editing."""
    batch_id: str = Field(
        ...,
        description="Unique batch identifier (UUID) provided by the caller",
        min_length=1,
    )
    items: List[BatchImageEditItem] = Field(
        ...,
        description="List of image edit requests (max 500)",
        min_length=1,
        max_length=500,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_id": "batch-550e8400-e29b-41d4-a716-446655440001",
                    "items": [
                        {
                            "input_image_url": "https://example.com/input1.png",
                            "prompt": "Convert to oil painting style",
                            "save_url": "https://example.com/upload/1.png"
                        },
                        {
                            "input_image_url": "https://example.com/input2.png",
                            "prompt": "Add dramatic lighting",
                            "save_url": "https://example.com/upload/2.png"
                        }
                    ]
                }
            ]
        }
    }
