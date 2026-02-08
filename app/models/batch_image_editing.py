"""Batch image editing request models."""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.common import AspectRatio


class BatchImageEditItem(BaseModel):
    """Single item in an image editing batch.
    
    Mirrors ImageEditRequest fields but without job_id (auto-generated).
    """
    item_id: str = Field(
        ...,
        description="REQUIRED: Client identifier for this item (returned in webhook)",
        min_length=1,
        max_length=100,
    )
    input_image_url: str = Field(
        ...,
        description="URL of the input image to edit",
    )
    prompt: str = Field(
        ...,
        description="Description of the edit",
        max_length=10000,
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
    # Dynamic LoRA support (per-item)
    lora_name: Optional[str] = Field(
        default=None,
        description="Optional LoRA to apply. Available: 'multiple-angles' (96 camera positions)",
    )
    lora_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LoRA strength (0.0-1.0). Default: 0.9 when LoRA is specified",
    )


class BatchImageEditRequest(BaseModel):
    """Request body for batch image editing."""
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
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                    "items": [
                        {
                            "item_id": "edit_001",
                            "input_image_url": "https://example.com/input1.png",
                            "prompt": "Convert to oil painting style",
                            "save_url": "https://example.com/upload/1.png"
                        },
                        {
                            "item_id": "edit_002",
                            "input_image_url": "https://example.com/input2.png",
                            "prompt": "Add dramatic lighting",
                            "save_url": "https://example.com/upload/2.png"
                        }
                    ]
                }
            ]
        }
    }
