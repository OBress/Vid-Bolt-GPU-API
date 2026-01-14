"""Image editing request and response models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.models.common import AspectRatio


class EditType(str, Enum):
    """Available image editing types."""

    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    STYLE_TRANSFER = "style_transfer"
    REMOVE_BACKGROUND = "remove_background"
    UPSCALE = "upscale"


class ImageEditRequest(BaseModel):
    """Request body for image editing."""

    job_id: str = Field(..., description="Unique job identifier")
    input_image_url: str = Field(..., description="URL of the input image to edit")
    prompt: str = Field(..., description="Description of the edit", max_length=2000)
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.r_16_9,
        description="Aspect ratio of the edited image",
    )
    mask_image_url: str | None = Field(default=None, description="URL of the mask image for inpainting")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    save_url: str = Field(..., description="Presigned URL (PUT) for direct storage upload")
    webhook_url: str = Field(
        ...,
        description="REQUIRED: URL to POST when editing completes (success or failure)",
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
                    "job_id": "550e8400-e29b-41d4-a716-446655440001",
                    "input_image_url": "https://example.com/input.png",
                    "prompt": "Convert to oil painting style",
                    "aspect_ratio": "16:9",
                    "save_url": "https://example.com/upload/edited.png",
                    "webhook_url": "https://myapp.com/api/gpu-callback",
                }
            ]
        }
    }


class ImageEditResponse(BaseModel):
    """Response for successful image editing."""

    status: Literal["completed"] = "completed"
    generation_time: float = Field(..., description="Processing time in seconds")
    save_url: str = Field(..., description="The URL where the edited image was saved")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "generation_time": 3.5,
                    "save_url": "https://example.com/upload/edited.png",
                }
            ]
        }
    }
