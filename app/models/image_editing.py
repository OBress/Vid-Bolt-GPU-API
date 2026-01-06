"""Image editing request and response models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EditType(str, Enum):
    """Available image editing types."""

    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    STYLE_TRANSFER = "style_transfer"
    REMOVE_BACKGROUND = "remove_background"
    UPSCALE = "upscale"


class ImageEditResponse(BaseModel):
    """Response for successful image editing."""

    status: Literal["completed"] = "completed"
    r2_key: str = Field(..., description="Storage key for edited image in R2")
    r2_url: str = Field(..., description="Public CDN URL for the edited image")
    input_r2_key: str = Field(..., description="Storage key for input image in R2")
    original_width: int = Field(..., description="Original image width")
    original_height: int = Field(..., description="Original image height")
    output_width: int = Field(..., description="Edited image width")
    output_height: int = Field(..., description="Edited image height")
    edit_type: str = Field(..., description="Type of edit applied")
    seed: int = Field(..., description="Seed used for generation")
    generation_time_ms: int = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "r2_key": "outputs/images/550e8400-e29b-41d4-a716-446655440001_edited.png",
                    "r2_url": "https://cdn.vid-bolt.com/outputs/images/550e8400-e29b-41d4-a716-446655440001_edited.png",
                    "input_r2_key": "inputs/550e8400-e29b-41d4-a716-446655440001/source.png",
                    "original_width": 1280,
                    "original_height": 720,
                    "output_width": 1280,
                    "output_height": 720,
                    "edit_type": "style_transfer",
                    "seed": 42,
                    "generation_time_ms": 3500,
                }
            ]
        }
    }
