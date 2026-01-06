"""Common response models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AspectRatio(str, Enum):
    """Common aspect ratios."""

    r_16_9 = "16:9"
    r_9_16 = "9:16"
    r_1_1 = "1:1"
    r_4_3 = "4:3"
    r_3_4 = "3:4"


def get_dimensions(aspect_ratio: AspectRatio) -> tuple[int, int]:
    """Get dimensions (width, height) for a given aspect ratio.
    
    Returns:
        tuple[int, int]: (width, height)
    """
    if aspect_ratio == AspectRatio.r_16_9:
        return 1280, 720
    elif aspect_ratio == AspectRatio.r_9_16:
        return 720, 1280
    elif aspect_ratio == AspectRatio.r_1_1:
        return 1024, 1024
    elif aspect_ratio == AspectRatio.r_4_3:
        return 1024, 768
    elif aspect_ratio == AspectRatio.r_3_4:
        return 768, 1024
    # Default fallback (should not be reached if properly validated)
    return 1280, 720


class ErrorResponse(BaseModel):

    """Standard error response format."""

    status: Literal["failed"] = "failed"
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "failed",
                    "error_code": "VALIDATION_ERROR",
                    "error_message": "Prompt cannot exceed 2000 characters",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    mock_mode: bool = Field(..., description="Whether mock mode is enabled")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "0.1.0",
                    "mock_mode": True,
                }
            ]
        }
    }


class StatusResponse(HealthResponse):
    """Detailed status response (authenticated).
    
    Currently identical to HealthResponse but kept for future extensibility
    (e.g., adding ComfyUI connection status, GPU info, etc.).
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "0.1.0",
                    "mock_mode": True,
                }
            ]
        }
    }

