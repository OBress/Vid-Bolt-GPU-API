"""Common response models."""

from typing import Literal

from pydantic import BaseModel, Field



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
    """Detailed status response (authenticated)."""

    r2_connected: bool = Field(..., description="Whether R2 storage is connected")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "0.1.0",
                    "mock_mode": True,
                    "r2_connected": True,
                }
            ]
        }
    }
