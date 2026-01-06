"""Video generation request and response models."""

from typing import Literal

from pydantic import BaseModel, Field


class VideoGenerateResponse(BaseModel):
    """Response for successful video generation."""

    status: Literal["completed"] = "completed"
    r2_key: str = Field(..., description="Storage key for video in R2")
    r2_url: str = Field(..., description="Public CDN URL for the video")
    input_r2_key: str = Field(..., description="Storage key for input image in R2")
    end_input_r2_key: str | None = Field(None, description="Storage key for end frame image in R2 (if provided)")
    duration_seconds: float = Field(..., description="Actual video duration in seconds")
    fps: int = Field(..., description="Frames per second")
    frame_count: int = Field(..., description="Total number of frames")
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")
    seed: int = Field(..., description="Seed used for generation")
    generation_time_ms: int = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "r2_key": "outputs/videos/550e8400-e29b-41d4-a716-446655440002.mp4",
                    "r2_url": "https://cdn.vid-bolt.com/outputs/videos/550e8400-e29b-41d4-a716-446655440002.mp4",
                    "input_r2_key": "inputs/550e8400-e29b-41d4-a716-446655440002/source.png",
                    "end_input_r2_key": None,
                    "duration_seconds": 4.0,
                    "fps": 24,
                    "frame_count": 96,
                    "width": 1280,
                    "height": 720,
                    "seed": 42,
                    "generation_time_ms": 8500,
                }
            ]
        }
    }
