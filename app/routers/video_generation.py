import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Form

from app.config import get_settings
from app.dependencies import APIKeyDep, StorageDep, GeneratorDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse, get_dimensions
from app.models.video_generation import VideoGenerateRequest, VideoGenerateResponse
from app.services.mock_generator import VideoGenerationParams

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/video",
    tags=["Video Generation"],
)

# Allowed FPS values
ALLOWED_FPS = {8, 12, 16, 24, 30}


def _validate_image_magic_bytes(data: bytes) -> bool:
    """Validate image by checking magic bytes."""
    if len(data) < 8:
        return False
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


@router.post(
    "/generate",
    response_model=VideoGenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        500: {"model": ErrorResponse, "description": "Generation or upload error"},
    },
    summary="Generate Video",
    description="Generate a video from an input image URL with AI-powered motion.",
)
async def generate_video(
    request: VideoGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
) -> VideoGenerateResponse:
    """Generate a video from an input image."""
    start_time = time.time()
    settings = get_settings()

    if request.fps not in ALLOWED_FPS:
        raise ValidationError(f"fps must be one of {sorted(ALLOWED_FPS)}, got {request.fps}")

    if request.duration_seconds > settings.max_video_duration_seconds:
        raise ValidationError(f"duration_seconds cannot exceed {settings.max_video_duration_seconds}")

    logger.info(
        f"Video generation request (URL flow)",
        extra={
            "job_id": request.job_id,
            "duration": request.duration_seconds,
            "fps": request.fps,
            "aspect_ratio": request.aspect_ratio,
            "has_end_image": request.end_image_url is not None,
            "has_save_url": True,
        },
    )

    # Download input image
    input_image_data = await storage.download_from_url(request.input_image_url)
    if not _validate_image_magic_bytes(input_image_data):
        raise ValidationError("input_image_url does not point to a valid image file")

    # Handle optional end image
    end_image_data: bytes | None = None
    if request.end_image_url is not None:
        end_image_data = await storage.download_from_url(request.end_image_url)
        if not _validate_image_magic_bytes(end_image_data):
            raise ValidationError("end_image_url does not point to a valid image file")

    # Calculate dimensions
    width, height = get_dimensions(request.aspect_ratio)

    params = VideoGenerationParams(
        job_id=request.job_id,
        input_image_data=input_image_data,
        prompt=request.prompt,
        duration_seconds=request.duration_seconds,
        fps=request.fps,
        width=width,
        height=height,
        seed=request.seed,
        end_image_data=end_image_data,
    )

    result = await generator.generate_video(params)

    # Upload output
    save_url = await storage.upload_to_url(
        data=result.video_data,
        url=request.save_url,
        content_type="video/mp4",
    )

    generation_time = round(time.time() - start_time, 2)

    return VideoGenerateResponse(
        status="completed",
        generation_time=generation_time,
        save_url=save_url,
    )
