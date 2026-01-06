import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Form

from app.config import get_settings
from app.dependencies import APIKeyDep, StorageDep, GeneratorDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse
from app.models.video_generation import VideoGenerateResponse
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
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_id: Annotated[str, Form(description="Unique job identifier")],
    input_image_url: Annotated[str, Form(description="URL of the first frame image")],
    prompt: Annotated[str, Form(description="Description of motion/action", max_length=2000)],
    duration_seconds: Annotated[float, Form(description="Video duration in seconds", ge=1.0, le=8.0)] = 4.0,
    fps: Annotated[int, Form(description="Frames per second (8, 12, 16, 24, or 30)")] = 24,
    motion_strength: Annotated[float, Form(description="Motion intensity", ge=0.0, le=1.0)] = 0.5,
    seed: Annotated[int | None, Form(description="Random seed for reproducibility")] = None,
    end_image_url: Annotated[str | None, Form(description="Optional URL of the end frame image")] = None,
    output_url: Annotated[str | None, Form(description="Optional presigned URL (PUT) for direct storage upload")] = None,
) -> VideoGenerateResponse:
    """Generate a video from an input image."""
    start_time = time.time()
    settings = get_settings()

    if fps not in ALLOWED_FPS:
        raise ValidationError(f"fps must be one of {sorted(ALLOWED_FPS)}, got {fps}")

    if duration_seconds > settings.max_video_duration_seconds:
        raise ValidationError(f"duration_seconds cannot exceed {settings.max_video_duration_seconds}")

    logger.info(
        f"Video generation request (URL flow)",
        extra={
            "job_id": job_id,
            "duration": duration_seconds,
            "fps": fps,
            "has_end_image": end_image_url is not None,
            "has_output_url": output_url is not None,
        },
    )

    # Download input image
    input_image_data = await storage.download_from_url(input_image_url)
    if not _validate_image_magic_bytes(input_image_data):
        raise ValidationError("input_image_url does not point to a valid image file")

    # Upload input to R2 for internal tracking
    input_r2_key, _ = storage.upload_input_image(input_image_data, job_id)

    # Handle optional end image
    end_image_data: bytes | None = None
    end_input_r2_key: str | None = None
    if end_image_url is not None:
        end_image_data = await storage.download_from_url(end_image_url)
        if not _validate_image_magic_bytes(end_image_data):
            raise ValidationError("end_image_url does not point to a valid image file")
        end_input_r2_key, _ = storage.upload_input_image(end_image_data, job_id, suffix="_end")

    params = VideoGenerationParams(
        job_id=job_id,
        input_image_data=input_image_data,
        prompt=prompt,
        duration_seconds=duration_seconds,
        fps=fps,
        motion_strength=motion_strength,
        seed=seed,
        end_image_data=end_image_data,
    )

    result = await generator.generate_video(params)

    if output_url:
        r2_url = await storage.upload_to_url(
            data=result.video_data,
            url=output_url,
            content_type="video/mp4",
        )
        r2_key = None
    else:
        r2_key, r2_url = storage.upload_video(result.video_data, job_id)

    generation_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Video generation completed",
        extra={
            "job_id": job_id,
            "seed": result.seed,
            "generation_time_ms": generation_time_ms,
        },
    )

    return VideoGenerateResponse(
        status="completed",
        r2_key=r2_key,
        r2_url=r2_url,
        input_r2_key=input_r2_key,
        end_input_r2_key=end_input_r2_key,
        duration_seconds=result.duration_seconds,
        fps=result.fps,
        frame_count=result.frame_count,
        width=result.width,
        height=result.height,
        seed=result.seed,
        generation_time_ms=generation_time_ms,
    )
