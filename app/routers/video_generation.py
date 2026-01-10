import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Request, HTTPException

from app.config import get_settings
from app.dependencies import APIKeyDep, StorageDep, GeneratorDep, JobManagerDep, ModelManagerDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse, get_dimensions
from app.models.video_generation import VideoGenerateRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import VideoGenerationParams
from app.services.model_manager import ModelMode

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
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy (concurrency limit reached)"},
        503: {"model": ErrorResponse, "description": "Video mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Generate Video",
    description="Generate a video from an image and prompt. Returns the generated video URL.",
)
async def generate_video(
    request: Request,
    body: VideoGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
) -> AsyncJobResponse:
    """Generate a video from an input image (Async)."""
    settings = get_settings()

    # 1. Determine active generator and ensure mode (if not mock)
    active_generator = generator

    if not settings.mock_mode:
        if not await model_manager.ensure_mode(ModelMode.VIDEO):
            # This legacy endpoint might expect to manually switch or error?
            # Original code check `current_mode`.
            # Let's keep consistent with others and try to auto-switch if possible
            if not await model_manager.ensure_mode(ModelMode.VIDEO):
                raise HTTPException(
                     status_code=409,
                     detail="System is busy."
                )
        active_generator = model_manager.get_video_generator()

    # 2. Validation
    if body.fps not in ALLOWED_FPS:
        raise ValidationError(f"fps must be one of {sorted(ALLOWED_FPS)}")
    if body.duration_seconds > settings.max_video_duration_seconds:
        raise ValidationError(f"duration exceeds {settings.max_video_duration_seconds}s limit")

    # 3. Download inputs (Fail Fast if invalid)
    try:
        input_image_data = await storage.download_from_url(body.input_image_url)
        if not _validate_image_magic_bytes(input_image_data):
            raise ValidationError("Invalid input image")

        end_image_data = None
        if body.end_image_url:
            end_image_data = await storage.download_from_url(body.end_image_url)
            if not _validate_image_magic_bytes(end_image_data):
                raise ValidationError("Invalid end image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    width, height = get_dimensions(body.aspect_ratio)
    # Override with explicit dimensions if provided
    if body.width is not None and body.height is not None:
        width, height = body.width, body.height

    params = VideoGenerationParams(
        job_id=body.job_id,
        input_image_data=input_image_data,
        prompt=body.prompt,
        negative_prompt="",
        duration_seconds=body.duration_seconds,
        frame_rate=float(body.fps),
        width=width,
        height=height,
        seed=body.seed,
        end_image_data=end_image_data,
    )

    # 4. Submit Job
    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        mode=ModelMode.VIDEO,
        task_func=_run_video_generation,
        generator=active_generator,
        storage=storage,
        params=params,
        save_url=body.save_url,
    )

    if not submitted:
        raise HTTPException(status_code=429, detail="System busy: Max concurrent video jobs reached")

    return AsyncJobResponse(
        job_id=body.job_id,
        status_url=str(request.url_for("get_job_status", job_id=body.job_id)),
    )


async def _run_video_generation(
    generator: GeneratorDep,
    storage: StorageDep,
    params: VideoGenerationParams,
    save_url: str,
) -> JobResult:
    """Background task for video generation."""
    start_time = time.time()

    result = await generator.generate_video(params)

    final_url = await storage.upload_to_url(
        data=result.video_data,
        url=save_url,
        content_type="video/mp4",
    )

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        duration_seconds=getattr(result, "duration_seconds", params.duration_seconds),
        has_audio=getattr(result, "has_audio", False),
    )
