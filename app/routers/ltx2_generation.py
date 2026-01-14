"""LTX-2 video generation endpoints.

This module provides two endpoints for LTX-2 video generation:
- POST /api/v1/ltx2/generate - Standard I2V (image-to-video) generation
- POST /api/v1/ltx2/interpolate - Keyframe interpolation
"""

import logging
import math
import time

from fastapi import APIRouter

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, GeneratorDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse, get_dimensions
from app.models.ltx2_generation import (
    LTX2GenerateRequest,
    KeyframeInterpolateRequest,
    round_up_to_valid_frames,
)
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import (
    VideoGenerationParams as LTX2VideoParams, 
    KeyframeInterpolationParams,
)
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/ltx2",
    tags=["LTX-2 Video Generation"],
)


def _validate_image_magic_bytes(data: bytes) -> bool:
    """Validate image by checking magic bytes."""
    if len(data) < 8:
        return False
    # PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # JPEG
    if data[:3] == b"\xff\xd8\xff":
        return True
    # WebP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    return False


# ============================================================================
# I2V (Image-to-Video) Endpoint
# ============================================================================

@router.post(
    "/generate",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Video mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="LTX-2 Video Generation",
    description="Generate a video using LTX-2 (I2V or T2V).",
)
async def generate_video(
    request: Request,
    body: LTX2GenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Generate a video from a start frame image (Async)."""
    
    # 1. Determine active generator (if not mock)
    active_generator = generator

    if not settings.mock_mode:
        # Note: Worker will handle mode switching automatically
        active_generator = model_manager.get_video_generator()

    # 2. Check if generator supports LTX-2 (Video Generation)
    if not hasattr(active_generator, "generate_video"):
         raise ValidationError("Current generator does not support video generation.")

    # 3. Pre-download checks
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

    params = LTX2VideoParams(
        job_id=body.job_id,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        input_image_data=input_image_data,
        end_image_data=end_image_data,
        duration_seconds=body.duration_seconds,
        frame_rate=body.frame_rate,
        width=width,
        height=height,
        seed=body.seed,
        enhance_prompt=body.enhance_prompt,
    )

    # 4. Submit Job
    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.VIDEO_GENERATION,
        task_func=_run_ltx2_generation,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        generator=active_generator,
        storage=storage,
        params=params,
        save_url=body.save_url,
    )

    if not submitted:
        raise HTTPException(status_code=429, detail="System busy: Max concurrent jobs reached")

    return AsyncJobResponse(
        job_id=body.job_id,
        status_url=str(request.url_for("get_job_status", job_id=body.job_id)),
    )


async def _run_ltx2_generation(
    generator: GeneratorDep,
    storage: StorageDep,
    params: LTX2VideoParams,
    save_url: str,
) -> JobResult:
    """Background task for LTX-2 I2V."""
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
        duration_seconds=result.duration_seconds,
        has_audio=result.has_audio,
        metadata={"upscale_info": getattr(result, "upscale_info", None)}
    )


# ============================================================================
# Keyframe Interpolation Endpoint
# ============================================================================

@router.post(
    "/interpolate",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Video mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="LTX-2 Keyframe Interpolation",
    description="Generate a video by interpolating between keyframes using LTX-2.",
)
async def interpolate_keyframes(
    request: Request,
    body: KeyframeInterpolateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Generate a video by interpolating between keyframes (Async)."""
    
    # 1. Determine active generator (if not mock)
    active_generator = generator

    if not settings.mock_mode:
        # Note: Worker will handle mode switching automatically
        active_generator = model_manager.get_video_generator()

    if not hasattr(active_generator, "generate_keyframe_video"):
        raise ValidationError("Current generator does not support keyframe interpolation.")

    # 2. Keyframe download loop
    keyframes = []
    try:
        for idx, kf in enumerate(body.keyframes):
            image_data = await storage.download_from_url(kf.image_url)
            if not _validate_image_magic_bytes(image_data):
                raise ValidationError(f"Keyframe {idx} invalid image")
            keyframes.append((image_data, kf.frame_index, kf.strength))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    width, height = get_dimensions(body.aspect_ratio)
    # Override with explicit dimensions if provided
    if body.width is not None and body.height is not None:
        width, height = body.width, body.height

    params = KeyframeInterpolationParams(
        job_id=body.job_id,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        keyframes=keyframes,
        duration_seconds=body.duration_seconds,
        frame_rate=body.frame_rate,
        width=width,
        height=height,
        seed=body.seed,
        enhance_prompt=body.enhance_prompt,
    )

    # 3. Submit
    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.VIDEO_GENERATION,
        task_func=_run_ltx2_interpolation,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        generator=active_generator,
        storage=storage,
        params=params,
        save_url=body.save_url,
    )

    if not submitted:
        raise HTTPException(status_code=429, detail="System busy: Max concurrent jobs reached")

    return AsyncJobResponse(
        job_id=body.job_id,
        status_url=str(request.url_for("get_job_status", job_id=body.job_id)),
    )


async def _run_ltx2_interpolation(
    generator: GeneratorDep,
    storage: StorageDep,
    params: KeyframeInterpolationParams,
    save_url: str,
) -> JobResult:
    """Background task for interpolation."""
    start_time = time.time()
    
    result = await generator.generate_keyframe_video(params)
    
    final_url = await storage.upload_to_url(
        data=result.video_data,
        url=save_url,
        content_type="video/mp4",
    )
    
    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        duration_seconds=result.duration_seconds,
        has_audio=result.has_audio,
        metadata={"upscale_info": getattr(result, "upscale_info", None)}
    )
