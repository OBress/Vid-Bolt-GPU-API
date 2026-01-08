"""LTX-2 video generation endpoints.

This module provides two endpoints for LTX-2 video generation:
- POST /api/v1/ltx2/generate - Standard I2V (image-to-video) generation
- POST /api/v1/ltx2/interpolate - Keyframe interpolation
"""

import logging
import math
import time

from fastapi import APIRouter

from app.dependencies import APIKeyDep, StorageDep, GeneratorDep, VideoModeDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse, get_dimensions
from app.models.ltx2_generation import (
    LTX2GenerateRequest,
    LTX2GenerateResponse,
    KeyframeInterpolateRequest,
    KeyframeInterpolateResponse,
    round_up_to_valid_frames,
)
from app.services.ltx2_generator import LTX2VideoParams, KeyframeInterpolationParams

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
    response_model=LTX2GenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        503: {"model": ErrorResponse, "description": "Video mode not active or system busy"},
        500: {"model": ErrorResponse, "description": "Generation or upload error"},
    },
    summary="Generate Video from Image (I2V)",
    description=(
        "Generate a video from a single start frame image using LTX-2. "
        "Optionally provide an end frame for interpolation. "
        "The video includes synchronized AI-generated audio."
    ),
)
async def generate_video(
    request: LTX2GenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    _mode_guard: VideoModeDep,
) -> LTX2GenerateResponse:
    """Generate a video from a start frame image."""
    start_time = time.time()

    logger.info(
        f"LTX-2 I2V generation request",
        extra={
            "job_id": request.job_id,
            "duration_seconds": request.duration_seconds,
            "frame_rate": request.frame_rate,
            "aspect_ratio": request.aspect_ratio,
            "has_end_image": request.end_image_url is not None,
        },
    )

    # Download input image
    input_image_data = await storage.download_from_url(request.input_image_url)
    if not _validate_image_magic_bytes(input_image_data):
        raise ValidationError("input_image_url does not point to a valid image file")

    # Download optional end image
    end_image_data: bytes | None = None
    if request.end_image_url is not None:
        end_image_data = await storage.download_from_url(request.end_image_url)
        if not _validate_image_magic_bytes(end_image_data):
            raise ValidationError("end_image_url does not point to a valid image file")

    # Calculate dimensions
    width, height = get_dimensions(request.aspect_ratio)

    # Check if generator supports LTX-2 video generation
    if not hasattr(generator, "generate_video"):
        raise ValidationError(
            "Current generator does not support video generation. "
            "Ensure LTX2Generator is configured."
        )

    params = LTX2VideoParams(
        job_id=request.job_id,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        input_image_data=input_image_data,
        end_image_data=end_image_data,
        duration_seconds=request.duration_seconds,
        frame_rate=request.frame_rate,
        width=width,
        height=height,
        seed=request.seed,
        enhance_prompt=request.enhance_prompt,
    )

    result = await generator.generate_video(params)

    # Upload output
    save_url = await storage.upload_to_url(
        data=result.video_data,
        url=request.save_url,
        content_type="video/mp4",
    )

    generation_time = round(time.time() - start_time, 2)

    logger.info(
        f"LTX-2 I2V generation completed",
        extra={
            "job_id": request.job_id,
            "generation_time_s": generation_time,
            "duration_seconds": result.duration_seconds,
            "has_audio": result.has_audio,
        },
    )

    return LTX2GenerateResponse(
        status="completed",
        generation_time=generation_time,
        save_url=save_url,
        duration_seconds=result.duration_seconds,
        has_audio=result.has_audio,
        upscale_info=getattr(result, "upscale_info", None),
    )


# ============================================================================
# Keyframe Interpolation Endpoint
# ============================================================================

@router.post(
    "/interpolate",
    response_model=KeyframeInterpolateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        503: {"model": ErrorResponse, "description": "Video mode not active or system busy"},
        500: {"model": ErrorResponse, "description": "Generation or upload error"},
    },
    summary="Generate Keyframe Interpolation Video",
    description=(
        "Generate a video by interpolating between multiple keyframe images using LTX-2. "
        "Each keyframe specifies an image, target frame index, and conditioning strength. "
        "The video includes synchronized AI-generated audio."
    ),
)
async def interpolate_keyframes(
    request: KeyframeInterpolateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    _mode_guard: VideoModeDep,
) -> KeyframeInterpolateResponse:
    """Generate a video by interpolating between keyframes."""
    start_time = time.time()

    # Calculate frame count for validation
    requested_frames = math.ceil(request.duration_seconds * request.frame_rate) + 1
    num_frames = round_up_to_valid_frames(requested_frames)
    max_frame_idx = num_frames - 1

    # Validate keyframe indices are within bounds
    for kf in request.keyframes:
        if kf.frame_index > max_frame_idx:
            raise ValidationError(
                f"Keyframe frame_index {kf.frame_index} exceeds max frame index "
                f"{max_frame_idx} for {request.duration_seconds}s duration at {request.frame_rate}fps. "
                f"(Rounded to {num_frames} frames)"
            )

    logger.info(
        f"LTX-2 keyframe interpolation request",
        extra={
            "job_id": request.job_id,
            "num_keyframes": len(request.keyframes),
            "duration_seconds": request.duration_seconds,
            "frame_rate": request.frame_rate,
            "aspect_ratio": request.aspect_ratio,
            "rounded_frames": num_frames,
        },
    )

    # Download all keyframe images
    keyframes: list[tuple[bytes, int, float]] = []
    for idx, kf in enumerate(request.keyframes):
        image_data = await storage.download_from_url(kf.image_url)
        if not _validate_image_magic_bytes(image_data):
            raise ValidationError(
                f"Keyframe {idx} (frame_index={kf.frame_index}) does not point to a valid image"
            )
        keyframes.append((image_data, kf.frame_index, kf.strength))

    # Calculate dimensions
    width, height = get_dimensions(request.aspect_ratio)

    # Check if generator supports keyframe interpolation
    if not hasattr(generator, "generate_keyframe_video"):
        raise ValidationError(
            "Current generator does not support keyframe interpolation. "
            "Ensure LTX2Generator is configured."
        )

    params = KeyframeInterpolationParams(
        job_id=request.job_id,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        keyframes=keyframes,
        duration_seconds=request.duration_seconds,
        frame_rate=request.frame_rate,
        width=width,
        height=height,
        seed=request.seed,
        enhance_prompt=request.enhance_prompt,
    )

    result = await generator.generate_keyframe_video(params)

    # Upload output
    save_url = await storage.upload_to_url(
        data=result.video_data,
        url=request.save_url,
        content_type="video/mp4",
    )

    generation_time = round(time.time() - start_time, 2)

    logger.info(
        f"LTX-2 keyframe interpolation completed",
        extra={
            "job_id": request.job_id,
            "generation_time_s": generation_time,
            "duration_seconds": result.duration_seconds,
            "has_audio": result.has_audio,
        },
    )

    return KeyframeInterpolateResponse(
        status="completed",
        generation_time=generation_time,
        save_url=save_url,
        duration_seconds=result.duration_seconds,
        has_audio=result.has_audio,
        upscale_info=getattr(result, "upscale_info", None),
    )
