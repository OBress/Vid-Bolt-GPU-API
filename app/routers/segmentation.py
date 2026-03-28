"""Segmentation API router - SAM 3 image and video segmentation.

Endpoints:
- POST /api/v1/segment/image — Segment objects in an image by text/visual prompts
- POST /api/v1/segment/video — Track and segment objects across video frames
"""

import logging
import time

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse
from app.models.segmentation import ImageSegmentRequest, VideoSegmentRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import ImageSegmentationParams, VideoSegmentationParams
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/segment",
    tags=["Segmentation"],
)


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


def _validate_video_magic_bytes(data: bytes) -> bool:
    """Validate video by checking magic bytes (MP4/MOV)."""
    if len(data) < 12:
        return False
    # MP4: ftyp box at offset 4
    if data[4:8] == b"ftyp":
        return True
    # MOV
    if data[4:8] == b"moov" or data[4:8] == b"mdat":
        return True
    return False


# =============================================================================
# Image Segmentation
# =============================================================================

@router.post(
    "/image",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Segmentation mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Segment Image",
    description="Segment objects in an image using text or visual prompts (SAM 3). Returns segmentation masks.",
)
async def segment_image(
    request: Request,
    body: ImageSegmentRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Segment objects in an image (Async)."""

    # Validate at least one prompt type is provided
    if not body.text_prompt and not body.point_prompts and not body.box_prompts:
        raise HTTPException(
            status_code=400,
            detail="At least one prompt type required: text_prompt, point_prompts, or box_prompts",
        )

    # Download and validate input image
    try:
        input_image_data = await storage.download_from_url(body.input_image_url)
        if not _validate_image_magic_bytes(input_image_data):
            raise ValidationError("input_image_url is not a valid image")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert prompt lists to tuples for internal dataclass
    point_prompts = None
    if body.point_prompts:
        point_prompts = [(p[0], p[1]) for p in body.point_prompts]

    box_prompts = None
    if body.box_prompts:
        box_prompts = [(b[0], b[1], b[2], b[3]) for b in body.box_prompts]

    params = ImageSegmentationParams(
        job_id=body.job_id,
        input_image_data=input_image_data,
        text_prompt=body.text_prompt,
        point_prompts=point_prompts,
        box_prompts=box_prompts,
        max_objects=body.max_objects,
    )

    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.SEGMENTATION,
        task_func=_run_image_segment,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        model_manager=model_manager,
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


async def _run_image_segment(
    model_manager: ModelManagerDep,
    storage: StorageDep,
    params: ImageSegmentationParams,
    save_url: str,
) -> JobResult:
    """Background task for image segmentation."""
    start_time = time.time()

    segmenter = model_manager.get_segmenter()
    result = await segmenter.segment_image(params)

    # Upload the masks JSON
    final_url = await storage.upload_to_url(
        data=result.masks_data,
        url=save_url,
        content_type="application/json",
    )

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata={
            "object_count": result.object_count,
            "width": result.width,
            "height": result.height,
            "boxes": [list(b) for b in result.boxes],
            "scores": result.scores,
        },
    )


# =============================================================================
# Video Segmentation
# =============================================================================

@router.post(
    "/video",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Segmentation mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Segment Video",
    description="Track and segment objects across video frames using text prompts (SAM 3).",
)
async def segment_video(
    request: Request,
    body: VideoSegmentRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Track and segment objects across video frames (Async)."""

    # Download and validate input video
    try:
        input_video_data = await storage.download_from_url(body.input_video_url)
        if not _validate_video_magic_bytes(input_video_data):
            raise ValidationError("input_video_url is not a valid MP4 video")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    params = VideoSegmentationParams(
        job_id=body.job_id,
        input_video_data=input_video_data,
        text_prompt=body.text_prompt,
        output_format=body.output_format,
        max_frames=body.max_frames,
    )

    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.SEGMENTATION,
        task_func=_run_video_segment,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        model_manager=model_manager,
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


async def _run_video_segment(
    model_manager: ModelManagerDep,
    storage: StorageDep,
    params: VideoSegmentationParams,
    save_url: str,
) -> JobResult:
    """Background task for video segmentation."""
    start_time = time.time()

    segmenter = model_manager.get_segmenter()
    result = await segmenter.segment_video(params)

    # Upload the result
    content_type = "application/json" if result.output_format == "masks_json" else "video/mp4"
    final_url = await storage.upload_to_url(
        data=result.result_data,
        url=save_url,
        content_type=content_type,
    )

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata={
            "frame_count": result.frame_count,
            "object_count": result.object_count,
            "tracked_ids": result.tracked_ids,
            "output_format": result.output_format,
        },
    )
