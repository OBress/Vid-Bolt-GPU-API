"""Segmentation API router - SAM 3 image and video segmentation.

Endpoints:
- POST /api/v1/segment/image — Segment objects in an image by text/visual prompts
- POST /api/v1/segment/video — Track and segment objects across video frames
"""

import copy
import logging
import time

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse
from app.models.segmentation import ImageSegmentRequest, VideoSegmentRequest
from app.models.segmentation_animation import AnimateSegmentRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import ImageSegmentationParams, VideoSegmentationParams, ImageAnimationParams
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


async def _hydrate_segmentation_operations(storage: StorageDep, operations):
    """Download any external assets referenced by operations."""
    if not operations:
        return operations

    hydrated = copy.deepcopy(operations)
    for op in hydrated:
        if not isinstance(op, dict) or op.get("type") != "replace_background":
            continue

        image_url = op.get("image_url")
        if not image_url:
            continue

        bg_image_data = await storage.download_from_url(image_url)
        if not _validate_image_magic_bytes(bg_image_data):
            raise ValidationError("replace_background.image_url is not a valid image")
        op["_bg_image_data"] = bg_image_data

    return hydrated


def _normalize_video_object_prompts(body: VideoSegmentRequest):
    """Normalize video prompt modes to a single internal object-prompt list."""
    if body.object_prompts:
        return [{"label": op.label, "text": op.text} for op in body.object_prompts]
    if body.text_prompts:
        return [{"label": prompt.strip(), "text": prompt.strip()} for prompt in body.text_prompts]
    if body.text_prompt:
        text_prompt = body.text_prompt.strip()
        return [{"label": text_prompt, "text": text_prompt}]
    return None


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
    description="Segment objects in an image using text or visual prompts (SAM 3.1 checkpoints). Returns segmentation masks.",
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
    if not body.text_prompt and not body.point_prompts and not body.box_prompts and not body.box_prompts_labeled and not body.object_prompts:
        raise HTTPException(
            status_code=400,
            detail="At least one prompt type required: text_prompt, point_prompts, box_prompts, box_prompts_labeled, or object_prompts",
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

    box_prompts_labeled = None
    if body.box_prompts_labeled:
        box_prompts_labeled = [((bp.box[0], bp.box[1], bp.box[2], bp.box[3]), bp.label) for bp in body.box_prompts_labeled]

    # Convert object_prompts to dict list for internal dataclass
    object_prompts = None
    if body.object_prompts:
        object_prompts = [{"label": op.label, "text": op.text} for op in body.object_prompts]

    operations = await _hydrate_segmentation_operations(storage, body.operations)

    params = ImageSegmentationParams(
        job_id=body.job_id,
        input_image_data=input_image_data,
        text_prompt=body.text_prompt,
        point_prompts=point_prompts,
        box_prompts=box_prompts,
        box_prompts_labeled=box_prompts_labeled,
        object_prompts=object_prompts,
        confidence_threshold=body.confidence_threshold,
        max_objects=body.max_objects,
        output_type=body.output_type,
        operations=operations,
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

    # Upload the result (masks JSON or processed image)
    final_url = await storage.upload_to_url(
        data=result.masks_data,
        url=save_url,
        content_type=result.content_type,
    )

    metadata = {
        "object_count": result.object_count,
        "width": result.width,
        "height": result.height,
        "boxes": [list(b) for b in result.boxes],
        "scores": result.scores,
        "output_type": params.output_type,
        "model_version": result.model_version,
    }
    if result.labels:
        metadata["labels"] = result.labels

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata=metadata,
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
    description="Track and segment objects across video frames using deterministic text and visual prompts (SAM 3.1).",
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

    # Validate at least one prompt type is provided
    if (
        not body.text_prompt
        and not body.text_prompts
        and not body.object_prompts
        and not body.point_prompts
        and not body.box_prompts
    ):
        raise HTTPException(
            status_code=400,
            detail="At least one prompt type required: text_prompt, text_prompts, object_prompts, point_prompts, or box_prompts",
        )

    # Download and validate input video
    try:
        input_video_data = await storage.download_from_url(body.input_video_url)
        if not _validate_video_magic_bytes(input_video_data):
            raise ValidationError("input_video_url is not a valid MP4 video")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    operations = await _hydrate_segmentation_operations(storage, body.operations)

    params = VideoSegmentationParams(
        job_id=body.job_id,
        input_video_data=input_video_data,
        text_prompt=body.text_prompt,
        text_prompts=body.text_prompts,
        object_prompts=_normalize_video_object_prompts(body),
        point_prompts=body.point_prompts,
        point_labels=body.point_labels,
        box_prompts=body.box_prompts,
        box_labels=body.box_labels,
        prompt_frame_index=body.prompt_frame_index,
        propagation_direction=body.propagation_direction,
        confidence_threshold=body.confidence_threshold,
        output_format=body.output_format,
        operations=operations,
        max_frames=body.max_frames,
        include_tracking_metadata=body.include_tracking_metadata,
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
            "prompt_to_obj_ids": result.prompt_to_obj_ids,
            "object_id_to_prompt_label": {str(k): v for k, v in result.object_id_to_prompt_label.items()},
            "model_version": result.model_version,
            "include_tracking_metadata": params.include_tracking_metadata,
        },
    )


# =============================================================================
# Animated Segmentation (Image → Video)
# =============================================================================

@router.post(
    "/animate",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Segmentation mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Animate Segmented Image",
    description=(
        "Generate an animated video from a segmented image. Segments the image "
        "using SAM 3.1 checkpoints, then renders animated visual effects (with easing, "
        "transitions, draw-on, pulse, etc.) to produce an MP4 video."
    ),
)
async def animate_segment(
    request: Request,
    body: AnimateSegmentRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Animate segmented image to video (Async)."""

    # Validate at least one prompt type is provided
    if (
        not body.text_prompt
        and not body.object_prompts
        and not body.point_prompts
        and not body.box_prompts
        and not body.box_prompts_labeled
    ):
        raise HTTPException(
            status_code=400,
            detail="At least one prompt type required: text_prompt, object_prompts, point_prompts, box_prompts, or box_prompts_labeled",
        )

    if not body.operations:
        raise HTTPException(
            status_code=400,
            detail="At least one operation is required for animation",
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

    # Convert prompts
    point_prompts = None
    if body.point_prompts:
        point_prompts = [(p[0], p[1]) for p in body.point_prompts]

    box_prompts = None
    if body.box_prompts:
        box_prompts = [(b[0], b[1], b[2], b[3]) for b in body.box_prompts]

    box_prompts_labeled = None
    if body.box_prompts_labeled:
        box_prompts_labeled = [((bp.box[0], bp.box[1], bp.box[2], bp.box[3]), bp.label) for bp in body.box_prompts_labeled]

    object_prompts = None
    if body.object_prompts:
        object_prompts = [{"label": op.label, "text": op.text} for op in body.object_prompts]

    operations = await _hydrate_segmentation_operations(storage, body.operations)

    params = ImageAnimationParams(
        job_id=body.job_id,
        input_image_data=input_image_data,
        text_prompt=body.text_prompt,
        object_prompts=object_prompts,
        point_prompts=point_prompts,
        box_prompts=box_prompts,
        box_prompts_labeled=box_prompts_labeled,
        confidence_threshold=body.confidence_threshold,
        max_objects=body.max_objects,
        duration_seconds=body.duration_seconds,
        fps=body.fps,
        operations=operations,
    )

    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.SEGMENTATION,
        task_func=_run_animate_segment,
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


async def _run_animate_segment(
    model_manager: ModelManagerDep,
    storage: StorageDep,
    params: ImageAnimationParams,
    save_url: str,
) -> JobResult:
    """Background task for animated segmentation."""
    start_time = time.time()

    segmenter = model_manager.get_segmenter()
    result = await segmenter.animate_image(params)

    final_url = await storage.upload_to_url(
        data=result.video_data,
        url=save_url,
        content_type="video/mp4",
    )

    metadata = {
        "width": result.width,
        "height": result.height,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "frame_count": result.frame_count,
        "object_count": result.object_count,
        "model_version": result.model_version,
    }
    if result.labels:
        metadata["labels"] = result.labels

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata=metadata,
    )
