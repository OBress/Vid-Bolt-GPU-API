"""Batch operations router.

This module provides endpoints for batch job submission and status tracking.
Supports image generation, image editing, and video generation batches.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request

from app.dependencies import (
    APIKeyDep,
    StorageDep,
    JobManagerDep,
    ModelManagerDep,
    SettingsDep,
)
from app.models.batch import AsyncBatchResponse, BatchInfo, BatchStatus
from app.models.batch_image_generation import BatchImageGenerateRequest
from app.models.batch_image_editing import BatchImageEditRequest
from app.models.batch_video_generation import BatchVideoGenerateRequest
from app.models.common import ErrorResponse
from app.services.batch_manager import BatchManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/batch",
    tags=["Batch Operations"],
)


# Global BatchManager instance (set during startup)
_batch_manager_instance: BatchManager | None = None


def set_batch_manager_instance(instance: BatchManager) -> None:
    """Set the global BatchManager instance (called during startup)."""
    global _batch_manager_instance
    _batch_manager_instance = instance


def get_batch_manager() -> BatchManager:
    """Get the BatchManager instance."""
    if _batch_manager_instance is None:
        raise RuntimeError("BatchManager not initialized. Server startup may have failed.")
    return _batch_manager_instance


BatchManagerDep = Annotated[BatchManager, Depends(get_batch_manager)]


# =============================================================================
# Image Generation Batch
# =============================================================================

@router.post(
    "/image/generate",
    response_model=AsyncBatchResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        409: {"model": ErrorResponse, "description": "Batch ID already exists"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Batch Image Generation",
    description="Submit a batch of image generation requests. Returns immediately with batch status URL.",
)
async def batch_generate_images(
    request: Request,
    body: BatchImageGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    batch_manager: BatchManagerDep,
    settings: SettingsDep,
) -> AsyncBatchResponse:
    """Submit a batch of image generation requests."""
    
    # Validate batch size
    if len(body.items) > BatchManager.MAX_IMAGE_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(body.items)} exceeds maximum of {BatchManager.MAX_IMAGE_BATCH_SIZE}"
        )
    
    # Get generator
    if settings.mock_mode:
        from app.dependencies import get_generator
        generator = get_generator(settings)
    else:
        generator = model_manager.get_image_generator()
    
    try:
        batch_info = await batch_manager.submit_image_generation_batch(
            batch_id=body.batch_id,
            items=body.items,
            generator=generator,
            storage=storage,
            webhook_url=body.webhook_url,
            webhook_secret=body.webhook_secret,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    
    logger.info(f"Accepted image generation batch {body.batch_id} with {len(body.items)} items")
    
    return AsyncBatchResponse(
        batch_id=body.batch_id,
        status=BatchStatus.PENDING,
        total_items=len(body.items),
        status_url=str(request.url_for("get_batch_status", batch_id=body.batch_id)),
        message=f"Batch accepted for processing ({len(body.items)} images)"
    )


# =============================================================================
# Image Editing Batch
# =============================================================================

@router.post(
    "/image/edit",
    response_model=AsyncBatchResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        409: {"model": ErrorResponse, "description": "Batch ID already exists"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Batch Image Editing",
    description="Submit a batch of image editing requests. Returns immediately with batch status URL.",
)
async def batch_edit_images(
    request: Request,
    body: BatchImageEditRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    batch_manager: BatchManagerDep,
    settings: SettingsDep,
) -> AsyncBatchResponse:
    """Submit a batch of image editing requests."""
    
    # Validate batch size
    if len(body.items) > BatchManager.MAX_IMAGE_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(body.items)} exceeds maximum of {BatchManager.MAX_IMAGE_BATCH_SIZE}"
        )
    
    # Get generator
    if settings.mock_mode:
        from app.dependencies import get_generator
        generator = get_generator(settings)
    else:
        generator = model_manager.get_image_editor()
    
    try:
        batch_info = await batch_manager.submit_image_editing_batch(
            batch_id=body.batch_id,
            items=body.items,
            generator=generator,
            storage=storage,
            webhook_url=body.webhook_url,
            webhook_secret=body.webhook_secret,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    
    logger.info(f"Accepted image editing batch {body.batch_id} with {len(body.items)} items")
    
    return AsyncBatchResponse(
        batch_id=body.batch_id,
        status=BatchStatus.PENDING,
        total_items=len(body.items),
        status_url=str(request.url_for("get_batch_status", batch_id=body.batch_id)),
        message=f"Batch accepted for processing ({len(body.items)} image edits)"
    )


# =============================================================================
# Video Generation Batch
# =============================================================================

@router.post(
    "/video/generate",
    response_model=AsyncBatchResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        409: {"model": ErrorResponse, "description": "Batch ID already exists"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Batch Video Generation",
    description="Submit a batch of video generation requests. Returns immediately with batch status URL.",
)
async def batch_generate_videos(
    request: Request,
    body: BatchVideoGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    batch_manager: BatchManagerDep,
    settings: SettingsDep,
) -> AsyncBatchResponse:
    """Submit a batch of video generation requests."""
    
    # Validate batch size (videos have lower limit)
    if len(body.items) > BatchManager.MAX_VIDEO_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(body.items)} exceeds maximum of {BatchManager.MAX_VIDEO_BATCH_SIZE}"
        )
    
    # Get generator
    if settings.mock_mode:
        from app.dependencies import get_generator
        generator = get_generator(settings)
    else:
        generator = model_manager.get_video_generator()
    
    try:
        batch_info = await batch_manager.submit_video_generation_batch(
            batch_id=body.batch_id,
            items=body.items,
            generator=generator,
            storage=storage,
            webhook_url=body.webhook_url,
            webhook_secret=body.webhook_secret,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    
    logger.info(f"Accepted video generation batch {body.batch_id} with {len(body.items)} items")
    
    return AsyncBatchResponse(
        batch_id=body.batch_id,
        status=BatchStatus.PENDING,
        total_items=len(body.items),
        status_url=str(request.url_for("get_batch_status", batch_id=body.batch_id)),
        message=f"Batch accepted for processing ({len(body.items)} videos)"
    )


# =============================================================================
# Batch Status & Collection
# =============================================================================

@router.get(
    "/{batch_id}",
    response_model=BatchInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Batch not found"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
    },
    summary="Get Batch Status",
    description="Get aggregate batch status and per-item results. Non-destructive.",
)
async def get_batch_status(
    batch_id: Annotated[str, Path(description="The unique batch ID")],
    api_key: APIKeyDep,
    batch_manager: BatchManagerDep,
) -> BatchInfo:
    """Get batch status (non-destructive)."""
    batch = batch_manager.get_batch(batch_id)
    
    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found. It may have expired (5 min) or been collected."
        )
    
    return batch


@router.delete(
    "/{batch_id}",
    response_model=BatchInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Batch not found"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
    },
    summary="Collect Batch Results",
    description="Get batch results and delete the batch. Use when done polling.",
)
async def collect_batch(
    batch_id: Annotated[str, Path(description="The unique batch ID")],
    api_key: APIKeyDep,
    batch_manager: BatchManagerDep,
) -> BatchInfo:
    """Collect batch results and delete the batch.
    
    This is the preferred method for final retrieval. Returns the batch
    status and then immediately deletes all batch tracking data.
    """
    batch = batch_manager.collect_batch(batch_id)
    
    if not batch:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found. It may have expired (5 min) or been collected."
        )
    
    logger.info(f"Collected batch {batch_id}: {batch.completed_items}/{batch.total_items} completed, {batch.failed_items} failed")
    
    return batch
