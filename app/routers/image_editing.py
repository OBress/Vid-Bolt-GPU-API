import logging
import time
from typing import Annotated

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, GeneratorDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse
from app.models.image_editing import ImageEditRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import ImageEditParams
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/image",
    tags=["Image Editing"],
)


def _validate_image_magic_bytes(data: bytes) -> bool:
    """Validate image by checking magic bytes."""
    if len(data) < 8:
        return False
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # JPEG: FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return True
    # WebP: 52 49 46 46 ... 57 45 42 50
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


@router.post(
    "/edit",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy (concurrency limit reached)"},
        503: {"model": ErrorResponse, "description": "Image mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Edit Image",
    description="Edit an image using AI (inpainting or instruction-based). Returns the edited image URL.",
)
async def edit_image(
    request: Request,
    body: ImageEditRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Edit an image with AI-powered transformations (Async)."""
    
    # 1. Determine active generator and ensure mode (if not mock)
    active_generator = generator

    if not settings.mock_mode:
        if not await model_manager.ensure_mode_for_job(JobType.IMAGE_EDITING):
            raise HTTPException(
                status_code=409,
                detail="System is currently busy processing other tasks. Please wait until they are finished."
            )
        active_generator = model_manager.get_image_editor()

    # 2. Pre-validation of input URLs (Fail fast)
    # We download images HERE (blocking the request slightly) to ensure they are valid
    # before queuing the job. This prevents queue slots being taken by bad requests.
    try:
        input_image_data = await storage.download_from_url(body.input_image_url)
        if not _validate_image_magic_bytes(input_image_data):
            raise ValidationError("input_image_url is not a valid image")

        mask_data = None
        if body.mask_image_url:
            mask_data = await storage.download_from_url(body.mask_image_url)
            if not _validate_image_magic_bytes(mask_data):
                raise ValidationError("mask_image_url is not a valid image")
    except Exception as e:
        # Wrap storage/validation errors
        raise HTTPException(status_code=400, detail=str(e))

    # Extract dimensions from input image to preserve original resolution
    from PIL import Image
    import io
    try:
        input_image = Image.open(io.BytesIO(input_image_data))
        width, height = input_image.size
        input_image.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read input image dimensions: {e}")

    params = ImageEditParams(
        job_id=body.job_id,
        input_image_data=input_image_data,
        prompt=body.prompt,
        width=width,
        height=height,
        mask_data=mask_data,
        seed=body.seed,
        # Dynamic LoRA support
        lora_name=body.lora_name,
        lora_strength=body.lora_strength,
    )

    # 3. Submit Job
    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.IMAGE_EDITING,
        task_func=_run_image_edit,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        # Args for task_func:
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


async def _run_image_edit(
    generator: GeneratorDep,
    storage: StorageDep,
    params: ImageEditParams,
    save_url: str,
) -> JobResult:
    """Background task for image editing."""
    start_time = time.time()
    
    result = await generator.edit_image(params)
    
    final_url = await storage.upload_to_url(
        data=result.image_data,
        url=save_url,
        content_type="image/png",
    )
    
    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata={
            "seed": result.seed,
            "width": result.width,
            "height": result.height
        }
    )
