"""Image generation endpoint."""

import logging
import time

from fastapi import APIRouter

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, GeneratorDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.models.common import ErrorResponse, get_dimensions
from app.models.image_generation import ImageGenerateRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import ImageGenerationParams
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/image",
    tags=["Image Generation"],
)


@router.post(
    "/generate",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy (concurrency limit reached)"},
        503: {"model": ErrorResponse, "description": "Image mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Generate Image",
    description="Generate an image from a text prompt using AI. Returns the generated image URL.",
)
async def generate_image(
    request: Request,
    body: ImageGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    """Generate an image from a text prompt (Async).

    Returns 202 Accepted if job is queued, or 429/503 if busy.
    """
    # 1. Determine active generator (if not mock)
    active_generator = generator
    
    if not settings.mock_mode:
        # Note: Worker will handle mode switching automatically
        active_generator = model_manager.get_image_generator()

    # 2. Prepare parameters
    if body.width and body.height:
        width, height = body.width, body.height
    else:
        width, height = get_dimensions(body.aspect_ratio)

    params = ImageGenerationParams(
        job_id=body.job_id,
        prompt=body.prompt,
        width=width,
        height=height,
        seed=body.seed,
        num_inference_steps=body.num_inference_steps,
        lora_name=body.lora_name if body.lora_name and body.lora_name.lower() != "none" else None,
    )

    # 3. Try to submit job
    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.IMAGE_GENERATION,
        task_func=_run_image_generation,
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
        raise HTTPException(
            status_code=429,
            detail="System busy: Maximum concurrent image generations reached."
        )

    # 4. Return Accepted response
    return AsyncJobResponse(
        job_id=body.job_id,
        status_url=str(request.url_for("get_job_status", job_id=body.job_id)),
    )


async def _run_image_generation(
    generator: GeneratorDep,
    storage: StorageDep,
    params: ImageGenerationParams,
    save_url: str,
) -> JobResult:
    """Background task for image generation."""
    start_time = time.time()
    
    # Generate
    result = await generator.generate_image(params)

    # Upload
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
            "height": result.height,
        }
    )

