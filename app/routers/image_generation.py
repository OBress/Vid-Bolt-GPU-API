"""Image generation endpoint."""

import logging
import time

from fastapi import APIRouter

from app.dependencies import APIKeyDep, StorageDep, GeneratorDep
from app.models.common import ErrorResponse
from app.models.image_generation import ImageGenerateRequest, ImageGenerateResponse
from app.services.mock_generator import ImageGenerationParams

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/image",
    tags=["Image Generation"],
)


@router.post(
    "/generate",
    response_model=ImageGenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        500: {"model": ErrorResponse, "description": "Generation or upload error"},
    },
    summary="Generate Image",
    description="Generate an image from a text prompt using AI. Returns the generated image URL.",
)
async def generate_image(
    request: ImageGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
) -> ImageGenerateResponse:
    """Generate an image from a text prompt.

    The image is generated using the configured AI model (or mock in development)
    and uploaded to R2 storage. The public CDN URL is returned.

    Args:
        request: Image generation parameters
        api_key: Validated API key (injected)
        storage: Storage service (injected)
        generator: Generator service (injected)

    Returns:
        ImageGenerateResponse with the generated image details
    """
    start_time = time.time()

    logger.info(
        f"Image generation request",
        extra={
            "job_id": request.job_id,
            "width": request.width,
            "height": request.height,
            "prompt_length": len(request.prompt),
        },
    )

    # Prepare generation parameters
    params = ImageGenerationParams(
        job_id=request.job_id,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        seed=request.seed,
        num_inference_steps=request.num_inference_steps,
    )

    # Generate image
    result = await generator.generate_image(params)

    # Upload output
    if request.output_url:
        r2_url = await storage.upload_to_url(
            data=result.image_data,
            url=request.output_url,
            content_type="image/png",
        )
        r2_key = None
    else:
        r2_key, r2_url = storage.upload_image(result.image_data, request.job_id)

    generation_time_ms = int((time.time() - start_time) * 1000)



    logger.info(
        f"Image generation completed",
        extra={
            "job_id": request.job_id,
            "seed": result.seed,
            "generation_time_ms": generation_time_ms,
        },
    )

    return ImageGenerateResponse(
        status="completed",
        r2_key=r2_key,
        r2_url=r2_url,
        width=result.width,
        height=result.height,
        seed=result.seed,
        generation_time_ms=generation_time_ms,
    )
