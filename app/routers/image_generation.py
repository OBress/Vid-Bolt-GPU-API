"""Image generation endpoint."""

import logging
import time

from fastapi import APIRouter

from app.dependencies import APIKeyDep, StorageDep, GeneratorDep
from app.models.common import ErrorResponse, get_dimensions
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
            "aspect_ratio": request.aspect_ratio,
            "prompt_length": len(request.prompt),
        },
    )

    # Calculate dimensions
    width, height = get_dimensions(request.aspect_ratio)

    # Prepare generation parameters
    params = ImageGenerationParams(
        job_id=request.job_id,
        prompt=request.prompt,
        width=width,
        height=height,
        seed=request.seed,
        num_inference_steps=request.num_inference_steps,
    )

    # Generate image
    result = await generator.generate_image(params)

    # Upload output
    save_url = await storage.upload_to_url(
        data=result.image_data,
        url=request.save_url,
        content_type="image/png",
    )

    generation_time = round(time.time() - start_time, 2)



    logger.info(
        f"Image generation completed",
        extra={
            "job_id": request.job_id,
            "seed": result.seed,
            "generation_time_s": generation_time,
        },
    )

    return ImageGenerateResponse(
        status="completed",
        generation_time=generation_time,
        save_url=save_url,
    )
