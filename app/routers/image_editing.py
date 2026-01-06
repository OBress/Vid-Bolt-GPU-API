import logging
import time
from typing import Annotated

from fastapi import APIRouter, Form

from app.config import get_settings
from app.dependencies import APIKeyDep, StorageDep, GeneratorDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse
from app.models.image_editing import EditType, ImageEditResponse
from app.services.mock_generator import ImageEditParams

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
    response_model=ImageEditResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        500: {"model": ErrorResponse, "description": "Processing or upload error"},
    },
    summary="Edit Image",
    description="Edit an existing image using AI. Input image must be provided as a URL.",
)
async def edit_image(
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    job_id: Annotated[str, Form(description="Unique job identifier")],
    input_image_url: Annotated[str, Form(description="URL of the input image to edit")],
    prompt: Annotated[str, Form(description="Description of the edit", max_length=2000)],
    edit_type: Annotated[EditType, Form(description="Type of edit to apply")] = EditType.STYLE_TRANSFER,
    strength: Annotated[float, Form(description="Edit strength", ge=0.0, le=1.0)] = 0.75,
    mask_image_url: Annotated[str | None, Form(description="URL of the mask image for inpainting")] = None,
    seed: Annotated[int | None, Form(description="Random seed for reproducibility")] = None,
    output_url: Annotated[str | None, Form(description="Optional presigned URL (PUT) for direct storage upload")] = None,
) -> ImageEditResponse:
    """Edit an image with AI-powered transformations."""
    start_time = time.time()

    if edit_type == EditType.INPAINT and mask_image_url is None:
        raise ValidationError("mask_image_url is required for inpaint edit type")

    logger.info(
        f"Image edit request (URL flow)",
        extra={
            "job_id": job_id,
            "edit_type": edit_type.value,
            "has_mask": mask_image_url is not None,
            "has_output_url": output_url is not None,
        },
    )

    # Download input image
    input_image_data = await storage.download_from_url(input_image_url)

    # Validate magic bytes
    if not _validate_image_magic_bytes(input_image_data):
        raise ValidationError("input_image_url does not point to a valid image file")

    # Download and validate mask if provided
    mask_data = None
    if mask_image_url is not None:
        mask_data = await storage.download_from_url(mask_image_url)
        if not _validate_image_magic_bytes(mask_data):
            raise ValidationError("mask_image_url does not point to a valid image file")

    # Upload input image to R2 for internal tracking (still required for our mock flow/tracking)
    input_r2_key, _ = storage.upload_input_image(input_image_data, job_id)

    # Prepare edit parameters
    params = ImageEditParams(
        job_id=job_id,
        input_image_data=input_image_data,
        prompt=prompt,
        edit_type=edit_type.value,
        strength=strength,
        mask_data=mask_data,
        seed=seed,
    )

    # Generate edited image
    result = await generator.edit_image(params)

    # Upload output
    if output_url:
        r2_url = await storage.upload_to_url(
            data=result.image_data,
            url=output_url,
            content_type="image/png",
        )
        r2_key = None
    else:
        r2_key, r2_url = storage.upload_image(result.image_data, job_id, suffix="_edited")

    generation_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Image edit completed",
        extra={
            "job_id": job_id,
            "seed": result.seed,
            "generation_time_ms": generation_time_ms,
        },
    )

    return ImageEditResponse(
        status="completed",
        r2_key=r2_key,
        r2_url=r2_url,
        input_r2_key=input_r2_key,
        original_width=result.original_width,
        original_height=result.original_height,
        output_width=result.output_width,
        output_height=result.output_height,
        edit_type=edit_type.value,
        seed=result.seed,
        generation_time_ms=generation_time_ms,
    )
