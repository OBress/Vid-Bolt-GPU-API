import logging
import time
from typing import Annotated

from fastapi import APIRouter, Form

from app.config import get_settings
from app.dependencies import APIKeyDep, StorageDep, GeneratorDep, ImageModeDep
from app.exceptions import ValidationError
from app.models.common import ErrorResponse, get_dimensions
from app.models.image_editing import EditType, ImageEditRequest, ImageEditResponse
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
        503: {"model": ErrorResponse, "description": "Image mode not active or system busy"},
        500: {"model": ErrorResponse, "description": "Processing or upload error"},
    },
    summary="Edit Image",
    description="Edit an existing image using AI. Input image must be provided as a URL.",
)
async def edit_image(
    request: ImageEditRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    generator: GeneratorDep,
    _mode_guard: ImageModeDep,
) -> ImageEditResponse:
    """Edit an image with AI-powered transformations."""
    start_time = time.time()
    
    # Validation logic for inpainting is now slightly ambiguous without edit_type in request
    # If mask is provided, we assume inpainting intent, but since edit_type is gone, 
    # we can't strictly validate against it unless we infer it. 
    # For now, let's keep mask validation simple or remove strict dependency on removed edit_type.
    # User asked to remove edit_type, so we can't check `request.edit_type`.
    
    # Use aspect ratio for logging and logic
    logger.info(
        f"Image edit request (URL flow)",
        extra={
            "job_id": request.job_id,
            "aspect_ratio": request.aspect_ratio,
            "has_mask": request.mask_image_url is not None,
            "has_save_url": True,
        },
    )

    # Download input image
    input_image_data = await storage.download_from_url(request.input_image_url)

    # Validate magic bytes
    if not _validate_image_magic_bytes(input_image_data):
        raise ValidationError("input_image_url does not point to a valid image file")

    # Download and validate mask if provided
    mask_data = None
    if request.mask_image_url is not None:
        mask_data = await storage.download_from_url(request.mask_image_url)
        if not _validate_image_magic_bytes(mask_data):
            raise ValidationError("mask_image_url does not point to a valid image file")

    # Calculate dimensions
    width, height = get_dimensions(request.aspect_ratio)

    # Prepare edit parameters
    params = ImageEditParams(
        job_id=request.job_id,
        input_image_data=input_image_data,
        prompt=request.prompt,
        width=width,
        height=height,
        mask_data=mask_data,
        seed=request.seed,
    )

    # Generate edited image
    result = await generator.edit_image(params)

    # Upload output
    save_url = await storage.upload_to_url(
        data=result.image_data,
        url=request.save_url,
        content_type="image/png",
    )

    generation_time = round(time.time() - start_time, 2)

    return ImageEditResponse(
        status="completed",
        generation_time=generation_time,
        save_url=save_url,
    )
