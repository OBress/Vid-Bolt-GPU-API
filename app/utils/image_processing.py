"""Image processing utilities for resolution normalization.

This module provides utilities for preprocessing images before video generation,
including padding to 64-divisible dimensions for two-stage LTX-2 pipeline.
"""

import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def pad_to_divisible(
    image_data: bytes,
    target_width: int,
    target_height: int,
    divisibility: int = 64,
) -> tuple[bytes, int, int, int, int]:
    """Resize image to 64-divisible dimensions for LTX-2 two-stage pipeline.
    
    The two-stage pipeline (with spatial upsampler) requires dimensions
    divisible by 64. This function resizes the input image to padded
    dimensions while preserving aspect ratio as much as possible.
    
    Args:
        image_data: Input image bytes (PNG/JPEG)
        target_width: Desired output width (will be padded up to 64-divisible)
        target_height: Desired output height (will be padded up to 64-divisible)
        divisibility: Divisor requirement (default: 64 for two-stage)
        
    Returns:
        Tuple of (padded_image_bytes, padded_w, padded_h, target_w, target_h)
    """
    image = Image.open(io.BytesIO(image_data))
    
    # Round up to nearest divisible value
    padded_w = ((target_width + divisibility - 1) // divisibility) * divisibility
    padded_h = ((target_height + divisibility - 1) // divisibility) * divisibility
    
    # Resize to padded dimensions
    resized = image.resize((padded_w, padded_h), Image.Resampling.LANCZOS)
    
    # Convert to bytes
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    buffer.seek(0)
    
    if (padded_w, padded_h) != (target_width, target_height):
        logger.info(
            f"Padded image from {target_width}x{target_height} to {padded_w}x{padded_h}"
        )
    
    return buffer.getvalue(), padded_w, padded_h, target_width, target_height


def center_crop_image(
    image_data: bytes,
    target_width: int,
    target_height: int,
) -> bytes:
    """Center crop an image to target dimensions.
    
    Args:
        image_data: Input image bytes
        target_width: Target width after cropping
        target_height: Target height after cropping
        
    Returns:
        Cropped image bytes
    """
    image = Image.open(io.BytesIO(image_data))
    orig_w, orig_h = image.size
    
    if (orig_w, orig_h) == (target_width, target_height):
        return image_data
    
    # Calculate crop box (center crop)
    left = (orig_w - target_width) // 2
    top = (orig_h - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    cropped = image.crop((left, top, right, bottom))
    
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    buffer.seek(0)
    
    logger.info(f"Center cropped image from {orig_w}x{orig_h} to {target_width}x{target_height}")
    
    return buffer.getvalue()
