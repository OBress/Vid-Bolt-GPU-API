"""LoRA Management API Router.

This module provides endpoints for managing Z-Image LoRA models.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.dependencies import APIKeyDep

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/loras/z-image",
    tags=["LoRA Management"],
)

settings = get_settings()

# Maximum LoRA file size: 500 MB (SafeTensors LoRAs are typically 50-200 MB)
MAX_LORA_SIZE_BYTES = 500 * 1024 * 1024


def _sanitize_lora_name(name: str) -> str:
    """Sanitize LoRA name to prevent path traversal.
    
    Args:
        name: The raw LoRA name from user input.
        
    Returns:
        Sanitized name with only the basename.
        
    Raises:
        HTTPException: If the name contains path traversal characters.
    """
    # Use basename to strip any directory components
    sanitized = Path(name).name
    # Reject if it still contains traversal patterns
    if ".." in sanitized or "/" in sanitized or "\\" in sanitized:
        raise HTTPException(status_code=400, detail="Invalid LoRA name")
    # Reject empty names
    if not sanitized or sanitized == ".safetensors":
        raise HTTPException(status_code=400, detail="Invalid LoRA name")
    return sanitized


class LoraInfo(BaseModel):
    """Information about a LoRA model."""
    name: str
    size_bytes: int
    modified_time: float


@router.get(
    "",
    response_model=List[LoraInfo],
    summary="List Z-Image LoRAs",
    description="List available LoRA models for Z-Image generation.",
)
async def list_loras(
    api_key: APIKeyDep,
) -> List[LoraInfo]:
    """List available LoRA models."""
    lora_dir = Path(settings.zimage_lora_path)
    if not lora_dir.exists():
        return []

    loras = []
    # Ensure directory exists before scanning
    os.makedirs(lora_dir, exist_ok=True)
    
    for file_path in lora_dir.glob("*.safetensors"):
        if file_path.is_file():
            stat = file_path.stat()
            loras.append(
                LoraInfo(
                    name=file_path.stem,
                    size_bytes=stat.st_size,
                    modified_time=stat.st_mtime,
                )
            )
            
    return sorted(loras, key=lambda x: x.name)


@router.post(
    "/upload",
    summary="Upload LoRA",
    description="Upload a new LoRA model (.safetensors file).",
)
async def upload_lora(
    api_key: APIKeyDep,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Upload a new LoRA model."""
    if not file.filename.endswith(".safetensors"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .safetensors files are supported.",
        )

    lora_dir = Path(settings.zimage_lora_path)
    os.makedirs(lora_dir, exist_ok=True)
    
    # Sanitize filename (basic check)
    filename = Path(file.filename).name
    file_path = lora_dir / filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"LoRA '{filename}' already exists. Delete it first or use a different name.",
        )
    
    try:
        # Read content with size limit to prevent disk exhaustion
        content = await file.read()
        if len(content) > MAX_LORA_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"LoRA file too large ({len(content) / (1024*1024):.0f} MB). Maximum is {MAX_LORA_SIZE_BYTES // (1024*1024)} MB.",
            )
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
            
        logger.info(f"Uploaded LoRA: {filename} ({len(content) / (1024*1024):.1f} MB)")
        return JSONResponse(
            status_code=201,
            content={"status": "success", "message": f"LoRA {filename} uploaded successfully"},
        )
    except HTTPException:
        raise  # Re-raise size limit errors as-is
    except Exception as e:
        logger.error(f"Failed to upload LoRA {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.put(
    "/{lora_name}",
    summary="Rename LoRA",
    description="Rename an existing LoRA model.",
)
async def rename_lora(
    lora_name: str,
    new_name: str,
    api_key: APIKeyDep,
) -> JSONResponse:
    """Rename a LoRA model."""
    # Sanitize inputs to prevent path traversal
    safe_lora_name = _sanitize_lora_name(lora_name)
    safe_new_name = _sanitize_lora_name(new_name)
    
    lora_dir = Path(settings.zimage_lora_path)
    
    src_path = lora_dir / f"{safe_lora_name}.safetensors"
    dst_path = lora_dir / f"{safe_new_name}.safetensors"
    
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"LoRA '{lora_name}' not found")
        
    if dst_path.exists():
        raise HTTPException(status_code=409, detail=f"LoRA '{new_name}' already exists")
        
    try:
        src_path.rename(dst_path)
        logger.info(f"Renamed LoRA: {lora_name} -> {new_name}")
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": f"Renamed to {new_name}"},
        )
    except Exception as e:
        logger.error(f"Failed to rename LoRA: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rename file: {str(e)}")


@router.delete(
    "/{lora_name}",
    summary="Delete LoRA",
    description="Delete a LoRA model.",
)
async def delete_lora(
    lora_name: str,
    api_key: APIKeyDep,
) -> JSONResponse:
    """Delete a LoRA model."""
    # Sanitize input to prevent path traversal
    safe_lora_name = _sanitize_lora_name(lora_name)
    
    lora_dir = Path(settings.zimage_lora_path)
    file_path = lora_dir / f"{safe_lora_name}.safetensors"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"LoRA '{lora_name}' not found")
        
    try:
        file_path.unlink()
        logger.info(f"Deleted LoRA: {lora_name}")
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": f"Deleted {lora_name}"},
        )
    except Exception as e:
        logger.error(f"Failed to delete LoRA: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
