"""GPU monitoring endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import APIKeyDep

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/gpu",
    tags=["GPU Monitoring"],
)


class GPUMemoryInfo(BaseModel):
    """GPU memory information for a single device."""
    device_index: int
    name: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float
    temperature_celsius: Optional[int] = None
    utilization_percent: Optional[int] = None


class GPUStatusResponse(BaseModel):
    """GPU status response."""
    available: bool
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    device_count: int = 0
    devices: list[GPUMemoryInfo] = []
    total_memory_gb: float = 0.0
    total_used_gb: float = 0.0
    total_free_gb: float = 0.0


@router.get(
    "/status",
    response_model=GPUStatusResponse,
    summary="Get GPU Status",
    description="Get detailed GPU memory and utilization information for all available GPUs.",
)
async def get_gpu_status(api_key: APIKeyDep) -> GPUStatusResponse:
    """Get GPU status and memory info.
    
    Returns detailed information about:
    - CUDA availability and version
    - Per-device memory usage
    - Total memory across all devices
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return GPUStatusResponse(available=False)
        
        device_count = torch.cuda.device_count()
        devices = []
        total_memory = 0.0
        total_used = 0.0
        total_free = 0.0
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            mem_info = torch.cuda.mem_get_info(i)
            free_bytes, total_bytes = mem_info
            used_bytes = total_bytes - free_bytes
            
            total_gb = total_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            
            devices.append(GPUMemoryInfo(
                device_index=i,
                name=props.name,
                total_gb=round(total_gb, 2),
                used_gb=round(used_gb, 2),
                free_gb=round(free_gb, 2),
                usage_percent=round((used_bytes / total_bytes) * 100, 1),
            ))
            
            total_memory += total_gb
            total_used += used_gb
            total_free += free_gb
        
        # Try to get CUDA version
        cuda_version = torch.version.cuda
        
        return GPUStatusResponse(
            available=True,
            cuda_version=cuda_version,
            device_count=device_count,
            devices=devices,
            total_memory_gb=round(total_memory, 2),
            total_used_gb=round(total_used, 2),
            total_free_gb=round(total_free, 2),
        )
        
    except ImportError:
        logger.warning("PyTorch not available for GPU status check")
        return GPUStatusResponse(available=False)
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        return GPUStatusResponse(available=False)


@router.post(
    "/clear-cache",
    summary="Clear GPU Cache",
    description="Force clear CUDA cache and run garbage collection. Useful after OOM errors.",
)
async def clear_gpu_cache(api_key: APIKeyDep) -> dict:
    """Clear GPU cache and run garbage collection.
    
    This can help recover from OOM situations by freeing
    up cached tensors and other GPU memory.
    """
    import gc
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"status": "skipped", "message": "CUDA not available"}
        
        # Get memory before
        free_before, total = torch.cuda.mem_get_info(0)
        used_before = total - free_before
        
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get memory after
        free_after, _ = torch.cuda.mem_get_info(0)
        used_after = total - free_after
        
        freed_mb = (used_before - used_after) / (1024**2)
        
        logger.info(f"GPU cache cleared, freed {freed_mb:.1f} MB")
        
        return {
            "status": "success",
            "freed_mb": round(freed_mb, 1),
            "used_before_gb": round(used_before / (1024**3), 2),
            "used_after_gb": round(used_after / (1024**3), 2),
        }
        
    except ImportError:
        return {"status": "skipped", "message": "PyTorch not available"}
    except Exception as e:
        logger.error(f"Failed to clear GPU cache: {e}")
        return {"status": "error", "message": str(e)}
