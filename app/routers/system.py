"""System and GPU monitoring endpoints."""

import logging
import platform
import os
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import get_settings, Settings, InferenceConfig
from app.dependencies import verify_api_key, get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["System"])


class GPUInfo(BaseModel):
    """GPU information."""
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    memory_usage_percent: float
    temperature_celsius: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None


class SystemInfo(BaseModel):
    """System information."""
    os: str
    os_version: str
    python_version: str
    cpu_count: int


class ModeInfo(BaseModel):
    """Current mode information."""
    mode: str
    is_busy: bool
    active_job_id: Optional[str]
    loaded_models: list[str]


class ConcurrencyLimits(BaseModel):
    """Concurrency configuration."""
    max_concurrent_image_generations: int
    max_concurrent_video_generations: int


class SystemStatusResponse(BaseModel):
    """Complete system status response."""
    system: SystemInfo
    gpu: Optional[GPUInfo] = None
    mode: Optional[ModeInfo] = None
    concurrency_limits: ConcurrencyLimits
    mock_mode: bool


def get_gpu_info() -> Optional[GPUInfo]:
    """Get GPU information using pynvml or torch."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        # Basic info from torch
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        
        # Use reserved (cached) memory as "used" for better accuracy
        used_memory = reserved_memory
        free_memory = total_memory - used_memory
        
        gpu_info = GPUInfo(
            name=name,
            memory_total_gb=round(total_memory / (1024**3), 2),
            memory_used_gb=round(used_memory / (1024**3), 2),
            memory_free_gb=round(free_memory / (1024**3), 2),
            memory_usage_percent=round((used_memory / total_memory) * 100, 1),
            cuda_version=torch.version.cuda,
        )
        
        # Try to get more detailed info from pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_info.temperature_celsius = temp
            except:
                pass
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info.gpu_utilization_percent = util.gpu
            except:
                pass
            
            # Driver version
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                gpu_info.driver_version = driver
            except:
                pass
            
            # More accurate memory info from NVML
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info.memory_total_gb = round(mem_info.total / (1024**3), 2)
                gpu_info.memory_used_gb = round(mem_info.used / (1024**3), 2)
                gpu_info.memory_free_gb = round(mem_info.free / (1024**3), 2)
                gpu_info.memory_usage_percent = round((mem_info.used / mem_info.total) * 100, 1)
            except:
                pass
            
            pynvml.nvmlShutdown()
        except ImportError:
            # pynvml not available, that's fine
            pass
        except Exception as e:
            logger.debug(f"NVML info unavailable: {e}")
        
        return gpu_info
        
    except ImportError:
        # torch not available
        return None
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


def get_system_info() -> SystemInfo:
    """Get system information."""
    return SystemInfo(
        os=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
    )


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="System Status",
    description="Get detailed system and GPU status. Requires authentication.",
    dependencies=[Depends(verify_api_key)],
)
async def get_system_status(
    settings: Settings = Depends(get_settings),
) -> SystemStatusResponse:
    """Get comprehensive system and GPU status.
    
    Returns:
    - System info (OS, Python version, CPU count)
    - GPU info (name, memory usage, temperature, utilization)
    - Current mode info (image/video, busy state, loaded models)
    - Concurrency limits
    """
    # Get GPU info
    gpu_info = None
    if not settings.mock_mode:
        gpu_info = get_gpu_info()
    
    # Get mode info from ModelManager if available
    mode_info = None
    try:
        model_manager = get_model_manager()
        status = model_manager.get_status()
        mode_info = ModeInfo(
            mode=status.mode.value,
            is_busy=status.is_busy,
            active_job_id=status.active_job_id,
            loaded_models=status.loaded_models,
        )
    except RuntimeError:
        # ModelManager not initialized (mock mode)
        pass
    
    return SystemStatusResponse(
        system=get_system_info(),
        gpu=gpu_info,
        mode=mode_info,
        concurrency_limits=ConcurrencyLimits(
            max_concurrent_image_generations=InferenceConfig.MAX_CONCURRENT_IMAGE_GENERATIONS,
            max_concurrent_video_generations=InferenceConfig.MAX_CONCURRENT_VIDEO_GENERATIONS,
        ),
        mock_mode=settings.mock_mode,
    )
