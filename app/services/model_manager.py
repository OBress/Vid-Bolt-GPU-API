"""Model Manager - Dynamic Image/Video Mode Switching.

This module provides the ModelManager service for dynamically switching between
Image Mode and Video Mode, efficiently managing GPU VRAM by loading/unloading
model groups as needed.

Image Mode loads:
- ZImageGenerator (text-to-image)
- LightX2VImageEditGenerator (image editing)

Video Mode loads:
- LTX2Generator (video generation with native 2x upsampling)
"""

import asyncio
import gc
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.services.interfaces import (
        ImageEditor,
        ImageGenerator,
        VideoGenerator,
    )

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    """Current model mode."""
    NONE = "none"
    IMAGE = "image"
    VIDEO = "video"
    SWITCHING = "switching"


class VRAMLoadMode(str, Enum):
    """VRAM loading strategy."""
    DYNAMIC = "dynamic"  # Load/unload models as needed (saves VRAM)
    STATIC = "static"    # Keep all models in VRAM (instant switching)


@dataclass
class ModeStatus:
    """Status information about the current mode."""
    mode: ModelMode
    is_busy: bool = False
    active_job_id: Optional[str] = None
    loaded_models: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "mode": self.mode.value,
            "is_busy": self.is_busy,
            "active_job_id": self.active_job_id,
            "loaded_models": self.loaded_models,
        }


class ModelManager:
    """Manages dynamic loading/unloading of model groups.
    
    This service orchestrates switching between Image Mode and Video Mode,
    ensuring only one group of models is loaded at a time to manage VRAM.
    
    Attributes:
        current_mode: The currently active mode
        is_busy: Whether a generation job is in progress
        active_job_id: ID of the current job (if any)
    """

    def __init__(self, settings: Settings):
        """Initialize the ModelManager.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._mode = ModelMode.NONE
        self._vram_mode = VRAMLoadMode.DYNAMIC
        self._is_busy = False
        self._active_job_id: Optional[str] = None
        self._lock = asyncio.Lock()
        
        # Generator instances (lazy loaded)
        self._zimage_generator: Optional["ImageGenerator"] = None
        self._lightx2v_generator: Optional["ImageEditor"] = None
        self._ltx2_generator: Optional["VideoGenerator"] = None
        
        logger.info("ModelManager initialized")

    @property
    def current_mode(self) -> ModelMode:
        """Get the current mode."""
        return self._mode

    @property
    def is_busy(self) -> bool:
        """Check if a generation is in progress."""
        return self._is_busy

    @property
    def active_job_id(self) -> Optional[str]:
        """Get the active job ID."""
        return self._active_job_id

    def get_status(self) -> ModeStatus:
        """Get the current mode status."""
        loaded_models = []
        
        if self._zimage_generator and self._zimage_generator._loaded:
            loaded_models.append("z-image-turbo")
        if self._lightx2v_generator and self._lightx2v_generator._loaded:
            loaded_models.append("qwen-image-edit-2511")
        if self._ltx2_generator and self._ltx2_generator._loaded:
            loaded_models.append("ltx-2-19b")
            
        return ModeStatus(
            mode=self._mode,
            is_busy=self._is_busy,
            active_job_id=self._active_job_id,
            loaded_models=loaded_models,
        )

    @property
    def vram_mode(self) -> VRAMLoadMode:
        """Get the current VRAM loading mode."""
        return self._vram_mode

    async def set_vram_mode(self, mode: VRAMLoadMode) -> None:
        """Set the VRAM loading mode.
        
        Args:
            mode: Target VRAM mode (DYNAMIC or STATIC)
            
        Raises:
            RuntimeError: If currently busy with a job
        """
        if self._is_busy:
            raise RuntimeError("Cannot change VRAM mode while a job is in progress")
        
        if mode == self._vram_mode:
            logger.info(f"Already in {mode.value} VRAM mode")
            return
        
        logger.info(f"Switching VRAM mode from {self._vram_mode.value} to {mode.value}")
        
        if mode == VRAMLoadMode.STATIC:
            # Load all models for instant switching
            await self.load_all_models()
        else:
            # Switch to dynamic: unload models not needed for current mode
            if self._mode == ModelMode.IMAGE:
                await self._unload_video_models()
            elif self._mode == ModelMode.VIDEO:
                await self._unload_image_models()
        
        self._vram_mode = mode
        logger.info(f"VRAM mode set to {mode.value}")

    async def load_all_models(self) -> None:
        """Load all models (Image + Video) into VRAM.
        
        This enables static mode where all models are kept loaded for instant
        switching, at the cost of higher VRAM usage.
        """
        logger.info("Loading ALL models into VRAM (Static Mode)...")
        
        # Load everything
        await self._load_image_models()
        await self._load_video_models()
        
        # Set VRAM mode to static and default to Image mode
        self._vram_mode = VRAMLoadMode.STATIC
        self._mode = ModelMode.IMAGE
        logger.info("All models loaded successfully")

    async def acquire_job_lock(self, job_id: str) -> bool:
        """Try to acquire the job lock for generation.
        
        Args:
            job_id: ID of the job requesting the lock
            
        Returns:
            True if lock acquired, False if busy
        """
        async with self._lock:
            if self._is_busy:
                return False
            self._is_busy = True
            self._active_job_id = job_id
            logger.info(f"Job lock acquired: {job_id}")
            return True

    async def release_job_lock(self, job_id: str) -> None:
        """Release the job lock after generation completes.
        
        Args:
            job_id: ID of the job releasing the lock
        """
        async with self._lock:
            if self._active_job_id == job_id:
                self._is_busy = False
                self._active_job_id = None
                logger.info(f"Job lock released: {job_id}")
            else:
                logger.warning(f"Job {job_id} tried to release lock held by {self._active_job_id}")

    async def ensure_mode(self, target_mode: ModelMode) -> bool:
        """Ensure the system is in the target mode, switching if necessary.

        This handles the automatic logic:
        - If already in target mode: Return True (Ready)
        - If in different mode (e.g. Image vs Video):
            - If BUSY with jobs: Return False (Reject request)
            - If IDLE: Switch mode and Return True (Ready)

        Args:
            target_mode: The required mode (ModelMode.IMAGE or ModelMode.VIDEO)

        Returns:
            True if mode is set and ready to proceed.
            False if system is busy in another mode and cannot switch.
        """
        if self._mode == target_mode:
            return True

        # Check if we can switch
        # Note: We check is_busy without acquiring lock yet.
        # The lock will be acquired by the job submission later.
        # There is a tiny race condition window here but acceptable for this architecture.
        if self._is_busy:
            logger.warning(
                f"Cannot auto-switch to {target_mode} because system is busy in {self._mode} mode."
            )
            return False

        # Attempt switch
        logger.info(f"Auto-switching from {self._mode} to {target_mode}...")
        try:
            if target_mode == ModelMode.IMAGE:
                await self.switch_to_image_mode()
            elif target_mode == ModelMode.VIDEO:
                await self.switch_to_video_mode()
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Auto-switch failed: {e}")
            return False

    async def switch_to_image_mode(self) -> None:
        """Switch to Image Mode.
        
        Unloads video models and loads image models (Z-Image + LightX2V).
        
        Raises:
            RuntimeError: If currently busy with a job
        """
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        if self._mode == ModelMode.IMAGE:
            logger.info("Already in Image Mode")
            return
            
        logger.info("Switching to Image Mode...")
        self._mode = ModelMode.SWITCHING
        
        try:
            # Unload video models (if not in static mode)
            if self._vram_mode != VRAMLoadMode.STATIC:
                await self._unload_video_models()
            
            # Load image models (if not already loaded)
            await self._load_image_models()
            
            self._mode = ModelMode.IMAGE
            logger.info("Successfully switched to Image Mode")
            
        except Exception as e:
            logger.error(f"Failed to switch to Image Mode: {e}")
            self._mode = ModelMode.NONE
            raise

    async def switch_to_video_mode(self) -> None:
        """Switch to Video Mode.
        
        Unloads image models and loads video models (LTX-2).
        
        Raises:
            RuntimeError: If currently busy with a job
        """
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        if self._mode == ModelMode.VIDEO:
            logger.info("Already in Video Mode")
            return
            
        logger.info("Switching to Video Mode...")
        self._mode = ModelMode.SWITCHING
        
        try:
            # Unload image models (if not in static mode)
            if self._vram_mode != VRAMLoadMode.STATIC:
                await self._unload_image_models()
            
            # Load video models (if not already loaded)
            await self._load_video_models()
            
            self._mode = ModelMode.VIDEO
            logger.info("Successfully switched to Video Mode")
            
        except Exception as e:
            logger.error(f"Failed to switch to Video Mode: {e}")
            self._mode = ModelMode.NONE
            raise

    async def _load_image_models(self) -> None:
        """Load image generation models."""
        from app.services.zimage_generator import ZImageGenerator
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        # Load Z-Image for text-to-image
        if self._zimage_generator is None:
            self._zimage_generator = ZImageGenerator(self._settings)
        
        if not self._zimage_generator._loaded:
            logger.info("Loading Z-Image Turbo models...")
            await asyncio.to_thread(self._zimage_generator.load_models)
        
        # Load LightX2V for image editing
        if self._lightx2v_generator is None:
            self._lightx2v_generator = LightX2VImageEditGenerator(self._settings)
        
        if not self._lightx2v_generator._loaded:
            logger.info("Loading LightX2V (Qwen-Image-Edit) models...")
            await asyncio.to_thread(self._lightx2v_generator.load_models)

    async def _unload_image_models(self) -> None:
        """Unload image generation models and free VRAM."""
        if self._zimage_generator and self._zimage_generator._loaded:
            logger.info("Unloading Z-Image Turbo models...")
            await asyncio.to_thread(self._zimage_generator.unload_models)
        
        if self._lightx2v_generator and self._lightx2v_generator._loaded:
            logger.info("Unloading LightX2V models...")
            await asyncio.to_thread(self._lightx2v_generator.unload_models)
        
        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    async def _load_video_models(self) -> None:
        """Load video generation models."""
        from app.services.ltx2_generator import LTX2Generator
        
        # Load LTX-2 for video generation (includes native 2x upsampling)
        if self._ltx2_generator is None:
            self._ltx2_generator = LTX2Generator(self._settings)
        
        if not self._ltx2_generator._loaded:
            logger.info("Loading LTX-2 19B models...")
            await asyncio.to_thread(self._ltx2_generator.load_models)
        
        logger.info("Successfully switched to Video Mode")

    async def _unload_video_models(self) -> None:
        """Unload video generation models and free VRAM."""
        if self._ltx2_generator and self._ltx2_generator._loaded:
            logger.info("Unloading LTX-2 models...")
            await asyncio.to_thread(self._ltx2_generator.unload_models)
        
        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def get_image_generator(self) -> "ImageGenerator":
        """Get the ImageGenerator for text-to-image.
        
        Returns:
            ImageGenerator instance
            
        Raises:
            RuntimeError: If not in Image Mode or generator not loaded
        """
        if self._mode != ModelMode.IMAGE:
            raise RuntimeError(f"Not in Image Mode (current: {self._mode.value})")
        
        if self._zimage_generator is None or not self._zimage_generator._loaded:
            raise RuntimeError("Z-Image generator not loaded")
        
        return self._zimage_generator

    def get_image_editor(self) -> "ImageEditor":
        """Get the ImageEditor for image editing.
        
        Returns:
            ImageEditor instance
            
        Raises:
            RuntimeError: If not in Image Mode or generator not loaded
        """
        if self._mode != ModelMode.IMAGE:
            raise RuntimeError(f"Not in Image Mode (current: {self._mode.value})")
        
        if self._lightx2v_generator is None or not self._lightx2v_generator._loaded:
            raise RuntimeError("LightX2V generator not loaded")
        
        return self._lightx2v_generator

    def get_video_generator(self) -> "VideoGenerator":
        """Get the VideoGenerator for video generation.
        
        Returns:
            VideoGenerator instance
            
        Raises:
            RuntimeError: If not in Video Mode or generator not loaded
        """
        if self._mode != ModelMode.VIDEO:
            raise RuntimeError(f"Not in Video Mode (current: {self._mode.value})")
        
        if self._ltx2_generator is None or not self._ltx2_generator._loaded:
            raise RuntimeError("LTX-2 generator not loaded")
        
        return self._ltx2_generator
