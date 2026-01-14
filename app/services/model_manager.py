"""Model Manager - 4-Mode VRAM Loading System.

This module provides the ModelManager service for dynamically switching between
4 VRAM loading modes, efficiently managing GPU VRAM by loading only the required
models for each use case.

Modes:
- IMAGE_GENERATION: ZImageGenerator only (text-to-image)
- IMAGE_EDITING: LightX2VImageEditGenerator only (image editing)
- VIDEO_GENERATION: LTX2Generator only (video generation)
- ALL: All models loaded simultaneously
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


class VRAMLoadMode(str, Enum):
    """VRAM loading mode - defines which models are loaded."""
    IMAGE_GENERATION = "image_generation"  # Z-Image Turbo only
    IMAGE_EDITING = "image_editing"        # LightX2V only
    VIDEO_GENERATION = "video_generation"  # LTX-2 only
    ALL = "all"                            # All models loaded


# For backwards compatibility and job scheduling
class JobType(str, Enum):
    """Job type for scheduling purposes."""
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    VIDEO_GENERATION = "video_generation"


@dataclass
class ModeStatus:
    """Status information about the current mode."""
    mode: VRAMLoadMode
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
    """Manages dynamic loading/unloading of AI models.
    
    This service orchestrates switching between 4 VRAM loading modes,
    ensuring only the required models are loaded to manage VRAM efficiently.
    
    Attributes:
        current_mode: The currently active VRAM loading mode
        is_busy: Whether a generation job is in progress
        active_job_id: ID of the current job (if any)
    """

    def __init__(self, settings: Settings):
        """Initialize the ModelManager.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._mode = VRAMLoadMode.IMAGE_GENERATION  # Default mode
        self._is_switching = False
        self._is_busy = False
        self._active_job_id: Optional[str] = None
        self._lock = asyncio.Lock()
        
        # Generator instances (lazy loaded)
        self._zimage_generator: Optional["ImageGenerator"] = None
        self._lightx2v_generator: Optional["ImageEditor"] = None
        self._ltx2_generator: Optional["VideoGenerator"] = None
        
        # Track loaded state
        self._loaded = False
        
        logger.info("ModelManager initialized")

    @property
    def current_mode(self) -> VRAMLoadMode:
        """Get the current VRAM loading mode."""
        return self._mode

    @property
    def vram_mode(self) -> VRAMLoadMode:
        """Alias for current_mode (backwards compatibility)."""
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
        
        # Debug logging for status
        z_loaded = self._zimage_generator and self._zimage_generator._loaded
        lx2v_loaded = self._lightx2v_generator and self._lightx2v_generator._loaded
        ltx2_loaded = self._ltx2_generator and self._ltx2_generator._loaded
        
        if self._mode == VRAMLoadMode.ALL and not (z_loaded or lx2v_loaded or ltx2_loaded):
             logger.debug(f"Status check: Mode={self._mode}, Z-Img={z_loaded}, LightX2V={lx2v_loaded}, LTX2={ltx2_loaded}")
        
        if z_loaded:
            loaded_models.append("z-image-turbo")
        if lx2v_loaded:
            loaded_models.append("qwen-image-edit-2511")
        if ltx2_loaded:
            loaded_models.append("ltx-2-19b")
            
        return ModeStatus(
            mode=self._mode,
            is_busy=self._is_busy,
            active_job_id=self._active_job_id,
            loaded_models=loaded_models,
        )

    async def set_vram_mode(self, mode: VRAMLoadMode) -> None:
        """Set the VRAM loading mode.
        
        Args:
            mode: Target VRAM mode
            
        Raises:
            RuntimeError: If currently busy with a job
        """
        if self._is_busy:
            raise RuntimeError("Cannot change VRAM mode while a job is in progress")
        
        if mode == self._mode and self._loaded:
            logger.info(f"Already in {mode.value} mode with models loaded")
            return
        
        logger.info(f"Switching VRAM mode from {self._mode.value} to {mode.value}")
        
        if mode == VRAMLoadMode.IMAGE_GENERATION:
            await self._switch_to_image_generation_mode()
        elif mode == VRAMLoadMode.IMAGE_EDITING:
            await self._switch_to_image_editing_mode()
        elif mode == VRAMLoadMode.VIDEO_GENERATION:
            await self._switch_to_video_generation_mode()
        elif mode == VRAMLoadMode.ALL:
            await self._load_all_models()
        
        self._mode = mode
        self._loaded = True
        logger.info(f"VRAM mode set to {mode.value}")

    async def ensure_mode_for_job(self, job_type: JobType) -> bool:
        """Ensure the system can handle the given job type.
        
        This handles automatic mode switching logic:
        - If in ALL mode: Always ready
        - If in matching mode: Ready
        - If in different mode and busy: Return False
        - If in different mode and idle: Switch and return True
        
        Args:
            job_type: The job type that needs to run
            
        Returns:
            True if ready to proceed, False if cannot switch
        """
        # Map job type to required mode
        required_mode = VRAMLoadMode(job_type.value)
        
        # ALL mode can handle any job
        if self._mode == VRAMLoadMode.ALL:
            return True
        
        # Already in the right mode
        if self._mode == required_mode:
            return True
        
        # Need to switch - check if busy
        if self._is_busy:
            logger.warning(
                f"Cannot auto-switch to {required_mode.value} because system is busy in {self._mode.value} mode."
            )
            return False
        
        # Switch mode
        logger.info(f"Auto-switching from {self._mode.value} to {required_mode.value}...")
        try:
            await self.set_vram_mode(required_mode)
            return True
        except Exception as e:
            logger.error(f"Auto-switch failed: {e}")
            return False

    # Legacy compatibility
    async def ensure_mode(self, target_mode) -> bool:
        """Legacy compatibility wrapper for ensure_mode_for_job."""
        # Map old ModelMode to new JobType
        mode_str = target_mode.value if hasattr(target_mode, 'value') else str(target_mode)
        if mode_str == "image":
            # Default image mode to image_generation
            return await self.ensure_mode_for_job(JobType.IMAGE_GENERATION)
        elif mode_str == "video":
            return await self.ensure_mode_for_job(JobType.VIDEO_GENERATION)
        return False

    async def _switch_to_image_generation_mode(self) -> None:
        """Switch to Image Generation mode (Z-Image Turbo only)."""
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Image Generation Mode (Z-Image only)...")
        self._is_switching = True
        
        try:
            # Unload other models
            await self._unload_lightx2v()
            await self._unload_ltx2()
            
            # Load Z-Image with dedicated mode (8 instances for concurrent pool)
            await self._load_zimage(target_mode=VRAMLoadMode.IMAGE_GENERATION)
            
            logger.info("Successfully switched to Image Generation Mode")
        finally:
            self._is_switching = False

    async def _switch_to_image_editing_mode(self) -> None:
        """Switch to Image Editing mode (LightX2V only)."""
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Image Editing Mode (LightX2V only)...")
        self._is_switching = True
        
        try:
            # Unload other models
            await self._unload_zimage()
            await self._unload_ltx2()
            
            # Load LightX2V
            await self._load_lightx2v()
            
            logger.info("Successfully switched to Image Editing Mode")
        finally:
            self._is_switching = False

    async def _switch_to_video_generation_mode(self) -> None:
        """Switch to Video Generation mode (LTX-2 only)."""
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Video Generation Mode (LTX-2 only)...")
        self._is_switching = True
        
        try:
            # Unload other models
            await self._unload_zimage()
            await self._unload_lightx2v()
            
            # Load LTX-2
            await self._load_ltx2()
            
            logger.info("Successfully switched to Video Generation Mode")
        finally:
            self._is_switching = False

    async def _load_all_models(self) -> None:
        """Load all models into VRAM (ALL mode)."""
        logger.info("Loading ALL models into VRAM...")
        self._is_switching = True
        
        # Clear any stale GPU memory before loading (helps with container restarts)
        self._force_gc()
        logger.info("Cleared GPU cache before loading models")
        
        try:
            # Pass ALL mode explicitly so Z-Image uses 1 instance, not 8
            await self._load_zimage(target_mode=VRAMLoadMode.ALL)
            await self._load_lightx2v()
            await self._load_ltx2()
            
            logger.info("All models loaded successfully")
        finally:
            self._is_switching = False

    # Legacy compatibility
    async def load_all_models(self) -> None:
        """Public wrapper for loading all models."""
        await self.set_vram_mode(VRAMLoadMode.ALL)

    # --- Individual Model Load/Unload ---

    async def _load_zimage(self, target_mode: VRAMLoadMode = None) -> None:
        """Load Z-Image Turbo model with concurrent instance pool.
        
        Instance count is determined by the target mode:
        - IMAGE_GENERATION: More instances (8) since we have full VRAM
        - ALL mode: Single instance (1) since sharing with other models
        
        Args:
            target_mode: The mode being switched to (defaults to current mode)
        """
        from app.services.zimage_generator import ZImageGenerator
        from app.services.zimage_pool import ZImageInstancePool
        
        # Use target_mode if provided, otherwise use current mode
        effective_mode = target_mode if target_mode is not None else self._mode
        
        # Determine instance count based on target mode
        if effective_mode == VRAMLoadMode.IMAGE_GENERATION:
            max_instances = self._settings.zimage_max_instances_dedicated
            logger.info(f"Loading Z-Image in dedicated mode with {max_instances} instances")
        else:
            # ALL mode or any other - use single instance
            max_instances = self._settings.zimage_max_instances_all
            logger.info(f"Loading Z-Image in shared mode with {max_instances} instance(s)")
        
        # Always recreate generator to get correct instance count for the mode
        if self._zimage_generator is not None and self._zimage_generator._loaded:
            # Already loaded - check if we need to reload with different instance count
            current_pool_size = (
                self._zimage_generator._pool.size 
                if self._zimage_generator._pool else 0
            )
            if current_pool_size == max_instances:
                logger.info(f"Z-Image already loaded with {max_instances} instances, skipping")
                return
            else:
                logger.info(f"Reloading Z-Image: {current_pool_size} -> {max_instances} instances")
                await asyncio.to_thread(self._zimage_generator.unload_models)
        
        # Create new generator with the appropriate instance count
        self._zimage_generator = ZImageGenerator(
            self._settings,
            max_instances=max_instances
        )
        
        if not self._zimage_generator._loaded:
            logger.info("Loading Z-Image Turbo models with concurrent pool...")
            # Load the pool if max_instances > 1
            if max_instances > 1 and not self._settings.zimage_dry_run:
                # Create and load the pool
                pool = ZImageInstancePool(
                    self._settings,
                    max_instances=max_instances,
                    dry_run=self._settings.zimage_dry_run
                )
                await asyncio.to_thread(pool.load_all)
                self._zimage_generator._pool = pool
                self._zimage_generator.is_loaded = True
            else:
                # Single instance or dry-run mode
                await asyncio.to_thread(self._zimage_generator.load_models)

    async def _unload_zimage(self) -> None:
        """Unload Z-Image Turbo model and pool."""
        if self._zimage_generator:
            # Unload pool if present
            if self._zimage_generator._pool is not None:
                logger.info("Unloading Z-Image instance pool...")
                await asyncio.to_thread(self._zimage_generator._pool.unload_all)
                self._zimage_generator._pool = None
            # Unload single pipeline if present
            if self._zimage_generator._loaded:
                logger.info("Unloading Z-Image Turbo models...")
                await asyncio.to_thread(self._zimage_generator.unload_models)
        self._force_gc()

    async def _load_lightx2v(self) -> None:
        """Load LightX2V (Qwen-Image-Edit) model.
        
        Instance count is determined by the current mode:
        - IMAGE_EDITING: More instances (5) since we have full VRAM
        - ALL mode: Fewer instances (2) since sharing with other models
        """
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        # Determine instance count based on target mode
        if self._mode == VRAMLoadMode.IMAGE_EDITING:
            max_instances = self._settings.lightx2v_max_instances_dedicated
            logger.info(f"Loading LightX2V in dedicated mode with {max_instances} instances")
        else:
            # ALL mode or any other - use conservative count
            max_instances = self._settings.lightx2v_max_instances_all
            logger.info(f"Loading LightX2V in shared mode with {max_instances} instances")
        
        # Always recreate generator to get correct instance count for the mode
        if self._lightx2v_generator is not None and self._lightx2v_generator._loaded:
            # Already loaded - check if we need to reload with different instance count
            current_pool_size = (
                self._lightx2v_generator._pool.size 
                if self._lightx2v_generator._pool else 0
            )
            if current_pool_size == max_instances:
                logger.info(f"LightX2V already loaded with {max_instances} instances, skipping")
                return
            else:
                logger.info(f"Reloading LightX2V: {current_pool_size} -> {max_instances} instances")
                await asyncio.to_thread(self._lightx2v_generator.unload_models)
        
        # Create new generator with the appropriate instance count
        self._lightx2v_generator = LightX2VImageEditGenerator(
            self._settings,
            max_instances=max_instances
        )
        
        if not self._lightx2v_generator._loaded:
            logger.info("Loading LightX2V (Qwen-Image-Edit) models...")
            await asyncio.to_thread(self._lightx2v_generator.load_models)

    async def _unload_lightx2v(self) -> None:
        """Unload LightX2V model."""
        if self._lightx2v_generator and self._lightx2v_generator._loaded:
            logger.info("Unloading LightX2V models...")
            await asyncio.to_thread(self._lightx2v_generator.unload_models)
        self._force_gc()

    async def _load_ltx2(self) -> None:
        """Load LTX-2 video model."""
        from app.services.ltx2_generator import LTX2Generator
        
        if self._ltx2_generator is None:
            self._ltx2_generator = LTX2Generator(self._settings)
        
        if not self._ltx2_generator._loaded:
            logger.info("Loading LTX-2 19B models...")
            await asyncio.to_thread(self._ltx2_generator.load_models)

    async def _unload_ltx2(self) -> None:
        """Unload LTX-2 model."""
        if self._ltx2_generator and self._ltx2_generator._loaded:
            logger.info("Unloading LTX-2 models...")
            await asyncio.to_thread(self._ltx2_generator.unload_models)
        self._force_gc()

    def _force_gc(self) -> None:
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    # --- Job Lock Management ---

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

    # --- Generator Getters ---

    def get_image_generator(self) -> "ImageGenerator":
        """Get the ImageGenerator for text-to-image.
        
        Returns:
            ImageGenerator instance
            
        Raises:
            RuntimeError: If not in a valid mode or generator not loaded
        """
        valid_modes = [VRAMLoadMode.IMAGE_GENERATION, VRAMLoadMode.ALL]
        if self._mode not in valid_modes:
            raise RuntimeError(f"Not in valid mode for image generation (current: {self._mode.value})")
        
        if self._zimage_generator is None or not self._zimage_generator._loaded:
            raise RuntimeError("Z-Image generator not loaded")
        
        return self._zimage_generator

    def get_image_editor(self) -> "ImageEditor":
        """Get the ImageEditor for image editing.
        
        Returns:
            ImageEditor instance
            
        Raises:
            RuntimeError: If not in a valid mode or generator not loaded
        """
        valid_modes = [VRAMLoadMode.IMAGE_EDITING, VRAMLoadMode.ALL]
        if self._mode not in valid_modes:
            raise RuntimeError(f"Not in valid mode for image editing (current: {self._mode.value})")
        
        if self._lightx2v_generator is None or not self._lightx2v_generator._loaded:
            raise RuntimeError("LightX2V generator not loaded")
        
        return self._lightx2v_generator

    def get_video_generator(self) -> "VideoGenerator":
        """Get the VideoGenerator for video generation.
        
        Returns:
            VideoGenerator instance
            
        Raises:
            RuntimeError: If not in a valid mode or generator not loaded
        """
        valid_modes = [VRAMLoadMode.VIDEO_GENERATION, VRAMLoadMode.ALL]
        if self._mode not in valid_modes:
            raise RuntimeError(f"Not in valid mode for video generation (current: {self._mode.value})")
        
        if self._ltx2_generator is None or not self._ltx2_generator._loaded:
            raise RuntimeError("LTX-2 generator not loaded")
        
        return self._ltx2_generator

    # --- Legacy compatibility for switch methods ---

    async def switch_to_image_mode(self) -> None:
        """Legacy: Switch to image mode (defaults to image_generation)."""
        await self.set_vram_mode(VRAMLoadMode.IMAGE_GENERATION)

    async def switch_to_video_mode(self) -> None:
        """Legacy: Switch to video mode."""
        await self.set_vram_mode(VRAMLoadMode.VIDEO_GENERATION)


# Legacy compatibility - keep ModelMode for job_manager
class ModelMode(str, Enum):
    """Legacy model mode enum for backwards compatibility."""
    NONE = "none"
    IMAGE = "image"
    VIDEO = "video"
    SWITCHING = "switching"
