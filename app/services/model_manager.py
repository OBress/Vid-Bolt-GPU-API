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
        MusicGenerator,
        Segmenter,
        VideoGenerator,
    )

from app.config import Settings

logger = logging.getLogger(__name__)


class VRAMLoadMode(str, Enum):
    """VRAM loading mode - defines which models are loaded."""
    IMAGE_GENERATION = "image_generation"  # Z-Image Turbo only
    IMAGE_EDITING = "image_editing"        # LightX2V only
    VIDEO_GENERATION = "video_generation"  # LTX-2 DistilledPipeline only (~40GB)
    AUDIO_CREATION = "audio_creation"      # ACE-Step only
    SEGMENTATION = "segmentation"          # SAM 3 only (~4-10GB)
    ALL = "all"                            # All models loaded (disabled)


# For backwards compatibility and job scheduling
class JobType(str, Enum):
    """Job type for scheduling purposes."""
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    VIDEO_GENERATION = "video_generation"
    MUSIC_GENERATION = "music_generation"
    SEGMENTATION = "segmentation"


@dataclass
class ModeStatus:
    """Status information about the current mode."""
    mode: VRAMLoadMode
    is_busy: bool = False
    active_job_id: Optional[str] = None
    loaded_models: list[str] = field(default_factory=list)
    # Switching progress fields
    is_switching: bool = False
    switching_target: Optional[str] = None
    switching_step: Optional[str] = None
    switching_progress: Optional[float] = None  # 0.0-1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "mode": self.mode.value,
            "is_busy": self.is_busy,
            "active_job_id": self.active_job_id,
            "loaded_models": self.loaded_models,
            "is_switching": self.is_switching,
            "switching_target": self.switching_target,
            "switching_step": self.switching_step,
            "switching_progress": self.switching_progress,
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
        
        # Switching progress tracking
        self._switching_target: Optional[VRAMLoadMode] = None
        self._switching_step: Optional[str] = None
        self._switching_progress: Optional[float] = None
        
        # Generator instances (lazy loaded)
        self._zimage_generator: Optional["ImageGenerator"] = None
        self._lightx2v_generator: Optional["ImageEditor"] = None
        self._ltx2_generator: Optional["VideoGenerator"] = None
        self._acestep_generator: Optional["MusicGenerator"] = None
        self._sam3_generator: Optional["Segmenter"] = None
        
        # Job manager reference (set via setter to break circular dependency)
        self._job_manager = None
        
        # Track loaded state
        self._loaded = False
        
        # Dynamic Z-Image loading in ALL mode
        # In ALL mode, Z-Image starts unloaded to provide VRAM headroom for video gen
        # It gets loaded on-demand when image gen is requested
        self._zimage_dynamic_loaded = False
        
        # Dynamic audio loading in ALL mode
        self._audio_dynamic_loaded = False
        
        # Dynamic SAM 3 loading in ALL mode
        self._sam3_dynamic_loaded = False
        
        logger.info("ModelManager initialized")

    def set_job_manager(self, job_manager) -> None:
        """Set the JobManager instance (breaks circular dependency)."""
        self._job_manager = job_manager

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
            loaded_models.append("ltx-2.3-22b")
        if self._acestep_generator and self._acestep_generator._loaded:
            loaded_models.append("ace-step-1.5")
        if self._sam3_generator and self._sam3_generator._loaded:
            loaded_models.append("sam3")
            
        return ModeStatus(
            mode=self._mode,
            is_busy=self._is_busy,
            active_job_id=self._active_job_id,
            loaded_models=loaded_models,
            is_switching=self._is_switching,
            switching_target=self._switching_target.value if self._switching_target else None,
            switching_step=self._switching_step,
            switching_progress=self._switching_progress,
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
        
        # Also check if any jobs are queued (prevents race between batch jobs)
        if self._job_manager and self._job_manager.has_pending_or_active_jobs():
            raise RuntimeError("Cannot change VRAM mode while jobs are queued or processing")
        
        if self._is_switching:
            raise RuntimeError(f"Already switching to {self._switching_target.value if self._switching_target else 'unknown'}")
        
        if mode == self._mode and self._loaded:
            logger.info(f"Already in {mode.value} mode with models loaded")
            return
        
        logger.info(f"Switching VRAM mode from {self._mode.value} to {mode.value}")
        
        # Set switching state
        self._is_switching = True
        self._switching_target = mode
        self._switching_step = "Initializing..."
        self._switching_progress = 0.0
        
        try:
            if mode == VRAMLoadMode.IMAGE_GENERATION:
                await self._switch_to_image_generation_mode()
            elif mode == VRAMLoadMode.IMAGE_EDITING:
                await self._switch_to_image_editing_mode()
            elif mode == VRAMLoadMode.VIDEO_GENERATION:
                await self._switch_to_video_generation_mode()
            elif mode == VRAMLoadMode.AUDIO_CREATION:
                await self._switch_to_audio_creation_mode()
            elif mode == VRAMLoadMode.SEGMENTATION:
                await self._switch_to_segmentation_mode()
            elif mode == VRAMLoadMode.ALL:
                await self._load_all_models()
            
            self._mode = mode
            self._loaded = True
            logger.info(f"VRAM mode set to {mode.value}")
        finally:
            # Clear switching state
            self._is_switching = False
            self._switching_target = None
            self._switching_step = None
            self._switching_progress = None
    
    def _set_switching_progress(self, step: str, progress: float) -> None:
        """Update switching progress for status reporting.
        
        Args:
            step: Description of current step
            progress: Progress percentage (0.0 to 1.0)
        """
        self._switching_step = step
        self._switching_progress = progress
        logger.info(f"Mode switch progress: {step} ({progress*100:.0f}%)")

    async def ensure_mode_for_job(self, job_type: JobType) -> bool:
        """Ensure the system can handle the given job type.
        
        This handles automatic mode switching logic:
        - If in ALL mode: Handle dynamic Z-Image loading/unloading
        - If in matching mode: Ready
        - If in different mode and busy: Return False
        - If in different mode and idle: Switch and return True
        
        Args:
            job_type: The job type that needs to run
            
        Returns:
            True if ready to proceed, False if cannot switch
        """
        # Map job type to required mode
        if job_type == JobType.MUSIC_GENERATION:
            required_mode = VRAMLoadMode.AUDIO_CREATION
        elif job_type == JobType.SEGMENTATION:
            required_mode = VRAMLoadMode.SEGMENTATION
        else:
            required_mode = VRAMLoadMode(job_type.value)
        
        # ALL mode: Handle dynamic Z-Image loading/unloading
        if self._mode == VRAMLoadMode.ALL:
            if job_type == JobType.IMAGE_GENERATION:
                # Load Z-Image if not loaded (fast operation ~2-3s)
                if not self._zimage_dynamic_loaded:
                    logger.info("ALL mode: Dynamically loading Z-Image for image generation...")
                    await self._load_zimage(target_mode=VRAMLoadMode.ALL)
                    self._zimage_dynamic_loaded = True
                    logger.info("ALL mode: Z-Image loaded dynamically")
                return True
            elif job_type == JobType.VIDEO_GENERATION:
                # Unload Z-Image if loaded to free VRAM for video gen
                if self._zimage_dynamic_loaded:
                    logger.info("ALL mode: Unloading Z-Image to free VRAM for video generation...")
                    await self._unload_zimage()
                    self._zimage_dynamic_loaded = False
                    logger.info("ALL mode: Z-Image unloaded, VRAM freed for video generation")
                # Unload ACE-Step if loaded to free VRAM for video gen
                if self._audio_dynamic_loaded:
                    logger.info("ALL mode: Unloading ACE-Step to free VRAM for video generation...")
                    await self._unload_acestep()
                    self._audio_dynamic_loaded = False
                    logger.info("ALL mode: ACE-Step unloaded, VRAM freed for video generation")
                return True
            elif job_type == JobType.IMAGE_EDITING:
                # Unload Z-Image if loaded to free VRAM for image editing (LightX2V)
                if self._zimage_dynamic_loaded:
                    logger.info("ALL mode: Unloading Z-Image to free VRAM for image editing...")
                    await self._unload_zimage()
                    self._zimage_dynamic_loaded = False
                    logger.info("ALL mode: Z-Image unloaded, VRAM freed for image editing")
                return True
            elif job_type == JobType.MUSIC_GENERATION:
                # Dynamically load audio models in ALL mode
                if not self._audio_dynamic_loaded:
                    logger.info("ALL mode: Dynamically loading audio models...")
                    await self._load_acestep()
                    self._audio_dynamic_loaded = True
                    logger.info("ALL mode: Audio models loaded dynamically")
                return True
            elif job_type == JobType.SEGMENTATION:
                # Dynamically load SAM 3 in ALL mode (lightweight ~4GB)
                if not self._sam3_dynamic_loaded:
                    logger.info("ALL mode: Dynamically loading SAM 3 for segmentation...")
                    await self._load_sam3()
                    self._sam3_dynamic_loaded = True
                    logger.info("ALL mode: SAM 3 loaded dynamically")
                return True
            else:
                # Unknown job type - just return True
                return True
        
        # Already in the right mode
        if self._mode == required_mode:
            return True
        
        # VIDEO_GENERATION mode: dynamically load SAM 3 for segmentation jobs
        # SAM 3 is lightweight (~4GB) and can coexist with LTX-2 (~66GB) on 80GB GPU
        if self._mode == VRAMLoadMode.VIDEO_GENERATION and job_type == JobType.SEGMENTATION:
            if not self._sam3_dynamic_loaded:
                logger.info("VIDEO mode: Dynamically loading SAM 3 for segmentation (~4GB)...")
                await self._load_sam3()
                self._sam3_dynamic_loaded = True
                logger.info("VIDEO mode: SAM 3 loaded dynamically alongside LTX-2")
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
        
        # Unload other models
        self._set_switching_progress("Unloading SAM 3...", 0.03)
        await self._unload_sam3()
        self._set_switching_progress("Unloading ACE-Step...", 0.05)
        await self._unload_acestep()
        self._set_switching_progress("Unloading LightX2V...", 0.1)
        await self._unload_lightx2v()
        self._set_switching_progress("Unloading LTX-2...", 0.3)
        await self._unload_ltx2()
        
        # Load Z-Image with dedicated mode (8 instances for concurrent pool)
        self._set_switching_progress("Loading Z-Image Turbo...", 0.5)
        await self._load_zimage(target_mode=VRAMLoadMode.IMAGE_GENERATION)
        
        self._set_switching_progress("Finalizing...", 1.0)
        logger.info("Successfully switched to Image Generation Mode")

    async def _switch_to_image_editing_mode(self) -> None:
        """Switch to Image Editing mode (LightX2V only)."""
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Image Editing Mode (LightX2V only)...")
        
        # Unload other models
        self._set_switching_progress("Unloading SAM 3...", 0.03)
        await self._unload_sam3()
        self._set_switching_progress("Unloading ACE-Step...", 0.05)
        await self._unload_acestep()
        self._set_switching_progress("Unloading Z-Image...", 0.1)
        await self._unload_zimage()
        self._set_switching_progress("Unloading LTX-2...", 0.3)
        await self._unload_ltx2()
        
        # Load LightX2V
        self._set_switching_progress("Loading LightX2V...", 0.5)
        await self._load_lightx2v()
        
        self._set_switching_progress("Finalizing...", 1.0)
        logger.info("Successfully switched to Image Editing Mode")

    async def _switch_to_video_generation_mode(self) -> None:
        """Switch to Video Generation mode (LTX-2 DistilledPipeline only).
        
        This mode is optimized for Image-to-Video generation with:
        - 1 start frame (required)
        - 1 optional end frame
        
        Uses the fast DistilledPipeline with fixed 8+4 step schedule (~40GB VRAM).
        """
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Video Generation Mode (LTX-2 DistilledPipeline only)...")
        
        # Unload other models
        self._set_switching_progress("Unloading SAM 3...", 0.02)
        await self._unload_sam3()
        self._set_switching_progress("Unloading ACE-Step...", 0.03)
        await self._unload_acestep()
        self._set_switching_progress("Unloading Z-Image...", 0.05)
        await self._unload_zimage()
        self._set_switching_progress("Unloading LightX2V...", 0.1)
        await self._unload_lightx2v()
        self._set_switching_progress("Unloading previous LTX-2...", 0.15)
        await self._unload_ltx2()
        
        # Load LTX-2 with DistilledPipeline only (this is the slow part)
        self._set_switching_progress("Loading LTX-2 models (this takes 2-3 minutes)...", 0.2)
        await self._load_ltx2()
        
        self._set_switching_progress("Finalizing...", 1.0)
        logger.info("Successfully switched to Video Generation Mode (~40GB VRAM)")

    async def _switch_to_audio_creation_mode(self) -> None:
        """Switch to Audio Creation mode (ACE-Step only).
        
        This mode is optimized for music generation.
        Uses ~4GB (ACE-Step) VRAM.
        """
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Audio Creation Mode (ACE-Step only)...")
        
        # Unload other models
        self._set_switching_progress("Unloading SAM 3...", 0.05)
        await self._unload_sam3()
        self._set_switching_progress("Unloading Z-Image...", 0.1)
        await self._unload_zimage()
        self._set_switching_progress("Unloading LightX2V...", 0.2)
        await self._unload_lightx2v()
        self._set_switching_progress("Unloading LTX-2...", 0.3)
        await self._unload_ltx2()
        
        # Load audio models
        self._set_switching_progress("Loading ACE-Step 1.5...", 0.5)
        await self._load_acestep()
        
        self._set_switching_progress("Finalizing...", 1.0)
        logger.info("Successfully switched to Audio Creation Mode")


    async def _switch_to_segmentation_mode(self) -> None:
        """Switch to Segmentation mode (SAM 3 only).
        
        This mode is optimized for image/video segmentation.
        Uses ~4-10GB VRAM.
        """
        if self._is_busy:
            raise RuntimeError("Cannot switch modes while a job is in progress")
        
        logger.info("Switching to Segmentation Mode (SAM 3 only)...")
        
        # Unload other models
        self._set_switching_progress("Unloading Z-Image...", 0.1)
        await self._unload_zimage()
        self._set_switching_progress("Unloading LightX2V...", 0.15)
        await self._unload_lightx2v()
        self._set_switching_progress("Unloading LTX-2...", 0.2)
        await self._unload_ltx2()
        self._set_switching_progress("Unloading ACE-Step...", 0.3)
        await self._unload_acestep()
        
        # Load SAM 3
        self._set_switching_progress("Loading SAM 3...", 0.5)
        await self._load_sam3()
        
        self._set_switching_progress("Finalizing...", 1.0)
        logger.info("Successfully switched to Segmentation Mode (~4-10GB VRAM)")

    async def _load_all_models(self) -> None:
        """Load all models into VRAM (ALL mode).
        
        Note: Z-Image is NOT loaded initially to provide VRAM headroom for video generation.
        Z-Image loads dynamically when an image generation request comes in (~2-3s).
        This allows video generation to have enough VRAM (~20GB freed).
        """
        logger.info("Loading ALL models into VRAM (Z-Image will load dynamically)...")
        
        # Clear any stale GPU memory before loading (helps with container restarts)
        self._set_switching_progress("Clearing GPU cache...", 0.05)
        self._force_gc()
        logger.info("Cleared GPU cache before loading models")
        
        # CRITICAL: Unload Z-Image if it was already loaded from a previous mode
        # This frees ~20GB VRAM for video generation headroom
        self._set_switching_progress("Unloading Z-Image for VRAM headroom...", 0.1)
        if self._zimage_generator is not None and self._zimage_generator._loaded:
            logger.info("Unloading Z-Image to free VRAM for video generation headroom...")
            await self._unload_zimage()
            self._force_gc()  # Force cleanup after unload
            logger.info("Z-Image unloaded successfully")
        
        # Mark Z-Image as dynamically managed (will load on-demand for image gen)
        self._zimage_dynamic_loaded = False
        logger.info("Z-Image will load dynamically on demand (saves ~20GB VRAM)")
        
        self._set_switching_progress("Loading LightX2V (1 instance for ALL mode)...", 0.2)
        await self._load_lightx2v()
        self._set_switching_progress("Loading LTX-2 (this takes 2-3 minutes)...", 0.5)
        await self._load_ltx2(lean_cache=True)  # Lean cache: only transformer, saves ~24GB for activations
        
        self._set_switching_progress("Finalizing...", 1.0)
        logger.info("ALL mode loaded: LightX2V + LTX-2 ready, Z-Image loads on demand")

    # Legacy compatibility
    async def load_all_models(self) -> None:
        """Public wrapper for loading all models."""
        await self.set_vram_mode(VRAMLoadMode.ALL)

    # --- Individual Model Load/Unload ---

    async def _load_zimage(self, target_mode: VRAMLoadMode = None) -> None:
        """Load Z-Image Turbo model.
        
        Uses vectorized batching internally - no pool needed.
        """
        from app.services.zimage_generator import ZImageGenerator
        
        if self._zimage_generator is not None and self._zimage_generator._loaded:
            logger.info("Z-Image already loaded, skipping")
            return
        
        # Create new generator
        self._zimage_generator = ZImageGenerator(self._settings)
        
        if not self._zimage_generator._loaded:
            logger.info("Loading Z-Image Turbo models...")
            await asyncio.to_thread(self._zimage_generator.load_models)

    async def _unload_zimage(self) -> None:
        """Unload Z-Image Turbo model."""
        if self._zimage_generator and self._zimage_generator._loaded:
            logger.info("Unloading Z-Image Turbo models...")
            await asyncio.to_thread(self._zimage_generator.unload_models)
        self._zimage_generator = None  # Drop all references to allow GC
        self._force_gc()

    async def _load_lightx2v(self) -> None:
        """Load LightX2V (Qwen-Image-Edit) model.
        
        Instance count is determined by the current mode:
        - IMAGE_EDITING: More instances (5) since we have full VRAM
        - ALL mode: Fewer instances (2) since sharing with other models
        """
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        # Determine instance count based on target mode
        # Check switching_target when switching, otherwise use current mode
        target_mode = self._switching_target if self._is_switching else self._mode
        if target_mode == VRAMLoadMode.IMAGE_EDITING:
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
        self._lightx2v_generator = None  # Drop all references to allow GC
        self._force_gc()

    async def _load_ltx2(self, lean_cache: bool = False) -> None:
        """Load LTX-2 video model (DistilledPipeline only).
        
        Loads only the DistilledPipeline which supports I2V with 1-2 keyframes.
        Uses ~40GB VRAM.
        
        Args:
            lean_cache: If True, only cache the transformer (not text_encoder/embeddings).
                       Used in ALL mode to save ~24GB VRAM for longer videos.
        """
        from app.services.ltx2_generator import LTX2Generator
        
        if self._ltx2_generator is None:
            self._ltx2_generator = LTX2Generator(self._settings)
        
        # Set caching mode before loading — checked by _patch_model_ledger_caching
        self._ltx2_generator.lean_cache = lean_cache
        
        if not self._ltx2_generator._loaded:
            logger.info(f"Loading LTX-2.3 22B (DistilledPipeline, lean_cache={lean_cache})...")
            await asyncio.to_thread(self._ltx2_generator.load_models)

    async def _unload_ltx2(self) -> None:
        """Unload LTX-2 model."""
        if self._ltx2_generator and self._ltx2_generator._loaded:
            logger.info("Unloading LTX-2 models...")
            await asyncio.to_thread(self._ltx2_generator.unload_models)
        self._ltx2_generator = None  # Drop all references to allow GC
        self._force_gc()

    async def _load_acestep(self) -> None:
        """Load ACE-Step 1.5 music generator."""
        from app.services.acestep_generator import ACEStepGenerator
        
        if self._acestep_generator is not None and self._acestep_generator._loaded:
            logger.info("ACE-Step already loaded, skipping")
            return
        
        self._acestep_generator = ACEStepGenerator(self._settings)
        
        if not self._acestep_generator._loaded:
            logger.info("Loading ACE-Step 1.5 models...")
            await asyncio.to_thread(self._acestep_generator.load_models)

    async def _unload_acestep(self) -> None:
        """Unload ACE-Step model."""
        if self._acestep_generator and self._acestep_generator._loaded:
            logger.info("Unloading ACE-Step models...")
            await asyncio.to_thread(self._acestep_generator.unload_models)
        self._acestep_generator = None  # Drop all references to allow GC
        self._force_gc()


    async def _load_sam3(self) -> None:
        """Load SAM 3 segmentation model."""
        from app.services.sam3_generator import SAM3Generator
        
        if self._sam3_generator is not None and self._sam3_generator._loaded:
            logger.info("SAM 3 already loaded, skipping")
            return
        
        self._sam3_generator = SAM3Generator(self._settings)
        
        if not self._sam3_generator._loaded:
            logger.info("Loading SAM 3 models...")
            await asyncio.to_thread(self._sam3_generator.load_models)

    async def _unload_sam3(self) -> None:
        """Unload SAM 3 model."""
        if self._sam3_generator and self._sam3_generator._loaded:
            logger.info("Unloading SAM 3 models...")
            await asyncio.to_thread(self._sam3_generator.unload_models)
        self._sam3_generator = None  # Drop all references to allow GC
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

    def get_music_generator(self) -> "MusicGenerator":
        """Get the MusicGenerator for music generation.
        
        Returns:
            MusicGenerator instance
            
        Raises:
            RuntimeError: If not in a valid mode or generator not loaded
        """
        valid_modes = [VRAMLoadMode.AUDIO_CREATION, VRAMLoadMode.ALL]
        if self._mode not in valid_modes:
            raise RuntimeError(f"Not in valid mode for music generation (current: {self._mode.value})")
        
        if self._acestep_generator is None or not self._acestep_generator._loaded:
            raise RuntimeError("ACE-Step generator not loaded")
        
        return self._acestep_generator


    def get_segmenter(self) -> "Segmenter":
        """Get the Segmenter for image/video segmentation.
        
        Returns:
            Segmenter instance
            
        Raises:
            RuntimeError: If not in a valid mode or generator not loaded
        """
        valid_modes = [VRAMLoadMode.SEGMENTATION, VRAMLoadMode.ALL, VRAMLoadMode.VIDEO_GENERATION]
        if self._mode not in valid_modes:
            raise RuntimeError(f"Not in valid mode for segmentation (current: {self._mode.value})")
        
        if self._sam3_generator is None or not self._sam3_generator._loaded:
            raise RuntimeError("SAM 3 generator not loaded")
        
        return self._sam3_generator

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
