"""LightX2V Instance Pool for concurrent processing.

This module provides a pool of LightX2V pipeline instances that enables
parallel image editing. Multiple pipelines are loaded into VRAM and managed
with an asyncio semaphore to limit concurrent access.

Key features:
- VRAM-aware pool sizing (calculates optimal instance count)
- Asyncio-based acquire/release for concurrent processing
- Graceful handling of instance failures
- Thread-safe for async operations
"""

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PooledInstance:
    """Container for a single pooled LightX2V pipeline."""
    
    pipeline: Any  # LightX2VPipeline instance
    instance_id: int
    temp_dir: tempfile.TemporaryDirectory
    in_use: bool = False


class LightX2VInstancePool:
    """Pool of LightX2V pipelines for concurrent image editing.
    
    This pool manages multiple LightX2V instances, allowing concurrent
    image processing up to the pool size. Uses asyncio semaphore for
    safe concurrent access.
    
    Example usage:
        pool = LightX2VInstancePool(settings, max_instances=4)
        pool.load_all()
        
        # In async context:
        instance = await pool.acquire()
        try:
            result = process_with_instance(instance, params)
        finally:
            await pool.release(instance)
    """
    
    def __init__(
        self, 
        settings: Any,  # Settings type
        max_instances: int = 4,
        dry_run: bool = False
    ):
        """Initialize the instance pool.
        
        Args:
            settings: Application settings with model paths
            max_instances: Maximum number of concurrent instances
            dry_run: If True, skip actual model loading
        """
        self.settings = settings
        self.max_instances = max_instances
        self.dry_run = dry_run
        
        self._instances: list[PooledInstance] = []
        self._available: asyncio.Queue[PooledInstance] = asyncio.Queue()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Whether the pool has loaded instances."""
        return self._is_loaded
    
    @property
    def size(self) -> int:
        """Current number of instances in the pool."""
        return len(self._instances)
    
    @property
    def available_count(self) -> int:
        """Number of currently available instances."""
        return self._available.qsize()
    
    def load_all(self) -> None:
        """Load all pipeline instances into VRAM.
        
        This should be called during application startup. Each instance
        is loaded with the same configuration (model, LoRA, etc.).
        """
        if self._is_loaded:
            logger.warning("Pool already loaded, skipping")
            return
        
        logger.info(
            f"Loading LightX2V instance pool with {self.max_instances} instances"
        )
        
        if self.dry_run:
            # Create mock instances for testing
            for i in range(self.max_instances):
                temp_dir = tempfile.TemporaryDirectory(prefix=f"lightx2v_pool_{i}_")
                instance = PooledInstance(
                    pipeline=None,
                    instance_id=i,
                    temp_dir=temp_dir,
                    in_use=False
                )
                self._instances.append(instance)
            
            logger.info(f"Created {self.max_instances} dry-run instances")
        else:
            # Load real pipeline instances
            self._load_real_instances()
        
        # Initialize the queue with all instances
        self._available = asyncio.Queue()
        for instance in self._instances:
            self._available.put_nowait(instance)
        
        # Create semaphore for limiting concurrent access
        self._semaphore = asyncio.Semaphore(self.max_instances)
        self._is_loaded = True
        
        logger.info(
            f"LightX2V pool loaded: {len(self._instances)} instances ready"
        )
    
    def _load_real_instances(self) -> None:
        """Load actual LightX2V pipelines."""
        from pathlib import Path
        
        # FIX: Explicitly set AI_DEVICE to cuda to prevent meta-tensor initialization issues
        # This is critical because the library defaults to lazy loading (meta device) if not set.
        try:
            import lightx2v_platform.base.global_var
            lightx2v_platform.base.global_var.AI_DEVICE = "cuda"
            logger.info("Set LightX2V AI_DEVICE to 'cuda'")
        except ImportError:
            logger.warning("Could not import lightx2v_platform to set AI_DEVICE")

        try:
            from lightx2v import LightX2VPipeline
        except ImportError as e:
            raise ImportError(
                "LightX2V is required for Qwen-Image-Edit-2511 generation. "
                "Install with: pip install -v git+https://github.com/ModelTC/LightX2V.git"
            ) from e
        
        model_path = Path(self.settings.lightx2v_model_path)
        lora_path = Path(self.settings.lightx2v_lora_path) / self.settings.lightx2v_lora_filename
        
        for i in range(self.max_instances):
            logger.info(f"Loading LightX2V instance {i + 1}/{self.max_instances}")
            
            # Create temp directory for this instance
            temp_dir = tempfile.TemporaryDirectory(prefix=f"lightx2v_pool_{i}_")
            
            # Initialize pipeline
            pipe = LightX2VPipeline(
                model_path=str(model_path.absolute()),
                model_cls="qwen-image-edit-2511",
                task="i2i",
            )
            
            # Enable CPU offloading if configured
            if self.settings.lightx2v_cpu_offload or self.settings.lightx2v_text_encoder_offload:
                pipe.enable_offload(
                    cpu_offload=self.settings.lightx2v_cpu_offload,
                    offload_granularity="block",
                    text_encoder_offload=self.settings.lightx2v_text_encoder_offload,
                    vae_offload=False,
                )
            
            # Enable LoRA
            pipe.enable_lora([{
                "path": str(lora_path.absolute()),
                "strength": self.settings.lightx2v_lora_strength
            }])
            
            # Resolution is controlled via custom_shape at generate() time 
            # (see lightx2v_generator.py). This forces exact input dimensions
            # and bypasses the resolution-based area calculation.
            
            # Create generator
            pipe.create_generator(
                attn_mode=self.settings.lightx2v_attn_mode,
                resize_mode=self.settings.lightx2v_resize_mode,
                infer_steps=self.settings.lightx2v_infer_steps,
                guidance_scale=self.settings.lightx2v_guidance_scale,
            )
            
            instance = PooledInstance(
                pipeline=pipe,
                instance_id=i,
                temp_dir=temp_dir,
                in_use=False
            )
            self._instances.append(instance)
            
            logger.info(f"Instance {i + 1}/{self.max_instances} loaded")
    
    async def acquire(self) -> PooledInstance:
        """Acquire an available pipeline instance.
        
        Blocks until an instance is available. Use in async context with
        try/finally to ensure release.
        
        Returns:
            PooledInstance ready for use
            
        Raises:
            RuntimeError: If pool is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Pool not loaded. Call load_all() first.")
        
        # Wait for semaphore (limits total concurrent usage)
        await self._semaphore.acquire()
        
        # Get an instance from the queue
        instance = await self._available.get()
        instance.in_use = True
        
        logger.debug(
            f"Acquired instance {instance.instance_id} "
            f"({self.available_count} remaining)"
        )
        
        return instance
    
    async def release(self, instance: PooledInstance) -> None:
        """Release a pipeline instance back to the pool.
        
        Args:
            instance: The instance to release
        """
        instance.in_use = False
        
        # Return to the queue
        await self._available.put(instance)
        
        # Release semaphore
        self._semaphore.release()
        
        logger.debug(
            f"Released instance {instance.instance_id} "
            f"({self.available_count} available)"
        )
    
    def unload_all(self) -> None:
        """Unload all instances and free VRAM.
        
        Called when switching VRAM modes or shutting down.
        """
        if not self._is_loaded:
            logger.info("Pool not loaded, nothing to unload")
            return
        
        logger.info(f"Unloading LightX2V pool ({len(self._instances)} instances)")
        
        import gc
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
        
        for instance in self._instances:
            try:
                # Cleanup temp directory
                if instance.temp_dir:
                    instance.temp_dir.cleanup()
                
                # Delete pipeline
                if instance.pipeline is not None:
                    del instance.pipeline
                    
            except Exception as e:
                logger.warning(f"Error unloading instance {instance.instance_id}: {e}")
        
        self._instances.clear()
        self._is_loaded = False
        
        # Force garbage collection and CUDA cleanup
        gc.collect()
        if has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("LightX2V pool unloaded")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._is_loaded:
            self.unload_all()
