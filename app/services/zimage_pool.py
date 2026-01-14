"""ZImage Instance Pool for concurrent image generation.

This module provides a pool of ZImagePipeline instances that enables
parallel image generation. Multiple pipelines are loaded into VRAM and managed
with an asyncio semaphore to limit concurrent access.

Key features:
- VRAM-aware pool sizing (calculates optimal instance count for 96GB GPU)
- Asyncio-based acquire/release for concurrent processing
- Graceful handling of instance failures
- Thread-safe for async operations
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


# Pool configuration constants
ZIMAGE_BASE_MODEL_GB = 8.0  # Base VRAM for Z-Image Turbo model weights
ZIMAGE_ACTIVATION_GB = 3.0  # Activation VRAM per concurrent generation at 1080p
ZIMAGE_MAX_POOL_SIZE = 12   # Absolute cap regardless of VRAM


@dataclass
class PooledZImageInstance:
    """Container for a single pooled ZImagePipeline."""
    
    pipeline: Any  # ZImagePipeline instance
    instance_id: int
    in_use: bool = False


class ZImageInstancePool:
    """Pool of ZImagePipeline instances for concurrent image generation.
    
    This pool manages multiple ZImage pipeline instances, allowing concurrent
    image processing up to the pool size. Uses asyncio semaphore for
    safe concurrent access.
    
    Example usage:
        pool = ZImageInstancePool(settings, max_instances=6)
        pool.load_all()
        
        # In async context:
        instance = await pool.acquire()
        try:
            result = await generate_with_instance(instance, params)
        finally:
            await pool.release(instance)
    """
    
    def __init__(
        self, 
        settings: "Settings",
        max_instances: int = 6,
        dry_run: bool = False
    ):
        """Initialize the instance pool.
        
        Args:
            settings: Application settings with model paths
            max_instances: Maximum number of concurrent instances
            dry_run: If True, skip actual model loading
        """
        self.settings = settings
        self.max_instances = min(max_instances, ZIMAGE_MAX_POOL_SIZE)
        self.dry_run = dry_run
        
        self._instances: list[PooledZImageInstance] = []
        self._available: asyncio.Queue[PooledZImageInstance] = asyncio.Queue()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()
        self._is_loaded = False
        
        # Shared components (loaded once, shared across instances)
        self._shared_text_encoder: Any = None
        self._shared_tokenizer: Any = None
        self._shared_vae: Any = None
    
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
    
    def check_vram_available(self, min_free_gb: float = 4.0) -> bool:
        """Check if there is sufficient VRAM available for generation.
        
        Args:
            min_free_gb: Minimum free VRAM required in GB
            
        Returns:
            True if enough VRAM is available, False otherwise
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return True  # Can't check, assume OK
            
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            
            if free_gb < min_free_gb:
                logger.warning(
                    f"Low VRAM: {free_gb:.1f}GB free, need {min_free_gb}GB minimum"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"VRAM check failed: {e}")
            return True  # Assume OK if check fails
    
    def load_all(self) -> None:
        """Load all pipeline instances into VRAM.
        
        This should be called during application startup. Each instance
        is loaded with shared components where possible to minimize VRAM.
        """
        if self._is_loaded:
            logger.warning("ZImage pool already loaded, skipping")
            return
        
        logger.info(
            f"Loading ZImage instance pool with {self.max_instances} instances"
        )
        
        if self.dry_run:
            # Create mock instances for testing
            for i in range(self.max_instances):
                instance = PooledZImageInstance(
                    pipeline=None,
                    instance_id=i,
                    in_use=False
                )
                self._instances.append(instance)
            
            logger.info(f"Created {self.max_instances} dry-run ZImage instances")
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
            f"ZImage pool loaded: {len(self._instances)} instances ready"
        )
    
    def _load_real_instances(self) -> None:
        """Load actual ZImagePipeline instances with shared components."""
        import torch
        from diffusers import ZImagePipeline
        
        model_path = Path(self.settings.zimage_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Z-Image model not found at {model_path.absolute()}. "
                f"Download with: huggingface-cli download Tongyi-MAI/Z-Image-Turbo "
                f"--local-dir {model_path}"
            )
        
        dtype = torch.bfloat16 if self.settings.zimage_dtype == "bfloat16" else torch.float16
        device = self.settings.zimage_device
        
        # Load first pipeline fully (will share components)
        logger.info(f"Loading primary ZImage pipeline from {model_path}")
        primary_pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        
        # Store shared components
        self._shared_text_encoder = primary_pipe.text_encoder
        self._shared_tokenizer = primary_pipe.tokenizer
        self._shared_vae = primary_pipe.vae
        
        # Create first instance
        self._instances.append(PooledZImageInstance(
            pipeline=primary_pipe,
            instance_id=0,
            in_use=False
        ))
        logger.info("Primary ZImage instance loaded (1/{})".format(self.max_instances))
        
        # Create additional instances with shared components
        for i in range(1, self.max_instances):
            logger.info(f"Loading ZImage instance {i + 1}/{self.max_instances}")
            
            # Load pipeline with shared components
            pipe = ZImagePipeline.from_pretrained(
                str(model_path),
                text_encoder=self._shared_text_encoder,
                tokenizer=self._shared_tokenizer,
                vae=self._shared_vae,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            
            instance = PooledZImageInstance(
                pipeline=pipe,
                instance_id=i,
                in_use=False
            )
            self._instances.append(instance)
            logger.info(f"Instance {i + 1}/{self.max_instances} loaded")
    
    async def acquire(self) -> PooledZImageInstance:
        """Acquire an available pipeline instance.
        
        Blocks until an instance is available. Use in async context with
        try/finally to ensure release.
        
        Returns:
            PooledZImageInstance ready for use
            
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
            f"Acquired ZImage instance {instance.instance_id} "
            f"({self.available_count} remaining)"
        )
        
        return instance
    
    async def release(self, instance: PooledZImageInstance) -> None:
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
            f"Released ZImage instance {instance.instance_id} "
            f"({self.available_count} available)"
        )
    
    def unload_all(self) -> None:
        """Unload all instances and free VRAM.
        
        Called when switching VRAM modes or shutting down.
        """
        if not self._is_loaded:
            logger.info("ZImage pool not loaded, nothing to unload")
            return
        
        logger.info(f"Unloading ZImage pool ({len(self._instances)} instances)")
        
        import gc
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
        
        for instance in self._instances:
            try:
                if instance.pipeline is not None:
                    del instance.pipeline
            except Exception as e:
                logger.warning(f"Error unloading instance {instance.instance_id}: {e}")
        
        # Clear shared components
        self._shared_text_encoder = None
        self._shared_tokenizer = None
        self._shared_vae = None
        
        self._instances.clear()
        self._is_loaded = False
        
        # Force garbage collection and CUDA cleanup
        gc.collect()
        if has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("ZImage pool unloaded")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._is_loaded:
            self.unload_all()


def calculate_optimal_pool_size(
    available_vram_gb: float = 96.0,
    other_models_loaded: bool = False
) -> int:
    """Calculate optimal number of ZImage instances for available VRAM.
    
    Args:
        available_vram_gb: Total VRAM available (default 96GB)
        other_models_loaded: Whether LightX2V/LTX-2 are also loaded
        
    Returns:
        Optimal number of instances (1 to ZIMAGE_MAX_POOL_SIZE)
    """
    # Reserve VRAM for other models if in ALL mode
    if other_models_loaded:
        # LightX2V (~28GB) + LTX-2 (~20GB) = ~48GB reserved
        available_vram_gb = max(0.0, available_vram_gb - 48.0)
    
    # Calculate: (available - base_model) / activation_per_instance
    usable_vram = available_vram_gb - ZIMAGE_BASE_MODEL_GB
    if usable_vram <= 0:
        return 1
    
    optimal = int(usable_vram / ZIMAGE_ACTIVATION_GB)
    
    # Apply bounds
    optimal = max(1, min(optimal, ZIMAGE_MAX_POOL_SIZE))
    
    logger.info(
        f"ZImage pool sizing: {available_vram_gb:.1f}GB available, "
        f"{ZIMAGE_ACTIVATION_GB}GB per instance, optimal = {optimal}"
    )
    
    return optimal
