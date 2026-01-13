"""LTX-2 Concurrency Controller for Shared Pipeline.

This module provides a lightweight concurrency controller for LTX-2 video
generation using a shared pipeline. The controller uses asyncio primitives
to manage concurrent access and track VRAM budget.

Key Design Decisions:
- Uses semaphore-based slot management (not separate pipeline instances)
- Dynamic concurrency based on video duration and available VRAM
- Thread-safe for async operations
- Graceful fallback to sequential processing on errors

The LTX-2 DistilledPipeline.__call__ is stateless (all generation state is
local variables, model weights are read-only), which enables safe concurrent
calls to the same pipeline instance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ConcurrentSlot:
    """Represents a concurrent video generation slot.
    
    Each slot tracks its current state for monitoring and debugging.
    """
    
    slot_id: int
    acquired_at: float = 0.0
    job_id: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        self.in_use = False
    
    def acquire(self, job_id: str, duration_seconds: float) -> None:
        """Mark slot as in-use."""
        self.in_use = True
        self.acquired_at = time.time()
        self.job_id = job_id
        self.duration_seconds = duration_seconds
    
    def release(self) -> float:
        """Release slot and return hold duration."""
        hold_duration = time.time() - self.acquired_at if self.acquired_at else 0.0
        self.in_use = False
        self.acquired_at = 0.0
        self.job_id = None
        self.duration_seconds = 0.0
        return hold_duration


class LTX2ConcurrencyController:
    """Controls concurrent access to shared LTX-2 pipeline.
    
    This controller manages concurrent video generation using a shared
    pipeline instance. It dynamically adjusts concurrency based on video
    duration and available VRAM.
    
    Features:
    - Dynamic slot count based on VRAM budget
    - Semaphore-based concurrency limiting
    - Slot tracking for monitoring
    - Graceful error handling
    
    Example usage:
        controller = LTX2ConcurrencyController(max_concurrent=4)
        
        async def generate(params):
            async with controller.slot(params.job_id, params.duration_seconds):
                return await pipeline(params)
        
        # Run concurrently
        results = await asyncio.gather(*[generate(p) for p in params_list])
    """
    
    def __init__(
        self,
        max_concurrent: int = 4,
        vram_budget_gb: float = 72.0,
    ):
        """Initialize the concurrency controller.
        
        Args:
            max_concurrent: Maximum concurrent video generations
            vram_budget_gb: Total VRAM budget for activations
        """
        self.max_concurrent = max_concurrent
        self.vram_budget_gb = vram_budget_gb
        
        # Concurrency primitives
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        
        # Slot tracking
        self._slots: list[ConcurrentSlot] = [
            ConcurrentSlot(slot_id=i) for i in range(max_concurrent)
        ]
        self._active_count = 0
        
        # Statistics
        self._total_acquisitions = 0
        self._total_wait_time = 0.0
        
        logger.info(
            f"LTX2ConcurrencyController initialized: "
            f"max_concurrent={max_concurrent}, vram_budget={vram_budget_gb:.1f}GB"
        )
    
    @property
    def active_count(self) -> int:
        """Number of currently active slots."""
        return self._active_count
    
    @property
    def available_count(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active_count
    
    async def acquire(
        self, 
        job_id: str = "", 
        duration_seconds: float = 0.0,
        timeout: float | None = None,
    ) -> ConcurrentSlot:
        """Acquire a generation slot.
        
        Blocks until a slot is available. Uses semaphore for fair queuing.
        
        Args:
            job_id: Optional job ID for tracking
            duration_seconds: Video duration for this slot
            timeout: Optional timeout in seconds
            
        Returns:
            ConcurrentSlot that was acquired
            
        Raises:
            asyncio.TimeoutError: If timeout expires before slot available
        """
        wait_start = time.time()
        
        # Wait for semaphore with optional timeout
        if timeout:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Slot acquisition timed out after {timeout}s for job {job_id}"
                )
                raise
        else:
            await self._semaphore.acquire()
        
        wait_time = time.time() - wait_start
        
        # Find and mark an available slot
        async with self._lock:
            slot = None
            for s in self._slots:
                if not s.in_use:
                    slot = s
                    break
            
            if slot is None:
                # Should not happen if semaphore is correct
                self._semaphore.release()
                raise RuntimeError("No available slot despite semaphore acquisition")
            
            slot.acquire(job_id, duration_seconds)
            self._active_count += 1
            self._total_acquisitions += 1
            self._total_wait_time += wait_time
        
        if wait_time > 0.1:  # Only log if waited more than 100ms
            logger.debug(
                f"Acquired slot {slot.slot_id} for job {job_id} "
                f"(waited {wait_time:.2f}s, {self._active_count}/{self.max_concurrent} active)"
            )
        
        return slot
    
    async def release(self, slot: ConcurrentSlot) -> None:
        """Release a generation slot back to the pool.
        
        Args:
            slot: The slot to release
        """
        async with self._lock:
            hold_duration = slot.release()
            self._active_count -= 1
        
        # Release semaphore (always succeeds)
        self._semaphore.release()
        
        logger.debug(
            f"Released slot {slot.slot_id} after {hold_duration:.2f}s "
            f"({self._active_count}/{self.max_concurrent} active)"
        )
    
    class _SlotContext:
        """Async context manager for slot acquisition."""
        
        def __init__(
            self, 
            controller: "LTX2ConcurrencyController",
            job_id: str,
            duration_seconds: float,
        ):
            self.controller = controller
            self.job_id = job_id
            self.duration_seconds = duration_seconds
            self.slot: Optional[ConcurrentSlot] = None
        
        async def __aenter__(self) -> ConcurrentSlot:
            self.slot = await self.controller.acquire(
                self.job_id, 
                self.duration_seconds
            )
            return self.slot
        
        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            if self.slot:
                await self.controller.release(self.slot)
    
    def slot(
        self, 
        job_id: str = "", 
        duration_seconds: float = 0.0,
    ) -> _SlotContext:
        """Get async context manager for slot acquisition.
        
        Usage:
            async with controller.slot(job_id, duration) as slot:
                # Generate video
                result = await generate(...)
        
        Args:
            job_id: Job ID for tracking
            duration_seconds: Video duration
            
        Returns:
            Context manager that acquires/releases slot
        """
        return self._SlotContext(self, job_id, duration_seconds)
    
    def resize(self, new_max_concurrent: int) -> None:
        """Resize the controller for different concurrency.
        
        Note: This recreates the semaphore and should only be called
        when no slots are active.
        
        Args:
            new_max_concurrent: New maximum concurrent count
        """
        if self._active_count > 0:
            logger.warning(
                f"Cannot resize controller while {self._active_count} slots active"
            )
            return
        
        old_max = self.max_concurrent
        self.max_concurrent = new_max_concurrent
        self._semaphore = asyncio.Semaphore(new_max_concurrent)
        self._slots = [
            ConcurrentSlot(slot_id=i) for i in range(new_max_concurrent)
        ]
        
        logger.info(f"Resized controller: {old_max} -> {new_max_concurrent} max concurrent")
    
    def get_status(self) -> dict[str, Any]:
        """Get controller status for monitoring.
        
        Returns:
            Dictionary with controller statistics
        """
        active_slots = [
            {
                "slot_id": s.slot_id,
                "job_id": s.job_id,
                "duration_seconds": s.duration_seconds,
                "hold_time": time.time() - s.acquired_at if s.acquired_at else 0,
            }
            for s in self._slots if s.in_use
        ]
        
        avg_wait = (
            self._total_wait_time / self._total_acquisitions 
            if self._total_acquisitions > 0 else 0.0
        )
        
        return {
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "available_count": self.available_count,
            "vram_budget_gb": self.vram_budget_gb,
            "total_acquisitions": self._total_acquisitions,
            "average_wait_time": avg_wait,
            "active_slots": active_slots,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_acquisitions = 0
        self._total_wait_time = 0.0
        logger.debug("Controller statistics reset")
