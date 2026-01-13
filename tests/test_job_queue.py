"""Tests for JobManager Queue System."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from app.services.job_manager import JobManager
from app.services.model_manager import VRAMLoadMode, JobType
from app.models.job import JobStatus
from app.config import get_settings

@pytest.fixture
def mock_settings():
    settings = get_settings()
    # Ensure limits don't block us for these logic tests
    return settings

@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.ensure_mode_for_job = AsyncMock(return_value=True)
    mm.vram_mode = VRAMLoadMode.IMAGE_GENERATION
    mm.current_mode = VRAMLoadMode.IMAGE_GENERATION
    return mm

@pytest.fixture
def job_manager(mock_settings, mock_model_manager):
    manager = JobManager(mock_settings, mock_model_manager)
    # Mock the worker loop so it doesn't auto-run during unit tests of logic
    # We will test _select_next_job directly, or use a controlled worker
    return manager

# -----------------------------------------------------------------------------
# Test Scheduling Logic Directly
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_mode_scheduling_fifo(job_manager, mock_model_manager):
    """Test that ALL mode follows strict FIFO."""
    mock_model_manager.vram_mode = VRAMLoadMode.ALL
    mock_model_manager.current_mode = VRAMLoadMode.ALL
    
    async def dummy_task(): pass
    
    # Submit jobs
    await job_manager.submit_job("job-video-1", JobType.VIDEO_GENERATION, dummy_task)
    await job_manager.submit_job("job-image-2", JobType.IMAGE_GENERATION, dummy_task)
    
    # Check queue
    assert job_manager._pending_jobs == ["job-video-1", "job-image-2"]
    
    # Select next
    next_job = job_manager._select_next_job()
    assert next_job == "job-video-1"
    
    # Remove first
    job_manager._pending_jobs.pop(0)
    
    # Select next
    next_job = job_manager._select_next_job()
    assert next_job == "job-image-2"


@pytest.mark.asyncio
async def test_per_type_scheduling_grouping(job_manager, mock_model_manager):
    """Test that per-type mode prioritizes current mode."""
    mock_model_manager.vram_mode = VRAMLoadMode.IMAGE_GENERATION
    mock_model_manager.current_mode = VRAMLoadMode.IMAGE_GENERATION
    
    async def dummy_task(): pass
    
    # Submit: Video -> Image -> Video
    await job_manager.submit_job("job-video-1", JobType.VIDEO_GENERATION, dummy_task)
    await job_manager.submit_job("job-image-2", JobType.IMAGE_GENERATION, dummy_task)
    await job_manager.submit_job("job-video-3", JobType.VIDEO_GENERATION, dummy_task)
    
    # Queue is FIFO by arrival time: [Video-1, Image-2, Video-3]
    assert job_manager._pending_jobs == ["job-video-1", "job-image-2", "job-video-3"]
    
    # 1. Should pick Image-2 because we are in IMAGE_GENERATION mode (skipping Video-1)
    next_job = job_manager._select_next_job()
    assert next_job == "job-image-2"
    
    # simulate processing Image-2
    job_manager._pending_jobs.remove("job-image-2")
    
    # Queue: [Video-1, Video-3]
    # Now only video jobs left. Should pick Video-1 (FIFO among remaining)
    next_job = job_manager._select_next_job()
    assert next_job == "job-video-1"


@pytest.mark.asyncio
async def test_per_type_scheduling_switch_mode(job_manager, mock_model_manager):
    """Test switching from one mode to another when queue empties."""
    mock_model_manager.vram_mode = VRAMLoadMode.VIDEO_GENERATION
    mock_model_manager.current_mode = VRAMLoadMode.VIDEO_GENERATION
    
    async def dummy_task(): pass
    
    # Submit: Image -> Image
    await job_manager.submit_job("job-image-1", JobType.IMAGE_GENERATION, dummy_task)
    await job_manager.submit_job("job-image-2", JobType.IMAGE_GENERATION, dummy_task)
    
    # Current mode is VIDEO_GENERATION, but no Video jobs.
    # Should fall back to picking the first available job (Image-1) and switch
    
    next_job = job_manager._select_next_job()
    assert next_job == "job-image-1"


# -----------------------------------------------------------------------------
# Test Worker Execution
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_execution(job_manager, mock_model_manager):
    """Test that jobs are actually executed and status updated."""
    
    # Mock task
    mock_task = AsyncMock(return_value="result_url")
    
    await job_manager.submit_job("job-exec-1", JobType.IMAGE_GENERATION, mock_task, arg1="test")
    
    # Manually run process_job (bypassing the loop to test logic)
    await job_manager._process_job("job-exec-1")
    
    # Verify
    job = job_manager.get_job("job-exec-1")
    assert job.status == JobStatus.COMPLETED
    assert job.result == "result_url"
    
    # functional check
    mock_task.assert_awaited_once_with(arg1="test")
    # ensure_mode check
    mock_model_manager.ensure_mode_for_job.assert_awaited_with(JobType.IMAGE_GENERATION)


# -----------------------------------------------------------------------------
# Test Edge Cases
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_next_job_empty(job_manager):
    """Test selection on empty queue returns None."""
    assert job_manager._select_next_job() is None


@pytest.mark.asyncio
async def test_select_next_job_no_manager(job_manager):
    """Test fallback when ModelManager is not linked (FIFO)."""
    job_manager._model_manager = None
    
    async def dummy_task(): pass
    job_manager._jobs["j1"] = MagicMock(_job_type=JobType.IMAGE_GENERATION)
    job_manager._jobs["j2"] = MagicMock(_job_type=JobType.VIDEO_GENERATION)
    job_manager._pending_jobs = ["j1", "j2"]
    
    # Should pick first regardless of mode
    assert job_manager._select_next_job() == "j1"


@pytest.mark.asyncio
async def test_select_next_job_stale_reference(job_manager, mock_model_manager):
    """Test handling of job IDs in queue that are missing from jobs map."""
    mock_model_manager.vram_mode = VRAMLoadMode.IMAGE_GENERATION
    mock_model_manager.current_mode = VRAMLoadMode.IMAGE_GENERATION
    
    # "stale-job" is in pending queue but NOT in self._jobs
    job_manager._pending_jobs = ["stale-job", "valid-job"]
    
    # valid-job setup
    mock_job = MagicMock()
    mock_job._job_type = JobType.VIDEO_GENERATION  # Different mode
    job_manager._jobs["valid-job"] = mock_job
    
    # Logic:
    # 1. Look for IMAGE_GENERATION job.
    #    - "stale-job": self._jobs.get("stale-job") -> None. Skip.
    #    - "valid-job": mode is VIDEO_GENERATION. Skip.
    # 2. Fallback to first in list: "stale-job".
    
    # The worker loop handles the stale job by checking get_job() and returning if None.
    # So _select_next_job RETURNING a stale job is actually expected behavior 
    # (garbage in, garbage out), as long as it doesn't crash.
    
    assert job_manager._select_next_job() == "stale-job"


@pytest.mark.asyncio
async def test_oom_error_detection(job_manager):
    """Test that OOM errors are correctly identified."""
    import torch
    
    # Case 1: PyTorch OOM Exception
    oom_exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert job_manager._check_oom_error(oom_exc, str(oom_exc)) is True
    
    # Case 2: String matching
    assert job_manager._check_oom_error(None, "RuntimeError: CUDA error: out of memory") is True
    assert job_manager._check_oom_error(None, "Some other error") is False


@pytest.mark.asyncio
async def test_worker_handles_switching_failure(job_manager, mock_model_manager):
    """Test that worker handles mode switching failures gracefully."""
    # Simulate switch failure
    mock_model_manager.ensure_mode_for_job.side_effect = Exception("Model load failed")
    
    mock_task = AsyncMock()
    await job_manager.submit_job("fail-job", JobType.IMAGE_GENERATION, mock_task)
    
    # Process
    await job_manager._process_job("fail-job")
    
    # Verify job failed
    job = job_manager.get_job("fail-job")
    assert job.status == JobStatus.FAILED
    assert "Model load failed" in job.error_message
    
    # Verify task was NOT executed
    mock_task.assert_not_called()
