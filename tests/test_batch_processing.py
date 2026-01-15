
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.models.job import JobStatus, JobResult
from app.models.internal import ImageGenerationResult
from app.services.job_manager import JobManager
from app.services.model_manager import JobType, VRAMLoadMode

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.mock_mode = True
    return settings

@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload_to_url.return_value = "https://r2.example.com/uploaded.png"
    return storage

@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.current_mode = VRAMLoadMode.ALL
    mm.ensure_mode_for_job = AsyncMock()
    
    # Mock generator
    generator = AsyncMock()
    # Return batch results
    async def generate_batch(params_list):
        results = []
        for params in params_list:
            res = ImageGenerationResult(
                image_data=b"fake_image_data",
                seed=123,
                width=params.width,
                height=params.height,
                generation_time=0.5
            )
            results.append(res)
        return results
        
    generator.generate_batch = AsyncMock(side_effect=generate_batch)
    mm.get_image_generator.return_value = generator
    
    return mm

@pytest.mark.asyncio
async def test_process_batch_uploads_and_job_result(mock_settings, mock_storage, mock_model_manager):
    """Test that _process_batch uploads results and constructs JobResult correctly."""
    
    job_manager = JobManager(mock_settings, mock_model_manager)
    job_manager._webhook_service = AsyncMock()
    
    # Create 2 fake jobs
    job_id_1 = "job_1"
    job_id_2 = "job_2"
    save_url_1 = "https://upload.example.com/1.png"
    save_url_2 = "https://upload.example.com/2.png"
    
    # Manually submit jobs without starting worker
    await job_manager.submit_job(
        job_id=job_id_1,
        job_type=JobType.IMAGE_GENERATION,
        task_func=AsyncMock(), # Unused in batch
        webhook_url="http://webhook.test",
        params=MagicMock(width=512, height=512),
        storage=mock_storage,
        save_url=save_url_1
    )
    
    await job_manager.submit_job(
        job_id=job_id_2,
        job_type=JobType.IMAGE_GENERATION,
        task_func=AsyncMock(), # Unused in batch
        webhook_url="http://webhook.test",
        params=MagicMock(width=512, height=512),
        storage=mock_storage,
        save_url=save_url_2
    )
    
    # Manually trigger batch processing
    bucket_key = (512, 512, JobType.IMAGE_GENERATION)
    job_ids = [job_id_1, job_id_2]
    
    await job_manager._process_batch(job_ids, bucket_key)
    
    # Verify uploads
    assert mock_storage.upload_to_url.call_count == 2
    
    # Verify JobResult for first job
    job1 = job_manager.get_job(job_id_1)
    assert job1.status == JobStatus.COMPLETED
    assert isinstance(job1.result, JobResult)
    assert job1.result.save_url == "https://r2.example.com/uploaded.png"
    
    # Verify JobResult for second job
    job2 = job_manager.get_job(job_id_2)
    assert job2.status == JobStatus.COMPLETED
    assert isinstance(job2.result, JobResult)
    assert job2.result.save_url == "https://r2.example.com/uploaded.png"
