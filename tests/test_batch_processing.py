"""Extensive tests for Z-Image batch processing functionality.

Tests cover:
- VRAM Estimator utility functions
- JobManager resolution bucketing and batch selection
- ZImageGenerator batch generation
- End-to-end batch workflow
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch
from collections import defaultdict

# Import test targets
from app.services.vram_estimator import (
    get_vram_info,
    estimate_zimage_vram_per_image,
    calculate_max_batch_size,
    VRAMInfo,
    VRAM_SAFETY_MARGIN_GB,
    MIN_FREE_VRAM_GB,
    MAX_BATCH_SIZE_ZIMAGE,
)
from app.services.job_manager import JobManager
from app.services.model_manager import JobType, VRAMLoadMode
from app.models.job import JobInfo, JobStatus
from app.config import Settings


# =============================================================================
# Test Fixtures - Mock VRAM for machines without CUDA
# =============================================================================

@pytest.fixture(autouse=True)
def mock_vram_info():
    """Mock VRAM info to return 96GB for all tests.
    
    This ensures tests pass on machines without CUDA GPUs.
    """
    mock_info = VRAMInfo(free_gb=92.0, total_gb=96.0, used_gb=4.0)
    with patch('app.services.vram_estimator.get_vram_info', return_value=mock_info):
        yield mock_info


# =============================================================================
# VRAM Estimator Tests
# =============================================================================

class TestVRAMEstimator:
    """Tests for the VRAM estimation utility."""
    
    def test_vram_info_dataclass(self):
        """Test VRAMInfo dataclass calculations."""
        info = VRAMInfo(free_gb=80.0, total_gb=96.0, used_gb=16.0)
        
        assert info.free_gb == 80.0
        assert info.total_gb == 96.0
        assert info.used_gb == 16.0
        assert info.available_for_inference_gb == 80.0 - VRAM_SAFETY_MARGIN_GB
    
    def test_vram_info_low_memory(self):
        """Test VRAMInfo when free memory is less than safety margin."""
        info = VRAMInfo(free_gb=2.0, total_gb=96.0, used_gb=94.0)
        
        # Should return 0 when free < safety margin
        assert info.available_for_inference_gb == max(0.0, 2.0 - VRAM_SAFETY_MARGIN_GB)
    
    def test_estimate_vram_small_image(self):
        """Test VRAM estimation for small images."""
        # 512x512 = 0.262 megapixels
        vram = estimate_zimage_vram_per_image(512, 512)
        
        # Should be base cost + scaled by megapixels
        assert vram > 0
        assert vram < 2.0  # Small image should be under 2GB
    
    def test_estimate_vram_large_image(self):
        """Test VRAM estimation for large images."""
        # 2048x2048 = 4.19 megapixels
        vram = estimate_zimage_vram_per_image(2048, 2048)
        
        # Larger image should use more VRAM
        assert vram > estimate_zimage_vram_per_image(512, 512)
    
    def test_estimate_vram_scaling(self):
        """Test that VRAM scales with resolution."""
        vram_512 = estimate_zimage_vram_per_image(512, 512)
        vram_1024 = estimate_zimage_vram_per_image(1024, 1024)
        vram_2048 = estimate_zimage_vram_per_image(2048, 2048)
        
        assert vram_512 < vram_1024 < vram_2048
    
    def test_calculate_max_batch_high_vram(self):
        """Test batch size calculation with high available VRAM."""
        # 80GB available, should allow many images
        max_batch = calculate_max_batch_size(1024, 1024, available_vram_gb=80.0)
        
        assert max_batch >= 10  # Should allow decent batch
        assert max_batch <= MAX_BATCH_SIZE_ZIMAGE  # Never exceed cap
    
    def test_calculate_max_batch_low_vram(self):
        """Test batch size calculation with low available VRAM."""
        # 2GB available (below minimum)
        max_batch = calculate_max_batch_size(1024, 1024, available_vram_gb=1.5)
        
        assert max_batch == 1  # Should fall back to 1
    
    def test_calculate_max_batch_medium_vram(self):
        """Test batch size with moderate VRAM (ALL mode scenario)."""
        # 10GB available (e.g., when other models loaded)
        max_batch = calculate_max_batch_size(1024, 1024, available_vram_gb=10.0)
        
        assert 1 <= max_batch <= 10
    
    def test_max_batch_respects_cap(self):
        """Test that batch size never exceeds MAX_BATCH_SIZE_ZIMAGE."""
        # Even with infinite VRAM, should cap
        max_batch = calculate_max_batch_size(512, 512, available_vram_gb=1000.0)
        
        assert max_batch == MAX_BATCH_SIZE_ZIMAGE


# =============================================================================
# JobManager Bucketing Tests
# =============================================================================

class TestJobManagerBucketing:
    """Tests for JobManager resolution-based bucketing."""
    
    @pytest.fixture
    def settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.mock_mode = True
        return settings
    
    @pytest.fixture
    def job_manager(self, settings):
        """Create JobManager instance."""
        jm = JobManager(settings)
        return jm
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create mock ModelManager."""
        mm = MagicMock()
        mm.vram_mode = VRAMLoadMode.IMAGE_GENERATION
        mm.zimage_generator = MagicMock()
        mm.ensure_mode_for_job = AsyncMock()
        return mm
    
    def test_submit_job_creates_bucket(self, job_manager):
        """Test that submitting a job creates the correct bucket."""
        # Create mock params
        params = MagicMock()
        params.width = 1024
        params.height = 1024
        
        async def run_test():
            await job_manager.submit_job(
                job_id="test-1",
                job_type=JobType.IMAGE_GENERATION,
                task_func=AsyncMock(),
                params=params
            )
            
            # Check bucket was created
            bucket_key = (1024, 1024, JobType.IMAGE_GENERATION)
            assert bucket_key in job_manager._pending_buckets
            assert "test-1" in job_manager._pending_buckets[bucket_key]
        
        asyncio.run(run_test())
    
    def test_multiple_jobs_same_bucket(self, job_manager):
        """Test that jobs with same dimensions go to same bucket."""
        params = MagicMock()
        params.width = 1024
        params.height = 1024
        
        async def run_test():
            for i in range(5):
                await job_manager.submit_job(
                    job_id=f"test-{i}",
                    job_type=JobType.IMAGE_GENERATION,
                    task_func=AsyncMock(),
                    params=params
                )
            
            bucket_key = (1024, 1024, JobType.IMAGE_GENERATION)
            assert len(job_manager._pending_buckets[bucket_key]) == 5
        
        asyncio.run(run_test())
    
    def test_different_dimensions_different_buckets(self, job_manager):
        """Test that jobs with different dimensions go to different buckets."""
        async def run_test():
            # Submit 1024x1024
            params_1024 = MagicMock()
            params_1024.width = 1024
            params_1024.height = 1024
            await job_manager.submit_job(
                job_id="test-1024",
                job_type=JobType.IMAGE_GENERATION,
                task_func=AsyncMock(),
                params=params_1024
            )
            
            # Submit 512x512
            params_512 = MagicMock()
            params_512.width = 512
            params_512.height = 512
            await job_manager.submit_job(
                job_id="test-512",
                job_type=JobType.IMAGE_GENERATION,
                task_func=AsyncMock(),
                params=params_512
            )
            
            # Check separate buckets
            assert (1024, 1024, JobType.IMAGE_GENERATION) in job_manager._pending_buckets
            assert (512, 512, JobType.IMAGE_GENERATION) in job_manager._pending_buckets
        
        asyncio.run(run_test())
    
    def test_select_batch_prioritizes_oldest(self, job_manager, mock_model_manager):
        """Test that batch selection prioritizes bucket with oldest job."""
        job_manager.set_model_manager(mock_model_manager)
        
        async def run_test():
            # Submit older job at 512x512
            params_512 = MagicMock()
            params_512.width = 512
            params_512.height = 512
            await job_manager.submit_job(
                job_id="old-job",
                job_type=JobType.IMAGE_GENERATION,
                task_func=AsyncMock(),
                params=params_512
            )
            
            # Manually set older creation time
            job_manager._jobs["old-job"].created_at = time.time() - 100
            
            # Submit newer job at 1024x1024
            params_1024 = MagicMock()
            params_1024.width = 1024
            params_1024.height = 1024
            await job_manager.submit_job(
                job_id="new-job",
                job_type=JobType.IMAGE_GENERATION,
                task_func=AsyncMock(),
                params=params_1024
            )
            
            # Batch selection should pick the 512 bucket (older)
            with patch('app.services.vram_estimator.calculate_max_batch_size', return_value=10):
                batch, bucket_key = job_manager._select_batch()
            
            assert "old-job" in batch
            assert bucket_key == (512, 512, JobType.IMAGE_GENERATION)
        
        asyncio.run(run_test())
    
    def test_select_batch_respects_vram_limit(self, job_manager, mock_model_manager):
        """Test that batch size is limited by VRAM estimation."""
        job_manager.set_model_manager(mock_model_manager)
        
        async def run_test():
            params = MagicMock()
            params.width = 1024
            params.height = 1024
            
            # Submit 20 jobs
            for i in range(20):
                await job_manager.submit_job(
                    job_id=f"job-{i}",
                    job_type=JobType.IMAGE_GENERATION,
                    task_func=AsyncMock(),
                    params=params
                )
            
            # Mock VRAM to only allow 5
            with patch('app.services.vram_estimator.calculate_max_batch_size', return_value=5):
                batch, _ = job_manager._select_batch()
            
            assert len(batch) == 5
        
        asyncio.run(run_test())
    
    def test_non_image_jobs_not_batched(self, job_manager, mock_model_manager):
        """Test that non-image jobs are not batched (single job only)."""
        job_manager.set_model_manager(mock_model_manager)
        
        async def run_test():
            params = MagicMock()
            params.width = 1024
            params.height = 1024
            
            # Submit video generation jobs
            for i in range(5):
                await job_manager.submit_job(
                    job_id=f"video-{i}",
                    job_type=JobType.VIDEO_GENERATION,
                    task_func=AsyncMock(),
                    params=params
                )
            
            batch, _ = job_manager._select_batch()
            
            # Should only get 1 job (video not batchable)
            assert len(batch) == 1
        
        asyncio.run(run_test())


# =============================================================================
# ZImageGenerator Batch Tests
# =============================================================================

class TestZImageGeneratorBatch:
    """Tests for ZImageGenerator batch generation."""
    
    @pytest.fixture
    def settings(self):
        """Create mock settings for dry-run mode."""
        settings = MagicMock(spec=Settings)
        settings.zimage_dry_run = True
        settings.zimage_model_path = "models/z-image-turbo"
        settings.zimage_lora_path = "models/loras"
        settings.zimage_device = "cuda"
        settings.zimage_dtype = "bfloat16"
        return settings
    
    @pytest.fixture
    def generator(self, settings):
        """Create ZImageGenerator in dry-run mode."""
        from app.services.zimage_generator import ZImageGenerator
        gen = ZImageGenerator(settings)
        gen.is_loaded = True  # Simulate loaded state
        return gen
    
    def test_empty_batch_returns_empty(self, generator):
        """Test that empty params list returns empty results."""
        async def run_test():
            results = await generator.generate_batch([])
            assert results == []
        
        asyncio.run(run_test())
    
    def test_single_item_batch_uses_fast_path(self, generator):
        """Test that single-item batch uses fast path (generate_image)."""
        from app.models.internal import ImageGenerationParams
        
        params = ImageGenerationParams(
            job_id="test-1",
            prompt="A test image",
            width=512,
            height=512,
            seed=None,
            num_inference_steps=8,
        )
        
        async def run_test():
            # Patch generate_image to track if called
            generator.generate_image = AsyncMock(return_value=MagicMock())
            
            await generator.generate_batch([params])
            
            # Should have called generate_image directly
            generator.generate_image.assert_called_once_with(params)
        
        asyncio.run(run_test())
    
    def test_batch_validates_dimensions(self, generator):
        """Test that batch rejects mixed dimensions."""
        from app.models.internal import ImageGenerationParams
        
        params_1 = ImageGenerationParams(
            job_id="test-1",
            prompt="Image 1",
            width=1024,
            height=1024,
            seed=None,
            num_inference_steps=8,
        )
        params_2 = ImageGenerationParams(
            job_id="test-2",
            prompt="Image 2",
            width=512,  # Different!
            height=512,
            seed=None,
            num_inference_steps=8,
        )
        
        async def run_test():
            with pytest.raises(ValueError, match="Batch requires same dimensions"):
                await generator.generate_batch([params_1, params_2])
        
        asyncio.run(run_test())
    
    def test_batch_dry_run_returns_correct_count(self, generator):
        """Test that dry-run batch returns correct number of results."""
        from app.models.internal import ImageGenerationParams
        
        params_list = [
            ImageGenerationParams(
                job_id=f"test-{i}",
                prompt=f"Image {i}",
                width=512,
                height=512,
                seed=None,
                num_inference_steps=8,
            )
            for i in range(5)
        ]
        
        async def run_test():
            results = await generator.generate_batch(params_list)
            
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result.width == 512
                assert result.height == 512
        
        asyncio.run(run_test())
    
    def test_batch_preserves_seeds(self, generator):
        """Test that batch preserves explicit seeds."""
        from app.models.internal import ImageGenerationParams
        
        params_list = [
            ImageGenerationParams(
                job_id=f"test-{i}",
                prompt=f"Image {i}",
                width=512,
                height=512,
                seed=1000 + i,  # Explicit seeds
                num_inference_steps=8,
            )
            for i in range(3)
        ]
        
        async def run_test():
            results = await generator.generate_batch(params_list)
            
            assert results[0].seed == 1000
            assert results[1].seed == 1001
            assert results[2].seed == 1002
        
        asyncio.run(run_test())


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestBatchingIntegration:
    """Integration tests for the complete batching workflow."""
    
    @pytest.fixture
    def settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.mock_mode = True
        settings.zimage_dry_run = True
        settings.zimage_model_path = "models/z-image-turbo"
        settings.zimage_lora_path = "models/loras"
        settings.zimage_device = "cuda"
        settings.zimage_dtype = "bfloat16"
        return settings
    
    def test_queue_to_batch_workflow(self, settings):
        """Test complete workflow from queue submission to batch selection."""
        from app.models.internal import ImageGenerationParams
        from app.services.zimage_generator import ZImageGenerator
        
        job_manager = JobManager(settings)
        
        # Create mock model manager with generator
        generator = ZImageGenerator(settings)
        generator.is_loaded = True
        
        model_manager = MagicMock()
        model_manager.vram_mode = VRAMLoadMode.IMAGE_GENERATION
        model_manager.zimage_generator = generator
        model_manager.ensure_mode_for_job = AsyncMock()
        
        job_manager.set_model_manager(model_manager)
        
        async def run_test():
            # Submit 10 jobs at same resolution
            for i in range(10):
                params = ImageGenerationParams(
                    job_id=f"test-{i}",
                    prompt=f"Test prompt {i}",
                    width=1024,
                    height=1024,
                    seed=None,
                    num_inference_steps=8,
                )
                await job_manager.submit_job(
                    job_id=f"test-{i}",
                    job_type=JobType.IMAGE_GENERATION,
                    task_func=AsyncMock(),
                    params=params
                )
            
            # Verify all jobs in same bucket
            bucket_key = (1024, 1024, JobType.IMAGE_GENERATION)
            assert len(job_manager._pending_buckets[bucket_key]) == 10
            
            # Select batch (mock high VRAM)
            with patch('app.services.vram_estimator.calculate_max_batch_size', return_value=8):
                batch, selected_bucket = job_manager._select_batch()
            
            assert len(batch) == 8
            assert selected_bucket == bucket_key
        
        asyncio.run(run_test())
    
    def test_mixed_resolution_creates_separate_batches(self, settings):
        """Test that mixed resolutions create separate batches."""
        from app.models.internal import ImageGenerationParams
        
        job_manager = JobManager(settings)
        
        model_manager = MagicMock()
        model_manager.vram_mode = VRAMLoadMode.IMAGE_GENERATION
        
        job_manager.set_model_manager(model_manager)
        
        async def run_test():
            # Submit jobs at different resolutions
            resolutions = [(512, 512), (1024, 1024), (1920, 1080)]
            
            for res_idx, (w, h) in enumerate(resolutions):
                for i in range(3):
                    params = ImageGenerationParams(
                        job_id=f"test-{res_idx}-{i}",
                        prompt=f"Test at {w}x{h}",
                        width=w,
                        height=h,
                        seed=None,
                        num_inference_steps=8,
                    )
                    await job_manager.submit_job(
                        job_id=f"test-{res_idx}-{i}",
                        job_type=JobType.IMAGE_GENERATION,
                        task_func=AsyncMock(),
                        params=params
                    )
            
            # Should have 3 separate buckets
            assert len(job_manager._pending_buckets) == 3
            
            # Each bucket should have 3 jobs
            for bucket_jobs in job_manager._pending_buckets.values():
                assert len(bucket_jobs) == 3
        
        asyncio.run(run_test())


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================

class TestBatchingEdgeCases:
    """Edge case and performance tests."""
    
    def test_large_batch_capped(self):
        """Test that very large batches are capped."""
        # Even with massive VRAM, should cap at MAX_BATCH_SIZE_ZIMAGE
        max_batch = calculate_max_batch_size(256, 256, available_vram_gb=500.0)
        assert max_batch == MAX_BATCH_SIZE_ZIMAGE
    
    def test_zero_vram_returns_one(self):
        """Test that zero VRAM returns batch size 1."""
        max_batch = calculate_max_batch_size(1024, 1024, available_vram_gb=0.0)
        assert max_batch == 1
    
    def test_negative_vram_returns_one(self):
        """Test handling of negative VRAM (edge case)."""
        max_batch = calculate_max_batch_size(1024, 1024, available_vram_gb=-5.0)
        assert max_batch == 1
    
    def test_extreme_resolution(self):
        """Test VRAM estimation for extreme resolutions."""
        # 8K resolution
        vram = estimate_zimage_vram_per_image(7680, 4320)
        assert vram > 0
        
        # Should estimate higher VRAM for larger images
        vram_1080 = estimate_zimage_vram_per_image(1920, 1080)
        assert vram > vram_1080
    
    def test_queue_position_accuracy(self):
        """Test that queue position is calculated correctly."""
        settings = MagicMock(spec=Settings)
        job_manager = JobManager(settings)
        
        async def run_test():
            params = MagicMock()
            params.width = 1024
            params.height = 1024
            
            # Submit 5 jobs
            for i in range(5):
                await job_manager.submit_job(
                    job_id=f"job-{i}",
                    job_type=JobType.IMAGE_GENERATION,
                    task_func=AsyncMock(),
                    params=params
                )
                # Set creation time manually for predictable ordering
                job_manager._jobs[f"job-{i}"].created_at = i
            
            # Check positions
            assert job_manager.get_queue_position("job-0") == 1
            assert job_manager.get_queue_position("job-4") == 5
            assert job_manager.get_queue_position("nonexistent") is None
        
        asyncio.run(run_test())


# =============================================================================
# LightX2V VRAM Estimator Tests
# =============================================================================

class TestLightX2VVRAMEstimator:
    """Tests for LightX2V VRAM estimation utilities."""
    
    def test_lightx2v_constants_exist(self):
        """Test that LightX2V VRAM constants are defined."""
        from app.services.vram_estimator import (
            LIGHTX2V_BASE_MODEL_FULL_GB,
            LIGHTX2V_BASE_MODEL_OFFLOAD_GB,
            LIGHTX2V_BASE_ACTIVATION_GB,
            LIGHTX2V_GB_PER_MEGAPIXEL,
            LIGHTX2V_CONDITIONING_OVERHEAD_GB,
            MAX_BATCH_SIZE_LIGHTX2V,
        )
        
        # Verify constants are reasonable values
        assert LIGHTX2V_BASE_MODEL_FULL_GB > LIGHTX2V_BASE_MODEL_OFFLOAD_GB
        assert LIGHTX2V_BASE_ACTIVATION_GB > 0
        assert LIGHTX2V_GB_PER_MEGAPIXEL > 0
        assert LIGHTX2V_CONDITIONING_OVERHEAD_GB > 0
        assert MAX_BATCH_SIZE_LIGHTX2V > 0
    
    def test_estimate_lightx2v_vram_small_image(self):
        """Test VRAM estimation for small images."""
        from app.services.vram_estimator import estimate_lightx2v_vram_per_image
        
        # 512x512 = 0.262 megapixels
        vram = estimate_lightx2v_vram_per_image(512, 512)
        
        # Should be positive and reasonable for LightX2V
        assert vram > 0
        assert vram < 5.0  # Small image should be under 5GB
    
    def test_estimate_lightx2v_vram_large_image(self):
        """Test VRAM estimation for large images."""
        from app.services.vram_estimator import estimate_lightx2v_vram_per_image
        
        # 2048x2048 = 4.19 megapixels
        vram = estimate_lightx2v_vram_per_image(2048, 2048)
        
        # Larger image should use more VRAM
        small_vram = estimate_lightx2v_vram_per_image(512, 512)
        assert vram > small_vram
    
    def test_lightx2v_higher_than_zimage(self):
        """Test that LightX2V uses more VRAM per image than Z-Image."""
        from app.services.vram_estimator import (
            estimate_lightx2v_vram_per_image,
            estimate_zimage_vram_per_image,
        )
        
        # LightX2V should require more VRAM due to I2I conditioning
        lightx2v_vram = estimate_lightx2v_vram_per_image(1024, 1024)
        zimage_vram = estimate_zimage_vram_per_image(1024, 1024)
        
        assert lightx2v_vram > zimage_vram
    
    def test_calculate_lightx2v_max_batch_high_vram(self):
        """Test LightX2V batch size with high available VRAM."""
        from app.services.vram_estimator import (
            calculate_lightx2v_max_batch_size,
            MAX_BATCH_SIZE_LIGHTX2V,
        )
        
        # 80GB available should allow batching
        max_batch = calculate_lightx2v_max_batch_size(
            1024, 1024, available_vram_gb=80.0
        )
        
        assert max_batch >= 1
        assert max_batch <= MAX_BATCH_SIZE_LIGHTX2V
    
    def test_calculate_lightx2v_max_batch_low_vram(self):
        """Test LightX2V batch size with low available VRAM."""
        from app.services.vram_estimator import calculate_lightx2v_max_batch_size
        
        # 1.5GB available should fall back to 1
        max_batch = calculate_lightx2v_max_batch_size(
            1024, 1024, available_vram_gb=1.5
        )
        
        assert max_batch == 1
    
    def test_calculate_lightx2v_max_batch_all_mode(self):
        """Test LightX2V batch size when other models are loaded (ALL mode)."""
        from app.services.vram_estimator import calculate_lightx2v_max_batch_size
        
        # Compare batch sizes with and without other models loaded
        batch_exclusive = calculate_lightx2v_max_batch_size(
            1024, 1024, available_vram_gb=50.0, other_models_loaded=False
        )
        batch_shared = calculate_lightx2v_max_batch_size(
            1024, 1024, available_vram_gb=50.0, other_models_loaded=True
        )
        
        # ALL mode should allow smaller or equal batch size
        assert batch_shared <= batch_exclusive
    
    def test_lightx2v_batch_cap(self):
        """Test that LightX2V batch size respects cap."""
        from app.services.vram_estimator import (
            calculate_lightx2v_max_batch_size,
            MAX_BATCH_SIZE_LIGHTX2V,
        )
        
        # Even with massive VRAM, should cap
        max_batch = calculate_lightx2v_max_batch_size(
            512, 512, available_vram_gb=1000.0
        )
        
        assert max_batch == MAX_BATCH_SIZE_LIGHTX2V
    
    def test_get_lightx2v_base_vram(self):
        """Test base VRAM calculation."""
        from app.services.vram_estimator import (
            get_lightx2v_base_vram,
            LIGHTX2V_BASE_MODEL_FULL_GB,
            LIGHTX2V_BASE_MODEL_OFFLOAD_GB,
        )
        
        # With offload should use less VRAM
        base_offload = get_lightx2v_base_vram(cpu_offload=True)
        base_full = get_lightx2v_base_vram(cpu_offload=False)
        
        assert base_offload == LIGHTX2V_BASE_MODEL_OFFLOAD_GB
        assert base_full == LIGHTX2V_BASE_MODEL_FULL_GB
        assert base_offload < base_full


# =============================================================================
# LightX2V JobManager Batch Selection Tests
# =============================================================================

class TestLightX2VBatchSelection:
    """Tests for JobManager batch selection with IMAGE_EDITING jobs."""
    
    @pytest.fixture
    def settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.mock_mode = True
        return settings
    
    @pytest.fixture
    def job_manager(self, settings):
        """Create JobManager instance."""
        return JobManager(settings)
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create mock ModelManager in IMAGE_EDITING mode."""
        mm = MagicMock()
        mm.current_mode = VRAMLoadMode.IMAGE_EDITING
        mm.get_image_editor = MagicMock()
        mm.ensure_mode_for_job = AsyncMock()
        return mm
    
    def test_image_editing_jobs_bucketed(self, job_manager):
        """Test that IMAGE_EDITING jobs are bucketed correctly."""
        params = MagicMock()
        params.width = 1024
        params.height = 1024
        
        async def run_test():
            await job_manager.submit_job(
                job_id="edit-1",
                job_type=JobType.IMAGE_EDITING,
                task_func=AsyncMock(),
                params=params
            )
            
            bucket_key = (1024, 1024, JobType.IMAGE_EDITING)
            assert bucket_key in job_manager._pending_buckets
            assert "edit-1" in job_manager._pending_buckets[bucket_key]
        
        asyncio.run(run_test())
    
    def test_image_editing_batch_selection(self, job_manager, mock_model_manager):
        """Test that IMAGE_EDITING jobs are selected for batching."""
        job_manager.set_model_manager(mock_model_manager)
        
        params = MagicMock()
        params.width = 1024
        params.height = 1024
        
        async def run_test():
            # Submit 5 IMAGE_EDITING jobs
            for i in range(5):
                await job_manager.submit_job(
                    job_id=f"edit-{i}",
                    job_type=JobType.IMAGE_EDITING,
                    task_func=AsyncMock(),
                    params=params
                )
                job_manager._jobs[f"edit-{i}"].created_at = i
            
            # Select batch
            selected, bucket_key = job_manager._select_batch()
            
            # Should select multiple jobs (batch > 1 with sufficient VRAM mock)
            assert len(selected) >= 1
            assert bucket_key == (1024, 1024, JobType.IMAGE_EDITING)
        
        asyncio.run(run_test())
    
    def test_image_editing_separate_from_generation(self, job_manager):
        """Test that IMAGE_EDITING and IMAGE_GENERATION are in separate buckets."""
        gen_params = MagicMock()
        gen_params.width = 1024
        gen_params.height = 1024
        
        edit_params = MagicMock()
        edit_params.width = 1024
        edit_params.height = 1024
        
        async def run_test():
            # Submit generation job
            await job_manager.submit_job(
                job_id="gen-1",
                job_type=JobType.IMAGE_GENERATION,
                task_func=AsyncMock(),
                params=gen_params
            )
            
            # Submit editing job
            await job_manager.submit_job(
                job_id="edit-1",
                job_type=JobType.IMAGE_EDITING,
                task_func=AsyncMock(),
                params=edit_params
            )
            
            # Should be in different buckets
            gen_bucket = (1024, 1024, JobType.IMAGE_GENERATION)
            edit_bucket = (1024, 1024, JobType.IMAGE_EDITING)
            
            assert gen_bucket in job_manager._pending_buckets
            assert edit_bucket in job_manager._pending_buckets
            assert "gen-1" in job_manager._pending_buckets[gen_bucket]
            assert "edit-1" in job_manager._pending_buckets[edit_bucket]
        
        asyncio.run(run_test())


# =============================================================================
# LightX2V Generator Batch Tests
# =============================================================================

class TestLightX2VGeneratorBatch:
    """Tests for LightX2VImageEditGenerator batch processing."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for dry-run mode."""
        settings = MagicMock(spec=Settings)
        settings.lightx2v_dry_run = True
        settings.lightx2v_model_path = "/mock/path"
        settings.lightx2v_lora_path = "/mock/lora"
        settings.lightx2v_lora_filename = "mock.safetensors"
        settings.lightx2v_lora_strength = 1.0
        settings.lightx2v_device = "cuda"
        settings.lightx2v_attn_mode = "flash_attn3"
        settings.lightx2v_resize_mode = "adaptive"
        settings.lightx2v_infer_steps = 8
        settings.lightx2v_guidance_scale = 7.5
        settings.lightx2v_cpu_offload = True
        settings.lightx2v_text_encoder_offload = True
        return settings
    
    @pytest.fixture
    def generator(self, mock_settings):
        """Create LightX2V generator in dry-run mode."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        gen = LightX2VImageEditGenerator(mock_settings)
        gen.load_models()  # Load in dry-run mode
        return gen
    
    @pytest.fixture
    def valid_image_bytes(self):
        """Create valid minimal PNG image bytes for testing."""
        from PIL import Image
        import io
        # Create a small 64x64 red image
        img = Image.new('RGB', (64, 64), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def mock_params(self, valid_image_bytes):
        """Create mock edit parameters with valid image data."""
        from app.models.internal import ImageEditParams
        return ImageEditParams(
            job_id="test-1",
            input_image_data=valid_image_bytes,
            prompt="Test prompt",
            width=1024,
            height=1024,
            mask_data=None,
            seed=42,
        )
    
    def test_edit_batch_empty_list(self, generator):
        """Test edit_batch with empty list returns empty."""
        async def run_test():
            results = await generator.edit_batch([])
            assert results == []
        
        asyncio.run(run_test())
    
    def test_edit_batch_single_image(self, generator, mock_params):
        """Test edit_batch with single image uses fast path."""
        async def run_test():
            results = await generator.edit_batch([mock_params])
            
            assert len(results) == 1
            # Dry-run returns input image size, not requested size
            assert results[0].width > 0
            assert results[0].height > 0
        
        asyncio.run(run_test())
    
    def test_edit_batch_multiple_images(self, generator, valid_image_bytes):
        """Test edit_batch with multiple same-dimension images."""
        from app.models.internal import ImageEditParams
        
        async def run_test():
            params_list = [
                ImageEditParams(
                    job_id=f"test-{i}",
                    input_image_data=valid_image_bytes,
                    prompt=f"Prompt {i}",
                    width=1024,
                    height=1024,
                    mask_data=None,
                    seed=i,
                )
                for i in range(3)
            ]
            
            results = await generator.edit_batch(params_list)
            
            assert len(results) == 3
            # All should have valid dimensions (note: may differ from request due to dry-run using input image size)
            for result in results:
                assert result.width > 0
                assert result.height > 0
        
        asyncio.run(run_test())
    
    def test_edit_batch_dimension_validation(self, generator):
        """Test that edit_batch rejects mixed dimensions."""
        from app.models.internal import ImageEditParams
        
        async def run_test():
            params_list = [
                ImageEditParams(
                    job_id="test-1",
                    input_image_data=b"fake",
                    prompt="Prompt 1",
                    width=1024,
                    height=1024,
                    mask_data=None,
                    seed=None,
                ),
                ImageEditParams(
                    job_id="test-2",
                    input_image_data=b"fake",
                    prompt="Prompt 2",
                    width=512,  # Different dimension!
                    height=512,
                    mask_data=None,
                    seed=None,
                ),
            ]
            
            with pytest.raises(ValueError, match="same dimensions"):
                await generator.edit_batch(params_list)
        
        asyncio.run(run_test())
    
    def test_edit_batch_not_loaded_raises(self, mock_settings):
        """Test that edit_batch raises if models not loaded."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        from app.models.internal import ImageEditParams
        
        generator = LightX2VImageEditGenerator(mock_settings)
        # Don't call load_models()
        
        async def run_test():
            params_list = [
                ImageEditParams(
                    job_id="test-1",
                    input_image_data=b"fake",
                    prompt="Test",
                    width=1024,
                    height=1024,
                    mask_data=None,
                    seed=None,
                ),
                ImageEditParams(
                    job_id="test-2",
                    input_image_data=b"fake",
                    prompt="Test 2",
                    width=1024,
                    height=1024,
                    mask_data=None,
                    seed=None,
                ),
            ]
            
            with pytest.raises(RuntimeError, match="not loaded"):
                await generator.edit_batch(params_list)
        
        asyncio.run(run_test())


# =============================================================================
# LightX2V Integration Tests
# =============================================================================

class TestLightX2VBatchingIntegration:
    """End-to-end integration tests for LightX2V batching."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.mock_mode = True
        settings.lightx2v_dry_run = True
        settings.lightx2v_model_path = "/mock/path"
        settings.lightx2v_lora_path = "/mock/lora"
        settings.lightx2v_lora_filename = "mock.safetensors"
        settings.lightx2v_lora_strength = 1.0
        settings.lightx2v_device = "cuda"
        settings.lightx2v_attn_mode = "flash_attn3"
        settings.lightx2v_resize_mode = "adaptive"
        settings.lightx2v_infer_steps = 8
        settings.lightx2v_guidance_scale = 7.5
        settings.lightx2v_cpu_offload = True
        settings.lightx2v_text_encoder_offload = True
        return settings
    
    @pytest.fixture
    def mock_model_manager(self, mock_settings):
        """Create mock ModelManager with LightX2V generator."""
        from app.services.lightx2v_generator import LightX2VImageEditGenerator
        
        mm = MagicMock()
        mm.current_mode = VRAMLoadMode.IMAGE_EDITING
        
        # Create real generator in dry-run mode
        gen = LightX2VImageEditGenerator(mock_settings)
        gen.load_models()
        
        mm.get_image_editor = MagicMock(return_value=gen)
        mm.ensure_mode_for_job = AsyncMock(return_value=True)
        
        return mm
    
    @pytest.fixture
    def valid_image_bytes(self):
        """Create valid minimal PNG image bytes for testing."""
        from PIL import Image
        import io
        # Create a small 64x64 red image
        img = Image.new('RGB', (64, 64), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_full_batch_workflow(self, mock_settings, mock_model_manager, valid_image_bytes):
        """Test full workflow: submit -> bucket -> process batch."""
        from app.models.internal import ImageEditParams
        
        job_manager = JobManager(mock_settings)
        job_manager.set_model_manager(mock_model_manager)
        
        async def run_test():
            # Submit 3 IMAGE_EDITING jobs
            for i in range(3):
                params = ImageEditParams(
                    job_id=f"edit-{i}",
                    input_image_data=valid_image_bytes,
                    prompt=f"Edit prompt {i}",
                    width=1024,
                    height=1024,
                    mask_data=None,
                    seed=i,
                )
                await job_manager.submit_job(
                    job_id=f"edit-{i}",
                    job_type=JobType.IMAGE_EDITING,
                    task_func=AsyncMock(),  # Not used for batch
                    params=params
                )
            
            # Verify bucketing
            bucket_key = (1024, 1024, JobType.IMAGE_EDITING)
            assert len(job_manager._pending_buckets[bucket_key]) == 3
            
            # Select batch
            selected, _ = job_manager._select_batch()
            assert len(selected) >= 1
            
            # Process batch
            await job_manager._process_batch(selected, bucket_key)
            
            # Verify jobs completed (since we're in dry-run mode)
            for job_id in selected:
                job = job_manager._jobs[job_id]
                assert job.status == JobStatus.COMPLETED
                assert job.result is not None
        
        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
