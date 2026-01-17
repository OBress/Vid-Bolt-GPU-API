"""Tests for Batch API endpoints.

Run with: pytest tests/test_batch_api.py -v
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.batch import BatchStatus, BatchItemState


class TestBatchModels:
    """Test batch model validation."""
    
    def test_batch_image_generate_item_validation(self):
        """Test BatchImageGenerateItem field validation."""
        from app.models.batch_image_generation import BatchImageGenerateItem
        
        # Valid item
        item = BatchImageGenerateItem(
            item_id="item-1",
            prompt="A beautiful sunset",
            save_url="https://example.com/output.png"
        )
        assert item.prompt == "A beautiful sunset"
        assert item.aspect_ratio.value == "16:9"  # default
        assert item.num_inference_steps == 20  # default
    
    def test_batch_image_generate_item_custom_dimensions(self):
        """Test custom width/height validation."""
        from app.models.batch_image_generation import BatchImageGenerateItem
        
        # Both width and height provided - valid
        item = BatchImageGenerateItem(
            item_id="item-2",
            prompt="Test",
            width=1024,
            height=768,
            save_url="https://example.com/output.png"
        )
        assert item.width == 1024
        assert item.height == 768
    
    def test_batch_image_generate_item_partial_dimensions_invalid(self):
        """Test that providing only width or height raises error."""
        from app.models.batch_image_generation import BatchImageGenerateItem
        
        # Only width - invalid
        with pytest.raises(ValueError, match="Height must be provided"):
            BatchImageGenerateItem(
                item_id="item-3",
                prompt="Test",
                width=1024,
                save_url="https://example.com/output.png"
            )
        
        # Only height - invalid
        with pytest.raises(ValueError, match="Width must be provided"):
            BatchImageGenerateItem(
                item_id="item-4",
                prompt="Test",
                height=768,
                save_url="https://example.com/output.png"
            )
    
    def test_batch_request_max_items_image(self):
        """Test max items validation for image batches."""
        from app.models.batch_image_generation import BatchImageGenerateRequest, BatchImageGenerateItem
        
        # Create request with max items (500) - should work
        items = [
            BatchImageGenerateItem(item_id=f"item-{i}", prompt=f"Test {i}", save_url=f"https://example.com/{i}.png")
            for i in range(500)
        ]
        request = BatchImageGenerateRequest(batch_id="test-batch", items=items, webhook_url="http://test.webhook")
        assert len(request.items) == 500
    
    def test_batch_request_max_items_video(self):
        """Test max items validation for video batches (100)."""
        from app.models.batch_video_generation import BatchVideoGenerateRequest, BatchVideoGenerateItem
        
        items = [
            BatchVideoGenerateItem(
                item_id=f"video-{i}",
                start_frame_url=f"https://example.com/input{i}.png",
                prompt=f"Test {i}",
                save_url=f"https://example.com/{i}.mp4"
            )
            for i in range(100)
        ]
        request = BatchVideoGenerateRequest(batch_id="test-batch", items=items, webhook_url="http://test.webhook")
        assert len(request.items) == 100
    
    def test_batch_info_status_aggregation(self):
        """Test BatchInfo model creation."""
        from app.models.batch import BatchInfo, BatchItemStatus, BatchItemState, BatchStatus
        
        items = [
            BatchItemStatus(item_index=0, item_id="item-0", job_id="batch__item_0", status=BatchItemState.COMPLETED),
            BatchItemStatus(item_index=1, item_id="item-1", job_id="batch__item_1", status=BatchItemState.PROCESSING),
            BatchItemStatus(item_index=2, item_id="item-2", job_id="batch__item_2", status=BatchItemState.FAILED, error_message="OOM"),
        ]
        
        batch = BatchInfo(
            batch_id="test-batch",
            status=BatchStatus.PROCESSING,
            batch_type="image_generation",
            total_items=3,
            completed_items=1,
            failed_items=1,
            pending_items=0,
            processing_items=1,
            retrying_items=0,
            created_at=time.time(),
            items=items
        )
        
        assert batch.total_items == 3
        assert batch.completed_items == 1
        assert batch.failed_items == 1


class TestBatchManager:
    """Test BatchManager service logic."""
    
    def test_batch_manager_initialization(self):
        """Test BatchManager can be initialized."""
        from app.services.batch_manager import BatchManager
        from app.services.job_manager import JobManager
        from app.config import get_settings
        
        settings = get_settings()
        job_manager = JobManager(settings)
        batch_manager = BatchManager(settings, job_manager)
        
        assert batch_manager.MAX_IMAGE_BATCH_SIZE == 500
        assert batch_manager.MAX_VIDEO_BATCH_SIZE == 100
        assert batch_manager.BATCH_RETENTION_SECONDS == 300
    
    def test_batch_id_format(self):
        """Test that job IDs are formatted correctly from batch ID."""
        batch_id = "my-batch-123"
        expected_job_id = f"{batch_id}__item_5"
        
        # Verify the format matches what we use in batch_manager
        assert "__item_" in expected_job_id
        assert expected_job_id.startswith(batch_id)
    
    def test_is_batch_job(self):
        """Test batch job detection."""
        from app.services.batch_manager import BatchManager
        from app.services.job_manager import JobManager
        from app.config import get_settings
        
        settings = get_settings()
        job_manager = JobManager(settings)
        batch_manager = BatchManager(settings, job_manager)
        
        # No jobs yet
        assert batch_manager.is_batch_job("random-job-id") == False
        
        # Manually add a job to batch tracking
        batch_manager._job_to_batch["batch-123__item_0"] = "batch-123"
        assert batch_manager.is_batch_job("batch-123__item_0") == True
    
    def test_retry_count_tracking(self):
        """Test retry count management."""
        from app.services.batch_manager import BatchManager
        from app.services.job_manager import JobManager
        from app.config import get_settings
        
        settings = get_settings()
        job_manager = JobManager(settings)
        batch_manager = BatchManager(settings, job_manager)
        
        job_id = "batch-test__item_0"
        
        # Initialize retry count
        batch_manager._retry_counts[job_id] = 0
        assert batch_manager.get_retry_count(job_id) == 0
        
        # Simulate first failure - should return True (will retry)
        batch_manager._retry_counts[job_id] = 1
        assert batch_manager.get_retry_count(job_id) == 1


class TestBatchEndpoints:
    """Integration tests for batch API endpoints."""
    
    def test_batch_image_generate_endpoint_accepts_request(
        self, client: TestClient, api_key_headers: dict, mock_storage
    ):
        """Test that batch image generate endpoint accepts valid request."""
        batch_id = f"test-batch-{uuid.uuid4()}"
        
        response = client.post(
            "/api/v1/batch/image/generate",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [
                    {"item_id": "img-1", "prompt": "A cat", "save_url": "https://example.com/1.png"},
                    {"item_id": "img-2", "prompt": "A dog", "save_url": "https://example.com/2.png"},
                ],
                "webhook_url": "http://webhook.test"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["batch_id"] == batch_id
        assert data["status"] == "pending"
        assert data["total_items"] == 2
        assert "status_url" in data
    
    def test_batch_image_generate_requires_auth(self, client: TestClient):
        """Test that batch endpoints require authentication."""
        response = client.post(
            "/api/v1/batch/image/generate",
            json={
                "batch_id": "test",
                "items": [{"item_id": "i0", "prompt": "Test", "save_url": "https://example.com/1.png"}]
            }
        )
        
        assert response.status_code == 401
    
    def test_batch_image_generate_duplicate_batch_id(
        self, client: TestClient, api_key_headers: dict, mock_storage
    ):
        """Test that duplicate batch IDs are rejected."""
        batch_id = f"test-batch-{uuid.uuid4()}"
        
        # First submission
        response1 = client.post(
            "/api/v1/batch/image/generate",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [{"item_id": "i1", "prompt": "Test", "save_url": "https://example.com/1.png"}],
                "webhook_url": "http://webhook.test"
            }
        )
        assert response1.status_code == 202
        
        # Duplicate submission
        response2 = client.post(
            "/api/v1/batch/image/generate",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [{"item_id": "i2", "prompt": "Test 2", "save_url": "https://example.com/2.png"}],
                "webhook_url": "http://webhook.test"
            }
        )
        assert response2.status_code == 409
    
    def test_batch_status_endpoint(
        self, client: TestClient, api_key_headers: dict, mock_storage
    ):
        """Test GET batch status endpoint."""
        batch_id = f"test-batch-{uuid.uuid4()}"
        
        # Submit batch
        client.post(
            "/api/v1/batch/image/generate",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [
                    {"item_id": "img-1", "prompt": "A cat", "save_url": "https://example.com/1.png"},
                    {"item_id": "img-2", "prompt": "A dog", "save_url": "https://example.com/2.png"},
                ],
                "webhook_url": "http://webhook.test"
            }
        )
        
        # Get status
        response = client.get(
            f"/api/v1/batch/{batch_id}",
            headers=api_key_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == batch_id
        assert data["batch_type"] == "image_generation"
        assert data["total_items"] == 2
        assert "items" in data
        assert len(data["items"]) == 2
    
    def test_batch_not_found(self, client: TestClient, api_key_headers: dict):
        """Test 404 for non-existent batch."""
        response = client.get(
            "/api/v1/batch/non-existent-batch",
            headers=api_key_headers
        )
        
        assert response.status_code == 404
    
    def test_batch_collect_deletes_batch(
        self, client: TestClient, api_key_headers: dict, mock_storage
    ):
        """Test DELETE endpoint collects and removes batch."""
        batch_id = f"test-batch-{uuid.uuid4()}"
        
        # Submit batch
        client.post(
            "/api/v1/batch/image/generate",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [{"item_id": "test-item", "prompt": "Test", "save_url": "https://example.com/1.png"}],
                "webhook_url": "http://webhook.test"
            }
        )
        
        # Collect (DELETE)
        response = client.delete(
            f"/api/v1/batch/{batch_id}",
            headers=api_key_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == batch_id
        
        # Batch should no longer exist
        response2 = client.get(
            f"/api/v1/batch/{batch_id}",
            headers=api_key_headers
        )
        assert response2.status_code == 404
    
    def test_batch_video_generate_endpoint(
        self, client: TestClient, api_key_headers: dict, mock_storage
    ):
        """Test batch video generate endpoint."""
        batch_id = f"test-batch-{uuid.uuid4()}"
        
        response = client.post(
            "/api/v1/batch/video/generate",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [
                    {
                        "item_id": "vid-1",
                        "start_frame_url": "https://example.com/input.png",
                        "prompt": "Waves crashing",
                        "duration_seconds": 3.0,
                        "save_url": "https://example.com/1.mp4"
                    },
                ],
                "webhook_url": "http://webhook.test"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["batch_id"] == batch_id
        assert data["total_items"] == 1


class TestBatchImageEditing:
    """Tests for batch image editing endpoint."""
    
    def test_batch_image_edit_endpoint(
        self, client: TestClient, api_key_headers: dict, mock_storage
    ):
        """Test batch image edit endpoint."""
        batch_id = f"test-batch-{uuid.uuid4()}"
        
        response = client.post(
            "/api/v1/batch/image/edit",
            headers=api_key_headers,
            json={
                "batch_id": batch_id,
                "items": [
                    {
                        "item_id": "ed-1",
                        "input_image_url": "https://example.com/input.png",
                        "prompt": "Make it look vintage",
                        "save_url": "https://example.com/1.png"
                    },
                ],
                "webhook_url": "http://webhook.test"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["batch_id"] == batch_id
        assert data["total_items"] == 1
