"""Tests for LTX-2 Video Generation API Endpoints."""

import io
import os
import pytest
import asyncio

# Set test environment to true for conditional logic if any
os.environ["API_KEY"] = "test-api-key-12345"


@pytest.mark.asyncio
class TestLTX2GenerateEndpoint:
    """Test suite for POST /api/v1/ltx2/generate endpoint."""
    async def test_generate_valid_request(self, async_client, api_key_headers, mock_storage, mock_job_manager, mock_model_manager):
        """Test successful I2V video generation."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO
        
        response = await async_client.post(
            "/api/v1/ltx2/generate",
            headers=api_key_headers,
            json={
                "job_id": "test-i2v-001",
                "start_frame_url": "https://example.com/start.png",
                "prompt": "Gentle waves on the ocean, cinematic",
                "duration_seconds": 3.0,
                "frame_rate": 24.0,
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "test-i2v-001"
        
        # Poll for stats
        max_retries = 50 
        status_data = None
        for _ in range(max_retries):
            status_response = await async_client.get(data["status_url"], headers=api_key_headers)
            status_data = status_response.json()
            if status_data["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.2)
            
        assert status_data["status"] == "completed"
        assert "generation_time" in status_data["result"]
        assert "save_url" in status_data["result"]
        assert "duration_seconds" in status_data["result"]
        assert "has_audio" in status_data["result"]

    async def test_generate_with_end_image(self, async_client, api_key_headers, mock_storage, mock_model_manager):
        """Test I2V with start and end frame interpolation."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO
        
        response = await async_client.post(
            "/api/v1/ltx2/generate",
            headers=api_key_headers,
            json={
                "job_id": "test-i2v-002",
                "start_frame_url": "https://example.com/start.png",
                "end_frame_url": "https://example.com/end.png",
                "prompt": "Person walking from left to right",
                "duration_seconds": 5.0,
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"

    async def test_generate_missing_required_fields(self, async_client, api_key_headers, mock_model_manager):
        """Test validation error for missing fields."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        response = await async_client.post(
            "/api/v1/ltx2/generate",
            headers=api_key_headers,
            json={
                "job_id": "test-missing",
                # Missing start_frame_url
                "prompt": "Fail",
                "save_url": "https://example.com/upload.mp4",
            },
        )
        assert response.status_code == 400

    async def test_generate_missing_auth(self, async_client):
        """Test unauthorized access."""
        response = await async_client.post(
            "/api/v1/ltx2/generate",
            json={"prompt": "test"},
        )
        assert response.status_code == 401

    async def test_generate_duration_bounds(self, async_client, api_key_headers, mock_model_manager, mock_storage):
        """Test validation for duration bounds."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        # Mock storage download to prevent 404
        # Mock storage download to prevent 404 and pass magic byte validation
        # Use a minimal valid PNG header
        mock_storage.download_from_url.return_value = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        
        # Test minimum duration
        response = await async_client.post(
            "/api/v1/ltx2/generate",
            headers=api_key_headers,
            json={
                "job_id": "test-min-duration",
                "start_frame_url": "https://example.com/image.png",
                "prompt": "Short video",
                "duration_seconds": 0.5,  # Minimum allowed
                "save_url": "https://example.com/upload.mp4",
                "webhook_url": "http://webhook.test"
            },
        )
        assert response.status_code == 202

        response = await async_client.post(
            "/api/v1/ltx2/generate",
            headers=api_key_headers,
            json={
                "job_id": "test-duration-min",
                "start_frame_url": "https://example.com/start.png",
                "duration_seconds": 0.5,
                "save_url": "https://example.com/upload.mp4",
                "webhook_url": "http://webhook.test"
            },
        )
        assert response.status_code == 400 # Invalid duration should be 4000


@pytest.mark.asyncio
class TestKeyframeInterpolateEndpoint:
    """Test suite for POST /api/v1/ltx2/interpolate endpoint."""

    async def test_interpolate_valid_request(self, async_client, api_key_headers, mock_storage, mock_model_manager):
        """Test successful keyframe interpolation."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            headers=api_key_headers,
            json={
                "job_id": "test-keyframe-001",
                "prompt": "Smooth transition between poses",
                "keyframes": [
                    {"image_url": "https://example.com/kf1.png", "frame_index": 0, "strength": 1.0},
                    {"image_url": "https://example.com/kf2.png", "frame_index": 72, "strength": 1.0},
                ],
                "duration_seconds": 3.0,
                "frame_rate": 24.0,
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        
        # Poll for completion
        max_retries = 50
        for _ in range(max_retries):
            status_response = await async_client.get(data["status_url"], headers=api_key_headers)
            if status_response.json()["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.2)
            
        final_status = (await async_client.get(data["status_url"], headers=api_key_headers)).json()
        assert final_status["status"] == "completed"

    async def test_interpolate_single_keyframe(self, async_client, api_key_headers, mock_storage, mock_model_manager):
        """Test with single keyframe (treated as I2V)."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            headers=api_key_headers,
            json={
                "job_id": "test-single-kf",
                "prompt": "Animation from single frame",
                "keyframes": [
                    {"image_url": "https://example.com/start.png", "frame_index": 0, "strength": 1.0},
                ],
                "duration_seconds": 2.0,
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )

        assert response.status_code == 202

    async def test_interpolate_empty_keyframes(self, async_client, api_key_headers, mock_model_manager):
        """Test validation error for empty keyframes list."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            headers=api_key_headers,
            json={
                "job_id": "test-empty-kf",
                "prompt": "Fail",
                "keyframes": [],
                "duration_seconds": 2.0,
                "save_url": "https://example.com/upload/video.mp4",
            },
        )
        assert response.status_code == 400

    async def test_interpolate_too_many_keyframes(self, async_client, api_key_headers, mock_model_manager):
        """Test validation error for too many keyframes."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        keyframes = [
            {"image_url": f"https://example.com/{i}.png", "frame_index": i, "strength": 1.0}
            for i in range(20) # Use a large number to be safe
        ]
        
        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            headers=api_key_headers,
            json={
                "job_id": "test-many-kf",
                "prompt": "Many frames",
                "keyframes": keyframes,
                "duration_seconds": 5.0,
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )
        assert response.status_code == 400

    async def test_interpolate_missing_auth(self, async_client):
        """Test missing auth."""
        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            json={"prompt": "test"},
        )
        assert response.status_code == 401

    async def test_interpolate_with_negative_prompt(self, async_client, api_key_headers, mock_storage, mock_model_manager):
        """Test with negative prompt included."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            headers=api_key_headers,
            json={
                "job_id": "test-negative-prompt",
                "prompt": "Beautiful landscape animation",
                "negative_prompt": "blurry, distorted, low quality",
                "keyframes": [
                    {"image_url": "https://example.com/start.png", "frame_index": 0, "strength": 1.0},  
                    {"image_url": "https://example.com/end.png", "frame_index": 120, "strength": 1.0},  
                ],
                "duration_seconds": 5.0,
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )

        assert response.status_code == 202

    async def test_interpolate_custom_aspect_ratio(self, async_client, api_key_headers, mock_storage, mock_model_manager):

        """Test with different aspect ratios."""
        from app.services.model_manager import ModelMode
        mock_model_manager._mode = ModelMode.VIDEO

        response = await async_client.post(
            "/api/v1/ltx2/interpolate",
            headers=api_key_headers,
            json={
                "job_id": "test-aspect-ratio",
                "prompt": "Vertical video",
                "keyframes": [
                    {"image_url": "https://example.com/kf.png", "frame_index": 0, "strength": 1.0},
                ],
                "duration_seconds": 2.0,
                "aspect_ratio": "9:16",
                "save_url": "https://example.com/upload/video.mp4",
                "webhook_url": "http://webhook.test"
            },
        )

        assert response.status_code == 202
