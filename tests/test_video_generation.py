"""Tests for video generation endpoint."""

import pytest

# Check if moviepy.editor is available for video generation tests
try:
    from moviepy.editor import ImageClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False


@pytest.mark.skipif(not HAS_MOVIEPY, reason="moviepy not installed")
def test_generate_video_success(client, api_key_headers, mock_storage, sample_job_id, mock_job_manager, mock_model_manager):
    """Test successful video generation using URLs."""
    input_url = "https://example.com/first-frame.png"
    
    # Mock Video Mode
    from app.services.model_manager import ModelMode
    with patch("app.services.model_manager.ModelManager.current_mode", new_callable=lambda: ModelMode.VIDEO):
        response = client.post(
            "/api/v1/video/generate",
            headers=api_key_headers,
            json={
                "job_id": sample_job_id,
                "start_frame_url": input_url,
                "prompt": "Zoom in slowly",
                "duration_seconds": 4.0,
                "fps": 24,
                "aspect_ratio": "16:9",
                "save_url": "https://example.com/save.mp4",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == sample_job_id

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "save_url" in data
    assert data["save_url"] == "https://example.com/save.mp4"
    assert "generation_time" in data

    # Verify storage calls
    mock_storage.download_from_url.assert_called_with(input_url)
    mock_storage.upload_to_url.assert_called()


@pytest.mark.skipif(not HAS_MOVIEPY, reason="moviepy not installed")
def test_generate_video_with_end_frame(client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test video generation with an end frame URL."""
    input_url = "https://example.com/start.png"
    end_url = "https://example.com/end.png"
    
    response = client.post(
        "/api/v1/video/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "start_frame_url": input_url,
            "end_frame_url": end_url,
            "prompt": "Morph start into end",
            "save_url": "https://example.com/save.mp4",
        },
    )

    assert response.status_code == 200
    # Verify both URLs were downloaded
    assert mock_storage.download_from_url.call_count == 2


@pytest.mark.skipif(not HAS_MOVIEPY, reason="moviepy not installed")
def test_generate_video_with_output_url(client, api_key_headers, mock_storage, sample_job_id, mock_model_manager):
    """Test video generation with a custom output URL."""
    input_url = "https://example.com/start.png"
    output_url = "https://custom-storage.com/video.mp4?token=123"
    
    response = client.post(
        "/api/v1/video/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "start_frame_url": input_url,
            "prompt": "Action sequence",
            "save_url": output_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["save_url"] == output_url

    # Verify storage calls
    mock_storage.upload_to_url.assert_called()


def test_generate_video_validation_error(client, api_key_headers, sample_job_id, mock_model_manager):
    """Test validation error for missing required fields on LTX2 endpoint."""
    from app.services.model_manager import ModelMode
    mock_model_manager._mode = ModelMode.VIDEO
    
    # The old /api/v1/video/generate endpoint was replaced by /api/v1/ltx2/generate
    # Test that the ltx2 endpoint validates required fields
    response = client.post(
        "/api/v1/ltx2/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            # Missing start_frame_url which is required
            "prompt": "Action",
            "save_url": "https://example.com/save.mp4",
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_generate_video_unauthorized(client):
    """Test unauthorized access to LTX2 endpoint."""
    response = client.post("/api/v1/ltx2/generate", json={"job_id": "test", "save_url": "https://example.com/save.mp4"})
    assert response.status_code == 401

