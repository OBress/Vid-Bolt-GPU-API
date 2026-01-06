"""Tests for video generation endpoint."""

import pytest


def test_generate_video_success(client, api_key_headers, mock_storage, sample_job_id):
    """Test successful video generation using URLs."""
    input_url = "https://example.com/first-frame.png"
    response = client.post(
        "/api/v1/video/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Zoom in slowly",
            "duration_seconds": 4.0,
            "fps": 24,
            "aspect_ratio": "16:9",
            "save_url": "https://example.com/save.mp4",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "save_url" in data
    assert data["save_url"] == "https://example.com/save.mp4"
    assert "generation_time" in data

    # Verify storage calls
    mock_storage.download_from_url.assert_called_with(input_url)
    mock_storage.upload_to_url.assert_called()


def test_generate_video_with_end_frame(client, api_key_headers, mock_storage, sample_job_id):
    """Test video generation with an end frame URL."""
    input_url = "https://example.com/start.png"
    end_url = "https://example.com/end.png"
    
    response = client.post(
        "/api/v1/video/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "end_image_url": end_url,
            "prompt": "Morph start into end",
            "save_url": "https://example.com/save.mp4",
        },
    )

    assert response.status_code == 200
    # Verify both URLs were downloaded
    assert mock_storage.download_from_url.call_count == 2


def test_generate_video_with_output_url(client, api_key_headers, mock_storage, sample_job_id):
    """Test video generation with a custom output URL."""
    input_url = "https://example.com/start.png"
    output_url = "https://custom-storage.com/video.mp4?token=123"
    
    response = client.post(
        "/api/v1/video/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": input_url,
            "prompt": "Action sequence",
            "save_url": output_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["save_url"] == output_url

    # Verify storage calls
    mock_storage.upload_to_url.assert_called()
    assert mock_storage.upload_to_url.call_args.kwargs["url"] == output_url


def test_generate_video_validation_error(client, api_key_headers, sample_job_id):
    """Test validation error for invalid FPS."""
    response = client.post(
        "/api/v1/video/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "input_image_url": "https://example.com/start.png",
            "prompt": "Action",
            "fps": 60,  # Invalid FPS
            "save_url": "https://example.com/save.mp4",
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_generate_video_unauthorized(client):
    """Test unauthorized access."""
    response = client.post("/api/v1/video/generate", json={"job_id": "test", "save_url": "https://example.com/save.mp4"})
    assert response.status_code == 401
