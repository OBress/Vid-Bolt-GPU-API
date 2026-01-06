"""Tests for image generation endpoint."""

from unittest.mock import MagicMock

import pytest


def test_generate_image_success(client, api_key_headers, mock_storage, sample_job_id):
    """Test successful image generation."""
    response = client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "prompt": "A beautiful sunset",
            "aspect_ratio": "1:1",
            "save_url": "https://example.com/save.png",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "save_url" in data
    assert data["save_url"] == "https://example.com/save.png"
    assert "generation_time" in data

    # Verify storage call
    mock_storage.upload_to_url.assert_called_once()


def test_generate_image_with_output_url(client, api_key_headers, mock_storage, sample_job_id):
    """Test image generation with a custom output URL."""
    custom_url = "https://custom-storage.com/upload/here?token=123"
    response = client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": sample_job_id,
            "prompt": "A beautiful sunset",
            "save_url": custom_url,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["save_url"] == custom_url

    # Verify storage call
    mock_storage.upload_to_url.assert_called_once()
    args, kwargs = mock_storage.upload_to_url.call_args
    assert kwargs["url"] == custom_url


def test_generate_image_validation_error(client, api_key_headers):
    """Test validation error for invalid aspect ratio."""
    response = client.post(
        "/api/v1/image/generate",
        headers=api_key_headers,
        json={
            "job_id": "test-job",
            "prompt": "sunset",
            "aspect_ratio": "invalid-ratio",
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "failed"
    assert data["error_code"] == "VALIDATION_ERROR"


def test_generate_image_unauthorized(client):
    """Test unauthorized access (missing API key)."""
    response = client.post(
        "/api/v1/image/generate",
        json={
            "job_id": "test-job",
            "prompt": "sunset",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert data["error_code"] == "MISSING_API_KEY"


def test_generate_image_invalid_key(client, invalid_api_key_headers):
    """Test access with invalid API key."""
    response = client.post(
        "/api/v1/image/generate",
        headers=invalid_api_key_headers,
        json={
            "job_id": "test-job",
            "prompt": "sunset",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert data["error_code"] == "INVALID_API_KEY"
