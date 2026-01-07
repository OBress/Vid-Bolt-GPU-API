"""Tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_health_check(client: TestClient) -> None:
    """Test basic health check without authentication."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "mock_mode" in data


def test_health_check_no_auth_required(client: TestClient) -> None:
    """Test that health check does not require authentication."""
    # Should work without any headers
    response = client.get("/health")
    assert response.status_code == 200


def test_status_requires_auth(client: TestClient) -> None:
    """Test that /api/v1/status requires authentication."""
    response = client.get("/api/v1/status")

    assert response.status_code == 401
    data = response.json()
    assert data["status"] == "failed"
    assert data["error_code"] == "MISSING_API_KEY"


def test_status_invalid_auth(
    client: TestClient,
    invalid_api_key_headers: dict[str, str],
) -> None:
    """Test that /api/v1/status rejects invalid API key."""
    response = client.get("/api/v1/status", headers=invalid_api_key_headers)

    assert response.status_code == 401
    data = response.json()
    assert data["status"] == "failed"
    assert data["error_code"] == "INVALID_API_KEY"


def test_status_with_valid_auth(
    client: TestClient,
    api_key_headers: dict[str, str],
) -> None:
    """Test that /api/v1/status works with valid authentication."""
    response = client.get("/api/v1/status", headers=api_key_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "mock_mode" in data

