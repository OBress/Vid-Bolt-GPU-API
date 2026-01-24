"""Tests for security related functionality."""

from fastapi.testclient import TestClient


def test_api_key_auth_success(client: TestClient) -> None:
    """Test that valid API key allows access."""
    # Using the key defined in conftest.py
    headers = {"X-API-Key": "test-api-key-12345"}
    response = client.get("/api/v1/status", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_key_auth_failure_invalid(client: TestClient) -> None:
    """Test that invalid API key denies access."""
    headers = {"X-API-Key": "wrong-key"}
    response = client.get("/api/v1/status", headers=headers)
    assert response.status_code == 401
    assert response.json()["error_code"] == "INVALID_API_KEY"

def test_api_key_auth_failure_missing(client: TestClient) -> None:
    """Test that missing API key denies access."""
    response = client.get("/api/v1/status")
    assert response.status_code == 401
    assert response.json()["error_code"] == "MISSING_API_KEY"
