"""Tests for SSRF protection configurations."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.storage import StorageService
from app.services.webhook_service import WebhookService
from app.config import get_settings

@pytest.mark.asyncio
async def test_storage_service_ssrf_config():
    """Test that StorageService is configured to prevent redirects."""
    settings = get_settings()
    service = StorageService(settings)

    # We want to intercept the httpx.AsyncClient constructor call
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.content = b"data"
        mock_client.aclose = AsyncMock() # Fix for TypeError
        mock_client_cls.return_value = mock_client

        try:
            await service.download_from_url("http://example.com/image.png")
        except Exception:
            pass

        # Check how AsyncClient was initialized or used
        # In StorageService, follow_redirects is passed to client.get()
        # response = await client.get(url, follow_redirects=True)
        call_kwargs = mock_client.get.call_args[1]

        # Verify that redirects are disabled
        assert call_kwargs.get("follow_redirects") is False, "Security regression: StorageService must disable redirects"


@pytest.mark.asyncio
async def test_webhook_service_ssrf_config():
    """Test that WebhookService is configured to prevent redirects."""
    settings = get_settings()
    service = WebhookService(settings)

    # WebhookService initializes client in start()
    # self._http_client = httpx.AsyncClient(..., follow_redirects=False)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock() # Fix for TypeError
        mock_client_cls.return_value = mock_client

        await service.start()

        # Verify constructor args
        # call_args[1] is kwargs
        init_kwargs = mock_client_cls.call_args[1]

        # Verify that redirects are disabled
        assert init_kwargs.get("follow_redirects") is False, "Security regression: WebhookService must disable redirects"

        await service.stop()
