import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import httpx
from app.services.storage import StorageService
from app.services.webhook_service import WebhookService
from app.config import Settings

@pytest.mark.asyncio
async def test_storage_service_follow_redirects_vulnerability():
    """Verify that StorageService.download_from_url disables redirects (Safe)."""
    settings = Settings()
    service = StorageService(settings)

    # Mock validate_external_url to pass
    with patch("app.services.storage.validate_external_url"):
        # Mock httpx.AsyncClient in the storage module
        with patch("app.services.storage.httpx.AsyncClient") as mock_client_cls:
            # Setup the context manager
            mock_instance = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_instance
            mock_client_cls.return_value.__aexit__.return_value = None

            # Setup response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"image data"
            mock_response.raise_for_status.return_value = None
            mock_instance.get.return_value = mock_response

            # Call the method
            await service.download_from_url("http://example.com/image.png")

            # Check arguments passed to client.get
            mock_instance.get.assert_called_once()
            args, kwargs = mock_instance.get.call_args

            # Assert that follow_redirects is False (confirming fix)
            assert kwargs.get("follow_redirects") is False, "Vulnerability check: follow_redirects should be False"

@pytest.mark.asyncio
async def test_webhook_service_follow_redirects_vulnerability():
    """Verify that WebhookService disables redirects (Safe)."""
    settings = Settings()
    service = WebhookService(settings)

    # Mock httpx.AsyncClient in the webhook_service module
    with patch("app.services.webhook_service.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value = mock_instance

        await service.start()

        # Check arguments passed to AsyncClient constructor
        # In code: self._http_client = httpx.AsyncClient(..., follow_redirects=True)

        mock_client_cls.assert_called_once()
        args, kwargs = mock_client_cls.call_args

        # Assert that follow_redirects is False (confirming fix)
        assert kwargs.get("follow_redirects") is False, "Vulnerability check: follow_redirects should be False"

        await service.stop()
