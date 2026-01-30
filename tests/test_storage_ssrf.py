import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.storage import StorageService
from app.exceptions import ValidationError
from app.config import Settings

@pytest.mark.asyncio
async def test_ssrf_redirect_validation():
    """
    Test that redirects are validated against SSRF rules.
    The current implementation (vulnerable) will follow redirects without validation.
    The fixed implementation should raise ValidationError when redirected to a private IP.
    """
    settings = Settings()
    service = StorageService(settings)

    # We want to mock httpx.AsyncClient so we control the response
    with patch("app.services.storage.httpx.AsyncClient") as MockClient:
        # Setup the mock client instance
        mock_client_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_client_instance

        # Create a mock response that is a redirect to a private IP
        # Note: In the real world, httpx with follow_redirects=True would automatically
        # make a second request. Since we mock .get(), we just return the 302 immediately.
        # The vulnerable code accepts the 302 content.
        # The fixed code will see the 302, try to validate the Location, and raise ValidationError.

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.is_redirect = True
        redirect_response.headers = {"Location": "http://192.168.1.1/secret"}
        redirect_response.content = b"Redirecting..."
        # raise_for_status should do nothing for 302
        redirect_response.raise_for_status.return_value = None

        # When client.get is called, return the redirect response
        mock_client_instance.get.return_value = redirect_response

        # We expect ValidationError because the redirect points to a private IP.
        # Current vulnerable code will NOT raise this (it will return the 302 response or fail elsewhere).
        with pytest.raises(ValidationError, match="private"):
            await service.download_from_url("http://example.com/safe")
