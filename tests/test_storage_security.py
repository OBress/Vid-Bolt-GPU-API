
import pytest
from unittest.mock import AsyncMock, patch
from app.services.storage import StorageService
from app.config import Settings
from app.exceptions import ValidationError

@pytest.mark.asyncio
async def test_ssrf_manual_redirect_handling():
    """
    Test that the implementation manually handles redirects and validates
    intermediate URLs.
    """
    settings = Settings()
    service = StorageService(settings)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # 1. Test normal download (no redirect)
        # Should call get with follow_redirects=False
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.content = b"data"
        mock_client.get.return_value = mock_response

        url = "http://example.com/image.png"
        await service.download_from_url(url)

        args, kwargs = mock_client.get.call_args
        assert kwargs.get("follow_redirects") is False, "Must disable automatic redirects"

@pytest.mark.asyncio
async def test_ssrf_blocks_redirect_to_internal():
    """
    Test that a redirect to an internal IP is blocked.
    """
    settings = Settings()
    service = StorageService(settings)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Setup sequence of responses:
        # 1. 302 Redirect to private IP

        response1 = AsyncMock()
        response1.status_code = 302
        response1.is_redirect = True
        response1.headers = {"Location": "http://192.168.1.1/secret"}

        mock_client.get.side_effect = [response1]

        url = "http://example.com/redirect"

        # Should raise ValidationError due to private IP
        with pytest.raises(ValidationError) as exc_info:
            await service.download_from_url(url)

        assert "private" in str(exc_info.value).lower()

        # Verify it called get once
        assert mock_client.get.call_count == 1

@pytest.mark.asyncio
async def test_ssrf_follows_safe_redirect():
    """
    Test that safe redirects are followed.
    """
    settings = Settings()
    service = StorageService(settings)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # 1. 302 Redirect to safe URL
        # 2. 200 OK

        response1 = AsyncMock()
        response1.status_code = 302
        response1.is_redirect = True
        response1.headers = {"Location": "http://example.com/final"}

        response2 = AsyncMock()
        response2.status_code = 200
        response2.is_redirect = False
        response2.content = b"final data"

        mock_client.get.side_effect = [response1, response2]

        url = "http://example.com/start"

        content = await service.download_from_url(url)

        assert content == b"final data"
        assert mock_client.get.call_count == 2

        # Check second call url
        args, _ = mock_client.get.call_args_list[1]
        assert args[0] == "http://example.com/final"

@pytest.mark.asyncio
async def test_ssrf_max_redirects_exceeded():
    """Test that too many redirects raise ValidationError."""
    settings = Settings()
    service = StorageService(settings)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Simulate infinite loop or just many redirects
        redirect_response = AsyncMock()
        redirect_response.status_code = 302
        redirect_response.is_redirect = True
        redirect_response.headers = {"Location": "http://example.com/loop"}

        # We need enough side effects for max_redirects + 1
        # The code allows 5 redirects, so loop runs 6 times before failing else branch
        mock_client.get.side_effect = [redirect_response] * 10

        url = "http://example.com/start"

        with pytest.raises(ValidationError) as exc_info:
            await service.download_from_url(url)

        assert "too many redirects" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_ssrf_redirect_missing_location():
    """Test that redirect without Location header raises ValidationError."""
    settings = Settings()
    service = StorageService(settings)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        response1 = AsyncMock()
        response1.status_code = 302
        response1.is_redirect = True
        response1.headers = {} # Missing Location

        mock_client.get.side_effect = [response1]

        url = "http://example.com/start"

        with pytest.raises(ValidationError) as exc_info:
            await service.download_from_url(url)

        assert "without location header" in str(exc_info.value).lower()
