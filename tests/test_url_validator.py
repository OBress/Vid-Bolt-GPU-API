"""Tests for URL validation utility (SSRF protection)."""

import pytest
from unittest.mock import patch

from app.exceptions import ValidationError
from app.utils.url_validator import validate_external_url


class TestValidateExternalUrl:
    """Test cases for the validate_external_url function."""

    def test_blocks_localhost_ip(self):
        """Should block 127.0.0.1 (localhost)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://127.0.0.1/test")
        assert "loopback" in str(exc_info.value).lower()

    def test_blocks_localhost_hostname(self):
        """Should block localhost hostname after DNS resolution."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://localhost/test")
        assert "loopback" in str(exc_info.value).lower()

    def test_blocks_private_ip_192_168(self):
        """Should block 192.168.x.x private IPs."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://192.168.1.1/test")
        assert "private" in str(exc_info.value).lower()

    def test_blocks_private_ip_10(self):
        """Should block 10.x.x.x private IPs."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://10.0.0.1/test")
        assert "private" in str(exc_info.value).lower()

    def test_blocks_private_ip_172_16(self):
        """Should block 172.16.x.x private IPs."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://172.16.0.1/test")
        assert "private" in str(exc_info.value).lower()

    def test_blocks_aws_metadata_endpoint(self):
        """Should block AWS metadata endpoint."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://169.254.169.254/latest/meta-data/")
        assert "metadata" in str(exc_info.value).lower()

    def test_blocks_gcp_metadata_endpoint(self):
        """Should block GCP metadata endpoint."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://metadata.google.internal/computeMetadata/v1/")
        assert "metadata" in str(exc_info.value).lower()

    def test_blocks_ipv6_loopback(self):
        """Should block IPv6 loopback address."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("http://[::1]/test")
        assert "loopback" in str(exc_info.value).lower()

    def test_blocks_dns_rebinding_attack(self):
        """Should block URLs where hostname resolves to private IP."""
        # Mock socket.getaddrinfo to simulate DNS rebinding
        with patch("app.utils.url_validator.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (2, 1, 6, "", ("192.168.1.100", 80))  # Simulated private IP resolution
            ]
            with pytest.raises(ValidationError) as exc_info:
                validate_external_url("http://evil-rebind.example.com/")
            assert "private" in str(exc_info.value).lower()

    def test_allows_public_url(self):
        """Should allow public URLs like example.com."""
        # This should not raise
        validate_external_url("https://example.com/webhook")

    def test_allows_public_ip(self):
        """Should allow public IP addresses."""
        # 8.8.8.8 is Google's public DNS
        validate_external_url("http://8.8.8.8/test")

    def test_rejects_missing_hostname(self):
        """Should reject URLs without a hostname."""
        with pytest.raises(ValidationError) as exc_info:
            validate_external_url("file:///etc/passwd")
        assert "missing hostname" in str(exc_info.value).lower()
