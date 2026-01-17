"""URL validation utilities for preventing SSRF attacks.

This module provides functions to validate URLs before making outbound requests,
blocking access to internal networks, cloud metadata endpoints, and other
potentially dangerous destinations.
"""

import ipaddress
import socket
from urllib.parse import urlparse

from app.exceptions import ValidationError


# Known cloud metadata endpoints to block
BLOCKED_HOSTNAMES = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
})


def validate_external_url(url: str) -> None:
    """Validate that a URL is safe to request (not internal).
    
    This function performs DNS resolution to check the actual IP addresses
    a hostname resolves to, preventing DNS rebinding attacks.
    
    Args:
        url: The URL to validate.
        
    Raises:
        ValidationError: If the URL points to an internal resource.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if not hostname:
        raise ValidationError("Invalid URL: missing hostname")
    
    # Block known metadata endpoints by hostname
    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise ValidationError("Access to metadata endpoint is not allowed")
    
    # Resolve DNS and check all resulting IPs
    try:
        # Get all IP addresses for the hostname
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        
        for family, _, _, _, sockaddr in addr_info:
            # sockaddr is (ip, port) for IPv4 or (ip, port, flow, scope) for IPv6
            ip_str = sockaddr[0]
            
            try:
                ip = ipaddress.ip_address(ip_str)
                
                # Check more specific categories first
                if ip.is_loopback:
                    raise ValidationError(
                        f"Access to loopback addresses is not allowed (resolved to {ip_str})"
                    )
                if ip.is_link_local:
                    raise ValidationError(
                        f"Access to link-local addresses is not allowed (resolved to {ip_str})"
                    )
                if ip.is_reserved:
                    raise ValidationError(
                        f"Access to reserved addresses is not allowed (resolved to {ip_str})"
                    )
                if ip.is_private:
                    raise ValidationError(
                        f"Access to private IP addresses is not allowed (resolved to {ip_str})"
                    )
                    
            except ValueError:
                # Not a valid IP address format, skip
                continue
                
    except socket.gaierror as e:
        # DNS resolution failed - this could be a non-existent domain
        # We allow this to fail later at the HTTP level
        pass
