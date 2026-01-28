## 2026-01-28 - SSRF via HTTP Redirects
**Vulnerability:** `httpx` with `follow_redirects=True` automatically follows redirects to blocked IP addresses (like localhost or 169.254.169.254), bypassing initial URL validation.
**Learning:** Initial validation of a URL is insufficient because the destination can change during the request chain (TOCTOU). Libraries often prioritize convenience over security by default.
**Prevention:** Always disable `follow_redirects` when fetching user-provided URLs. Implement a manual redirect loop that validates the `Location` header of every redirect against the SSRF policy before following it.
