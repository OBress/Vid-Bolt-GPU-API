# Sentinel's Journal

## 2026-01-25 - SSRF Bypass via Redirects
**Vulnerability:** `validate_external_url` checked the initial URL, but `httpx` was configured to follow redirects (`follow_redirects=True`), allowing attackers to bypass validation by redirecting to internal IPs.
**Learning:** Initial URL validation is insufficient when the HTTP client automatically follows redirects.
**Prevention:** Explicitly disable redirects (`follow_redirects=False`) when making requests to user-supplied URLs, or implement a custom transport/hook to validate every redirect target.
