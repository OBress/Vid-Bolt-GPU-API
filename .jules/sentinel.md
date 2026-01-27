## 2026-01-27 - SSRF via Open Redirects
**Vulnerability:** `StorageService.download_from_url` used `httpx.get(..., follow_redirects=True)` which bypassed the initial `validate_external_url` check if the server redirected to a forbidden IP (e.g., localhost).
**Learning:** `httpx` (and `requests`) handles redirects transparently, so "time-of-check to time-of-use" (TOCTOU) vulnerabilities are common if validation isn't applied to every redirect hop.
**Prevention:** Manually handle redirects by setting `follow_redirects=False` and validating the `Location` header in a loop.
