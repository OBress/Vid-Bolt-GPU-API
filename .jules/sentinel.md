## 2025-02-18 - SSRF Bypass via HTTP Redirects
**Vulnerability:** The application used `httpx.AsyncClient(follow_redirects=True)` in `StorageService`, which allowed an attacker to bypass SSRF protection by providing a safe URL that redirects to a restricted internal IP (e.g., AWS metadata service).
**Learning:** Initial URL validation is insufficient when using HTTP clients that automatically follow redirects. Attackers can leverage open redirects or their own servers to redirect the request to an internal resource after the check has passed.
**Prevention:** Disable automatic redirects (`follow_redirects=False`) and implement a manual redirect loop that validates the `Location` header of every redirect response against the allowed allowlist/blocklist before following it.
