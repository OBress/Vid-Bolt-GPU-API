## 2025-02-12 - SSRF Bypass via Redirects
**Vulnerability:** The `StorageService` used `httpx.get(url, follow_redirects=True)` after validating only the initial URL. This allowed attackers to provide a safe URL (e.g., to their own server) that redirects to an internal IP (SSRF), bypassing the validation.
**Learning:** `httpx` and `requests` follow redirects automatically by default or when configured, but they don't re-run validation logic on the redirected URLs.
**Prevention:** Always disable automatic redirects (`follow_redirects=False`) when fetching user-provided URLs. Implement a manual redirect loop that validates the `Location` header against the allowlist/blocklist before following it.
