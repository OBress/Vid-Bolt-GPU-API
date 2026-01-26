## 2026-01-26 - [SSRF Bypass via Redirects]
**Vulnerability:** Found `follow_redirects=True` explicitly set in `StorageService` and `WebhookService`, allowing SSRF bypass even with initial URL validation.
**Learning:** Initial URL validation is insufficient if the HTTP client follows redirects to blocked internal IP addresses.
**Prevention:** Strictly enforce `follow_redirects=False` in all `httpx` client instantiations and requests.
