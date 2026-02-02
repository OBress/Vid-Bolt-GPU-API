## 2024-05-22 - SSRF Bypass via HTTP Redirects
**Vulnerability:** `httpx` client was configured with `follow_redirects=True`, allowing attackers to bypass initial URL validation by redirecting to an internal IP (e.g., cloud metadata services).
**Learning:** Initial validation of a URL is insufficient if the HTTP client automatically follows redirects without re-validating the new destination.
**Prevention:** When fetching user-provided URLs, always disable automatic redirects (`follow_redirects=False`) and manually handle the redirect loop, validating the `Location` header at each step.
