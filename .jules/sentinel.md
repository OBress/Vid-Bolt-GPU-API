
## 2025-02-12 - SSRF Bypass via Redirects
**Vulnerability:** The application was using `httpx.get(..., follow_redirects=True)` which automatically follows redirects. This allowed an attacker to bypass the initial URL validation by providing a safe URL that redirects to a blocked internal IP (e.g., cloud metadata services).
**Learning:** Initial validation of a URL is insufficient if the HTTP client automatically follows redirects. Each redirect location must be validated against the security policy.
**Prevention:** Always disable automatic redirects (`follow_redirects=False`) when fetching user-provided URLs. Implement a manual redirect loop that validates the `Location` header before following it.
