## 2025-02-18 - SSRF Bypass via Redirects
**Vulnerability:** The application was vulnerable to Server-Side Request Forgery (SSRF) bypass through HTTP redirects. While the initial URL was validated against internal/blocked IPs, `httpx` was configured to automatically follow redirects (`follow_redirects=True`). An attacker could supply a safe URL (e.g., `http://example.com/redirect`) that redirects to a sensitive internal endpoint (e.g., `http://169.254.169.254/latest/meta-data/`), bypassing the initial check.
**Learning:** Validating only the initial URL is insufficient for SSRF protection when using HTTP clients that follow redirects automatically. Redirects often bridge the gap between the public internet and private networks.
**Prevention:**
1. Disable automatic redirects in HTTP clients (`follow_redirects=False`).
2. Manually handle redirects in a loop.
3. Validate *every* URL (initial and all redirect targets) against the deny-list/allow-list before making the request.
