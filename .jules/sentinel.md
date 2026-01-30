## 2026-01-30 - SSRF via HTTP Redirects
**Vulnerability:** The application was vulnerable to SSRF via unvalidated redirects because `httpx`'s `follow_redirects=True` was used without validating the target of the redirect. The initial URL was validated, but a malicious server could redirect to a private IP (e.g., 169.254.169.254) which `httpx` would follow automatically.
**Learning:** `httpx` (and `requests`) does not validate the destination of redirects against SSRF rules. Validating the initial URL is insufficient.
**Prevention:** Always use `follow_redirects=False` (or `allow_redirects=False`) in security-sensitive contexts. Implement a manual redirect loop that validates the `Location` header of every redirect against the allowed list/blocklist before following it.
