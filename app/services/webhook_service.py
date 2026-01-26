"""Webhook delivery service.

Handles async webhook delivery with retry logic for generation callbacks.
"""

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

import httpx

from app.config import Settings
from app.exceptions import ValidationError
from app.models.webhook import WebhookPayload
from app.utils.url_validator import validate_external_url

logger = logging.getLogger(__name__)


class WebhookService:
    """Async webhook delivery with single retry.
    
    Features:
    - Async background delivery (non-blocking)
    - Single retry on failure (30s delay)
    - HMAC-SHA256 signing with optional secret
    - Callback after successful delivery (for cleanup)
    """
    
    MAX_RETRIES = 1              # Only 1 retry (2 total attempts)
    RETRY_DELAY_SECONDS = 30     # Wait 30s before retry
    TIMEOUT_SECONDS = 10         # HTTP request timeout
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._http_client: Optional[httpx.AsyncClient] = None
        self._pending: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the webhook delivery worker."""
        self._running = True
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.TIMEOUT_SECONDS),
            # Security: Disable redirects to prevent SSRF bypass
            follow_redirects=False,
        )
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("WebhookService started (1 retry, 30s delay)")
    
    async def stop(self) -> None:
        """Stop the worker and close HTTP client."""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._http_client:
            await self._http_client.aclose()
        logger.info("WebhookService stopped")
    
    async def deliver(
        self,
        webhook_url: str,
        payload: WebhookPayload,
        secret: Optional[str] = None,
        on_success: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        """Schedule a webhook for delivery.
        
        Args:
            webhook_url: URL to POST payload to
            payload: WebhookPayload to send
            secret: Optional HMAC-SHA256 signing secret
            on_success: Optional async callback after successful delivery (e.g., cleanup)
        """
        delivery_item = {
            "url": webhook_url,
            "payload": payload,
            "secret": secret,
            "attempts": 0,
            "on_success": on_success,
        }
        await self._pending.put(delivery_item)
        logger.debug(f"Webhook scheduled for {payload.job_id} -> {webhook_url}")
    
    async def _worker_loop(self) -> None:
        """Process pending webhook deliveries."""
        while self._running:
            try:
                # Wait for delivery item
                item = await asyncio.wait_for(self._pending.get(), timeout=1.0)
                
                success = await self._try_deliver(item)
                
                if success:
                    # Call success callback (e.g., delete job data)
                    if item.get("on_success"):
                        try:
                            await item["on_success"]()
                        except Exception as e:
                            logger.error(f"Webhook success callback failed: {e}")
                    logger.info(f"Webhook delivered: {item['payload'].job_id}")
                    
                elif item["attempts"] < self.MAX_RETRIES:
                    # Schedule retry
                    item["attempts"] += 1
                    logger.warning(
                        f"Webhook failed, retry {item['attempts']}/{self.MAX_RETRIES} "
                        f"in {self.RETRY_DELAY_SECONDS}s: {item['payload'].job_id}"
                    )
                    # Wait and re-queue
                    await asyncio.sleep(self.RETRY_DELAY_SECONDS)
                    await self._pending.put(item)
                else:
                    # Permanent failure after retries exhausted
                    logger.error(
                        f"Webhook permanently failed after {item['attempts'] + 1} attempts: "
                        f"{item['payload'].job_id} -> {item['url']}"
                    )
                    # Still call success callback to clean up (job is done either way)
                    if item.get("on_success"):
                        try:
                            await item["on_success"]()
                        except Exception as e:
                            logger.error(f"Webhook cleanup callback failed: {e}")
                            
            except asyncio.TimeoutError:
                # No items in queue, just continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Webhook worker error: {e}")
    
    async def _try_deliver(self, item: Dict) -> bool:
        """Attempt to deliver a webhook.
        
        Returns:
            True if delivery succeeded, False otherwise
        """
        payload: WebhookPayload = item["payload"]
        webhook_url: str = item["url"]
        secret: Optional[str] = item.get("secret")
        
        # Validate URL to prevent SSRF attacks
        try:
            validate_external_url(webhook_url)
        except ValidationError as e:
            logger.error(f"Webhook URL validation failed: {e}")
            return False
        
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": payload.event,
            "X-Job-Id": payload.job_id,
        }
        
        # Add HMAC signature if secret provided
        if secret:
            signature = self._sign_payload(payload, secret)
            headers["X-Webhook-Signature"] = signature
        
        try:
            response = await self._http_client.post(
                webhook_url,
                json=payload.model_dump(),
                headers=headers,
            )
            response.raise_for_status()
            return True
            
        except httpx.TimeoutException:
            logger.warning(f"Webhook timeout after {self.TIMEOUT_SECONDS}s: {webhook_url}")
            return False
        except httpx.HTTPStatusError as e:
            logger.warning(f"Webhook HTTP error {e.response.status_code}: {webhook_url}")
            return False
        except Exception as e:
            logger.warning(f"Webhook delivery error: {type(e).__name__}: {e}")
            return False
    
    def _sign_payload(self, payload: WebhookPayload, secret: str) -> str:
        """Create HMAC-SHA256 signature for payload.
        
        Args:
            payload: The webhook payload to sign
            secret: The HMAC secret key
            
        Returns:
            Signature string in format "sha256={hex_digest}"
        """
        body = payload.model_dump_json()
        sig = hmac.new(
            secret.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256
        )
        return f"sha256={sig.hexdigest()}"


# Global instance (set during startup)
_webhook_service_instance: Optional[WebhookService] = None


def set_webhook_service_instance(service: WebhookService) -> None:
    """Set the global WebhookService instance."""
    global _webhook_service_instance
    _webhook_service_instance = service


def get_webhook_service() -> WebhookService:
    """Get the global WebhookService instance."""
    if _webhook_service_instance is None:
        raise RuntimeError("WebhookService not initialized")
    return _webhook_service_instance
