"""Cloudflare R2 storage service."""

import logging
import time
from typing import Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import Settings
from app.exceptions import R2ConnectionError, R2UploadError

logger = logging.getLogger(__name__)


class StorageService:
    """Service for interacting with Cloudflare R2 storage."""

    def __init__(self, settings: Settings):
        """Initialize storage service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._client = None

    @property
    def client(self):
        """Lazily initialize and return the S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.r2_endpoint_url,
                aws_access_key_id=self.settings.r2_access_key_id,
                aws_secret_access_key=self.settings.r2_secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
        return self._client

    async def download_from_url(self, url: str) -> bytes:
        """Download remote content to memory.

        Args:
            url: Remote URL to fetch

        Returns:
            Content bytes

        Raises:
            ValidationError: If download fails or content is too large
        """
        import httpx
        from app.exceptions import ValidationError

        logger.info(f"Downloading from URL: {url.split('?')[0]}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                content = response.content
                settings = get_settings()

                if len(content) > settings.max_image_size_bytes:
                    from app.exceptions import FileTooLargeError
                    raise FileTooLargeError(
                        f"Downloaded content exceeds maximum size of {settings.max_image_size_mb}MB"
                    )

                return content
        except httpx.HTTPError as e:
            logger.error(f"Failed to download asset: {e}")
            raise ValidationError(f"Failed to download image from URL: {e}")

    async def upload_to_url(

        self,
        data: bytes,
        url: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload data directly to a provided URL (likely presigned).

        Args:
            data: Data to upload
            url: Destination URL
            content_type: MIME type of the content

        Returns:
            The URL uploaded to

        Raises:
            R2UploadError: If upload fails
        """
        import httpx

        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.put(
                        url,
                        content=data,
                        headers={"Content-Type": content_type},
                    )
                    response.raise_for_status()

                logger.info(
                    f"Uploaded to custom URL",
                    extra={
                        "url_base": url.split("?")[0],
                        "size_bytes": len(data),
                        "content_type": content_type,
                    },
                )
                return url

            except httpx.HTTPError as e:
                logger.warning(
                    f"URL upload attempt {attempt + 1} failed",
                    extra={"url": url.split("?")[0], "error": str(e)},
                )
                if attempt == max_retries - 1:
                    raise R2UploadError(f"Failed to upload to URL after {max_retries} attempts: {e}")
                await asyncio.sleep(2**attempt)

        raise R2UploadError("Upload to URL failed")

    def upload_file(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> Tuple[str, str]:
        """Upload a file to R2 storage.

        Args:
            data: File content as bytes
            key: Storage key (path) in the bucket
            content_type: MIME type of the file

        Returns:
            Tuple of (r2_key, public_url)

        Raises:
            R2UploadError: If upload fails after retries
        """
        bucket = self.settings.r2_bucket_name
        max_retries = 3

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                self.client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )

                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Uploaded {key} to R2",
                    extra={
                        "key": key,
                        "size_bytes": len(data),
                        "content_type": content_type,
                        "duration_ms": duration_ms,
                    },
                )

                # Build public URL
                public_url = f"{self.settings.r2_public_url_base.rstrip('/')}/{key}"

                return key, public_url

            except ClientError as e:
                logger.warning(
                    f"R2 upload attempt {attempt + 1} failed",
                    extra={"key": key, "error": str(e)},
                )
                if attempt == max_retries - 1:
                    raise R2UploadError(f"Failed to upload after {max_retries} attempts: {e}")
                time.sleep(2**attempt)  # Exponential backoff

        # Should not reach here, but just in case
        raise R2UploadError("Upload failed")


    def upload_image(self, data: bytes, job_id: str, suffix: str = "") -> Tuple[str, str]:
        """Upload an image to R2.

        Args:
            data: Image data as bytes
            job_id: Job identifier
            suffix: Optional suffix (e.g., "_edited")

        Returns:
            Tuple of (r2_key, public_url)
        """
        key = f"outputs/images/{job_id}{suffix}.png"
        return self.upload_file(data, key, "image/png")

    def upload_video(self, data: bytes, job_id: str) -> Tuple[str, str]:
        """Upload a video to R2.

        Args:
            data: Video data as bytes
            job_id: Job identifier

        Returns:
            Tuple of (r2_key, public_url)
        """
        key = f"outputs/videos/{job_id}.mp4"
        return self.upload_file(data, key, "video/mp4")

    def upload_input_image(self, data: bytes, job_id: str, suffix: str = "") -> Tuple[str, str]:
        """Upload an input image to R2.

        Args:
            data: Image data as bytes
            job_id: Job identifier
            suffix: Optional suffix for the filename (e.g., "_end" for end frame)

        Returns:
            Tuple of (r2_key, public_url)
        """
        key = f"inputs/{job_id}/source{suffix}.png"
        return self.upload_file(data, key, "image/png")

    def test_connection(self) -> bool:
        """Test R2 connection by attempting to list objects.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.list_objects_v2(
                Bucket=self.settings.r2_bucket_name,
                MaxKeys=1,
            )
            logger.info("R2 connection test successful")
            return True
        except ClientError as e:
            logger.error(f"R2 connection test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"R2 connection test error: {e}")
            return False
