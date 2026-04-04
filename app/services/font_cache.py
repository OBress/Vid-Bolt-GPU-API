"""Remote font caching for segmentation labels."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from PIL import ImageFont

from app.services.storage import StorageService

logger = logging.getLogger(__name__)


@dataclass
class FontCacheResult:
    """Result of resolving a remote font asset."""

    path: Optional[Path]
    warning: Optional[str] = None


class FontCacheService:
    """Download and persist font files for reuse across segmentation jobs."""

    _SUPPORTED_EXTENSIONS = {".ttf", ".otf"}

    def __init__(self, cache_dir: Optional[Path] = None):
        root = Path(__file__).resolve().parents[2]
        self.cache_dir = cache_dir or root / "cache" / "fonts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}

    @staticmethod
    def normalize_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url
        return parsed._replace(query="", fragment="").geturl()

    async def ensure_font_cached(
        self,
        storage: StorageService,
        font_url: str,
    ) -> FontCacheResult:
        """Ensure a remote font exists on local disk and is loadable."""
        normalized_url = self.normalize_url(font_url)
        parsed = urlparse(normalized_url)
        extension = Path(parsed.path).suffix.lower()

        if extension not in self._SUPPORTED_EXTENSIONS:
            return FontCacheResult(
                path=None,
                warning=(
                    f"Unsupported font format for '{font_url}'. "
                    "Only .ttf and .otf fonts are supported; using default font."
                ),
            )

        key = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
        lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            cached_path = self._find_cached_font(key, extension)
            if cached_path is not None and self._font_is_loadable(cached_path):
                return FontCacheResult(path=cached_path)

            try:
                font_bytes = await storage.download_from_url(font_url)
            except Exception as exc:
                logger.warning("Failed to download font asset %s: %s", font_url, exc)
                return FontCacheResult(
                    path=None,
                    warning=f"Failed to download font '{font_url}'; using default font.",
                )

            content_hash = hashlib.sha256(font_bytes).hexdigest()[:16]
            font_path = self.cache_dir / f"{key}_{content_hash}{extension}"

            if not font_path.exists():
                font_path.write_bytes(font_bytes)

            if not self._font_is_loadable(font_path):
                try:
                    font_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove invalid cached font %s", font_path)
                return FontCacheResult(
                    path=None,
                    warning=f"Font '{font_url}' could not be loaded; using default font.",
                )

            return FontCacheResult(path=font_path)

    def _find_cached_font(self, key: str, extension: str) -> Optional[Path]:
        candidates = sorted(self.cache_dir.glob(f"{key}_*{extension}"))
        for candidate in candidates:
            if self._font_is_loadable(candidate):
                return candidate
        return None

    @staticmethod
    def _font_is_loadable(path: Path) -> bool:
        try:
            ImageFont.truetype(str(path), size=16)
            return True
        except Exception:
            return False


_font_cache_service: Optional[FontCacheService] = None


def get_font_cache_service() -> FontCacheService:
    """Return a shared font cache service instance."""
    global _font_cache_service
    if _font_cache_service is None:
        _font_cache_service = FontCacheService()
    return _font_cache_service
