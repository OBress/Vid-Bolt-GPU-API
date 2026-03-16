"""Model download service with progress tracking.

Automatically downloads required models from HuggingFace on startup
when MOCK_MODE=false and models are missing.
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


class DownloadStatus(str, Enum):
    """Model download status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Already exists


@dataclass
class ModelDownloadProgress:
    """Progress info for a single model download."""
    model_name: str
    status: DownloadStatus = DownloadStatus.PENDING
    progress_percent: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class OverallDownloadStatus:
    """Overall download status for all models."""
    status: str = "pending"  # pending, downloading, completed, failed
    ready: bool = False
    total_models: int = 0
    completed_models: int = 0
    current_model: Optional[str] = None
    models: dict[str, ModelDownloadProgress] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ModelDownloader:
    """Service for downloading models from HuggingFace with progress tracking."""

    # Model definitions: (name, hf_repo, local_path, files_or_full)
    MODELS = [
        {
            "name": "z-image-turbo",
            "repo": "Tongyi-MAI/Z-Image-Turbo",
            "local_dir": "models/z-image-turbo",
            "type": "full",  # Download entire repo
            "indicator_file": "model_index.json",
        },
        {
            "name": "qwen-image-edit-2511",
            "repo": "Qwen/Qwen-Image-Edit-2511",
            "local_dir": "models/qwen-image-edit-2511",
            "type": "full",
            "indicator_file": "config.json",
        },
        {
            "name": "lightx2v-lora",
            "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "local_dir": "models/loras/qwen-image-edit-2511",
            "type": "file",
            "filename": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
        },
        {
            "name": "qwen-multiple-angles-lora",
            "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            "local_dir": "models/loras/qwen-image-edit-multiple-angles",
            "type": "file",
            "filename": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        },
        {
            "name": "ltx2-checkpoint",
            "repo": "Lightricks/LTX-2.3-fp8",
            "local_dir": "models/ltx-2",
            "type": "file",
            "filename": "ltx-2.3-22b-dev-fp8.safetensors",
        },
        {
            "name": "ltx2-spatial-upsampler",
            "repo": "Lightricks/LTX-2.3",
            "local_dir": "models/ltx-2",
            "type": "file",
            "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        },
        {
            "name": "ltx2-distilled-lora",
            "repo": "Lightricks/LTX-2.3",
            "local_dir": "models/ltx-2",
            "type": "file",
            "filename": "ltx-2.3-22b-distilled-lora-384.safetensors",
        },
        {
            "name": "gemma-text-encoder",
            "repo": "google/gemma-3-12b-it-qat-q4_0-unquantized",
            "local_dir": "models/ltx-2/gemma-3-12b-it-qat",
            "type": "full",
            "indicator_file": "config.json",
        },
    ]

    def __init__(self, base_path: Path):
        """Initialize the downloader.
        
        Args:
            base_path: Base path for model storage (project root)
        """
        self.base_path = base_path
        self.status = OverallDownloadStatus()
        self._download_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Initialize model progress tracking
        for model in self.MODELS:
            self.status.models[model["name"]] = ModelDownloadProgress(
                model_name=model["name"]
            )
        self.status.total_models = len(self.MODELS)

    def _model_exists(self, model: dict) -> bool:
        """Check if a model already exists locally."""
        local_path = self.base_path / model["local_dir"]
        
        if model["type"] == "file":
            return (local_path / model["filename"]).exists()
        else:
            # For full repos, check for specific indicator file
            indicator = model.get("indicator_file", "config.json")
            return (local_path / indicator).exists()

    def check_all_models_exist(self) -> bool:
        """Check if all required models are already downloaded."""
        for model in self.MODELS:
            if not self._model_exists(model):
                return False
        return True

    def get_status(self) -> OverallDownloadStatus:
        """Get current download status (thread-safe)."""
        with self._lock:
            return self.status

    def is_ready(self) -> bool:
        """Check if all downloads are complete and ready."""
        with self._lock:
            return self.status.ready

    def start_download(self) -> None:
        """Start downloading models in background thread."""
        if self._download_thread and self._download_thread.is_alive():
            logger.warning("Download already in progress")
            return

        self._download_thread = threading.Thread(
            target=self._download_all_models,
            daemon=True,
            name="model-downloader"
        )
        self._download_thread.start()
        logger.info("Started background model download")

    def _download_all_models(self) -> None:
        """Download all models (runs in background thread)."""
        with self._lock:
            self.status.status = "downloading"
            self.status.started_at = datetime.utcnow()

        try:
            for model in self.MODELS:
                model_name = model["name"]
                
                with self._lock:
                    self.status.current_model = model_name
                    self.status.models[model_name].status = DownloadStatus.DOWNLOADING
                    self.status.models[model_name].started_at = datetime.utcnow()

                try:
                    if self._model_exists(model):
                        logger.info(f"Model {model_name} already exists, skipping")
                        with self._lock:
                            self.status.models[model_name].status = DownloadStatus.SKIPPED
                            self.status.models[model_name].progress_percent = 100.0
                            self.status.models[model_name].completed_at = datetime.utcnow()
                            self.status.completed_models += 1
                        continue

                    self._download_model(model)

                    with self._lock:
                        self.status.models[model_name].status = DownloadStatus.COMPLETED
                        self.status.models[model_name].progress_percent = 100.0
                        self.status.models[model_name].completed_at = datetime.utcnow()
                        self.status.completed_models += 1
                        
                    logger.info(f"Completed download: {model_name}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to download {model_name}: {error_msg}")
                    with self._lock:
                        self.status.models[model_name].status = DownloadStatus.FAILED
                        self.status.models[model_name].error = error_msg
                    # Continue with other models

            # Check final status
            with self._lock:
                failed = any(
                    m.status == DownloadStatus.FAILED 
                    for m in self.status.models.values()
                )
                if failed:
                    self.status.status = "failed"
                    self.status.error = "One or more models failed to download"
                else:
                    self.status.status = "completed"
                    self.status.ready = True
                self.status.completed_at = datetime.utcnow()
                self.status.current_model = None

        except Exception as e:
            logger.exception(f"Download process failed: {e}")
            with self._lock:
                self.status.status = "failed"
                self.status.error = str(e)
                self.status.completed_at = datetime.utcnow()

    def _download_model(self, model: dict) -> None:
        """Download a single model from HuggingFace."""
        local_dir = self.base_path / model["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {model['name']} from {model['repo']}...")

        if model["type"] == "file":
            # Download single file
            hf_hub_download(
                repo_id=model["repo"],
                filename=model["filename"],
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        else:
            # Download entire repository
            snapshot_download(
                repo_id=model["repo"],
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )


# Global instance
_downloader: Optional[ModelDownloader] = None


def get_model_downloader() -> Optional[ModelDownloader]:
    """Get the global model downloader instance."""
    return _downloader


def init_model_downloader(base_path: Path) -> ModelDownloader:
    """Initialize and return the global model downloader."""
    global _downloader
    _downloader = ModelDownloader(base_path)
    return _downloader
