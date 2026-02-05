"""ACE-Step 1.5 Music Generator Service."""

import asyncio
import gc
import io
import logging
import random
from pathlib import Path
from typing import Any, Dict

from app.config import Settings
from app.models.internal import MusicGenerationParams, MusicGenerationResult
from app.services.interfaces import MusicGenerator

logger = logging.getLogger(__name__)


class ACEStepGenerator(MusicGenerator):
    """Music generator using ACE-Step 1.5."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._settings = settings
        self._model = None
        self._is_loaded = False
        self._dry_run = settings.audio_dry_run
        logger.info(f"ACEStepGenerator initialized (dry_run={self._dry_run})")

    @property
    def _loaded(self) -> bool:
        return self._is_loaded

    def load_models(self) -> None:
        if self._is_loaded:
            return
        if self._dry_run:
            self._is_loaded = True
            logger.info("ACEStepGenerator loaded in dry-run mode")
            return

        logger.info("Loading ACE-Step 1.5 models...")
        try:
            from acestep import ACEStepPipeline

            model_path = Path(self._settings.acestep_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"ACE-Step model not found at {model_path}")

            self._model = ACEStepPipeline.from_pretrained(str(model_path), device="cuda")
            self._is_loaded = True
            logger.info("ACE-Step 1.5 models loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import ACE-Step: {e}")
            logger.warning("Falling back to dry-run mode")
            self._dry_run = True
            self._is_loaded = True

    def unload_models(self) -> None:
        if not self._is_loaded:
            return
        logger.info("Unloading ACE-Step models...")
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("ACE-Step models unloaded")

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": "ace-step-1.5",
            "loaded": self._is_loaded,
            "dry_run": self._dry_run,
            "default_duration": self._settings.acestep_default_duration,
            "max_duration": self._settings.acestep_max_duration,
            "sample_rate": self._settings.acestep_sample_rate,
        }

    async def generate_music(self, params: MusicGenerationParams) -> MusicGenerationResult:
        if not self._is_loaded:
            raise RuntimeError("ACE-Step models not loaded")

        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        if self._dry_run:
            return await self._generate_dry_run(params, seed)

        return await asyncio.to_thread(self._generate_sync, params, seed)

    def _generate_sync(self, params: MusicGenerationParams, seed: int) -> MusicGenerationResult:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        gen_kwargs = {
            "prompt": params.prompt,
            "duration": params.duration_seconds,
            "seed": seed,
        }
        if params.lyrics:
            gen_kwargs["lyrics"] = params.lyrics

        audio_output = self._model.generate(**gen_kwargs)
        audio_bytes = self._encode_wav(audio_output, self._settings.acestep_sample_rate)

        return MusicGenerationResult(
            audio_data=audio_bytes,
            duration_seconds=params.duration_seconds,
            sample_rate=self._settings.acestep_sample_rate,
            seed=seed,
        )

    async def _generate_dry_run(self, params: MusicGenerationParams, seed: int) -> MusicGenerationResult:
        """Generate a silent WAV for dry-run testing."""
        import wave

        sample_rate = self._settings.acestep_sample_rate
        num_samples = int(sample_rate * params.duration_seconds)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b'\x00\x00\x00\x00' * num_samples)

        return MusicGenerationResult(
            audio_data=buffer.getvalue(),
            duration_seconds=params.duration_seconds,
            sample_rate=sample_rate,
            seed=seed,
        )

    def _encode_wav(self, audio_tensor, sample_rate: int) -> bytes:
        import numpy as np
        import wave

        if hasattr(audio_tensor, 'cpu'):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor

        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        if len(audio_int16.shape) == 1:
            num_channels = 1
        else:
            num_channels = audio_int16.shape[0]
            audio_int16 = audio_int16.T.flatten()

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()
