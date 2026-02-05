"""AudioGen Sound Effect Generator Service."""

import asyncio
import gc
import io
import logging
import random
from pathlib import Path
from typing import Any, Dict

from app.config import Settings
from app.models.internal import SoundEffectParams, SoundEffectResult
from app.services.interfaces import SoundEffectGenerator

logger = logging.getLogger(__name__)


class AudioGenGenerator(SoundEffectGenerator):
    """Sound effect generator using Facebook's AudioGen."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._settings = settings
        self._model = None
        self._is_loaded = False
        self._dry_run = settings.audio_dry_run
        logger.info(f"AudioGenGenerator initialized (dry_run={self._dry_run})")

    @property
    def _loaded(self) -> bool:
        return self._is_loaded

    def load_models(self) -> None:
        if self._is_loaded:
            return
        if self._dry_run:
            self._is_loaded = True
            logger.info("AudioGenGenerator loaded in dry-run mode")
            return

        logger.info("Loading AudioGen (facebook/audiogen-medium) models...")
        try:
            from audiocraft.models import AudioGen

            model_path = Path(self._settings.audiogen_model_path)
            if model_path.exists():
                self._model = AudioGen.get_pretrained(str(model_path))
            else:
                # Auto-download from HuggingFace
                logger.info("AudioGen model not found locally, downloading from HuggingFace...")
                self._model = AudioGen.get_pretrained('facebook/audiogen-medium')

            self._model.set_generation_params(duration=self._settings.audiogen_default_duration)
            self._is_loaded = True
            logger.info("AudioGen models loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import AudioCraft: {e}")
            logger.warning("Falling back to dry-run mode")
            self._dry_run = True
            self._is_loaded = True

    def unload_models(self) -> None:
        if not self._is_loaded:
            return
        logger.info("Unloading AudioGen models...")
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
        logger.info("AudioGen models unloaded")

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": "audiogen-medium",
            "loaded": self._is_loaded,
            "dry_run": self._dry_run,
            "default_duration": self._settings.audiogen_default_duration,
            "max_duration": self._settings.audiogen_max_duration,
            "sample_rate": self._settings.audiogen_sample_rate,
        }

    async def generate_sound_effect(self, params: SoundEffectParams) -> SoundEffectResult:
        if not self._is_loaded:
            raise RuntimeError("AudioGen models not loaded")

        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        if self._dry_run:
            return await self._generate_dry_run(params, seed)

        return await asyncio.to_thread(self._generate_sync, params, seed)

    def _generate_sync(self, params: SoundEffectParams, seed: int) -> SoundEffectResult:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Set duration for this generation
        self._model.set_generation_params(duration=params.duration_seconds)

        # Generate [batch, channels, samples]
        wav = self._model.generate([params.prompt])
        audio_tensor = wav[0].cpu()

        audio_bytes = self._encode_wav(audio_tensor, self._model.sample_rate)

        return SoundEffectResult(
            audio_data=audio_bytes,
            duration_seconds=params.duration_seconds,
            sample_rate=self._model.sample_rate,
            seed=seed,
        )

    async def _generate_dry_run(self, params: SoundEffectParams, seed: int) -> SoundEffectResult:
        """Generate a silent WAV for dry-run testing."""
        import wave

        sample_rate = self._settings.audiogen_sample_rate
        num_samples = int(sample_rate * params.duration_seconds)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b'\x00\x00' * num_samples)

        return SoundEffectResult(
            audio_data=buffer.getvalue(),
            duration_seconds=params.duration_seconds,
            sample_rate=sample_rate,
            seed=seed,
        )

    def _encode_wav(self, audio_tensor, sample_rate: int) -> bytes:
        import numpy as np
        import wave

        if hasattr(audio_tensor, 'numpy'):
            audio_np = audio_tensor.numpy()
        else:
            audio_np = audio_tensor

        if len(audio_np.shape) == 2:
            num_channels = audio_np.shape[0]
            audio_np = audio_np.T.flatten()
        else:
            num_channels = 1

        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()
