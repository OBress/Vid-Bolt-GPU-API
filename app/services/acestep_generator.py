"""ACE-Step 1.5 Music Generator Service.

Uses the ACE-Step 1.5 hybrid LM+DiT architecture for commercial-grade
music generation. Supports text-to-music with optional lyrics, Chain-of-Thought
reasoning via the 5Hz Language Model, and auto-model selection based on VRAM.

Ref: https://github.com/ace-step/ACE-Step-1.5
"""

import asyncio
import gc
import io
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import Settings
from app.models.internal import MusicGenerationParams, MusicGenerationResult
from app.services.interfaces import MusicGenerator

logger = logging.getLogger(__name__)


class ACEStepGenerator(MusicGenerator):
    """Music generator using ACE-Step 1.5 (hybrid LM + DiT architecture)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._settings = settings
        self._dit_handler = None
        self._llm_handler = None
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

        logger.info("Loading ACE-Step 1.5 models (hybrid LM + DiT)...")
        try:
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler

            # --- Initialize DiT Handler ---
            self._dit_handler = AceStepHandler()

            # ACE-Step 1.5 auto-detects project_root and auto-downloads models.
            # project_root should point to the repo root (which contains "checkpoints/").
            # The handler will create checkpoints/ and download models there if needed.
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "repos", "ACE-Step-1.5")
            )

            # Use turbo model for fast inference (8 steps, <2s per song on A100)
            config_path = "acestep-v15-turbo"

            logger.info(f"Initializing DiT handler (project_root={project_root}, config={config_path})")
            status_msg, success = self._dit_handler.initialize_service(
                project_root=project_root,
                config_path=config_path,
                device="cuda",
            )
            if not success:
                raise RuntimeError(f"DiT initialization failed: {status_msg}")
            logger.info(f"DiT handler initialized: {status_msg}")

            # --- Initialize LLM Handler ---
            self._llm_handler = LLMHandler()

            # checkpoint_dir is where the LM model lives (checkpoints/ under project root)
            checkpoint_dir = os.path.join(project_root, "checkpoints")

            # Default LM model (1.7B is a good balance of quality and speed)
            lm_model_path = "acestep-5Hz-lm-1.7B"

            logger.info(f"Initializing LLM handler (checkpoint_dir={checkpoint_dir}, lm_model={lm_model_path})")
            lm_status_msg, lm_success = self._llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend="vllm",
                device="cuda",
            )
            if not lm_success:
                # LLM is optional - DiT can work without it (no CoT reasoning)
                logger.warning(f"LLM initialization failed (DiT-only mode): {lm_status_msg}")
            else:
                logger.info(f"LLM handler initialized: {lm_status_msg}")

            self._is_loaded = True
            logger.info("ACE-Step 1.5 models loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import ACE-Step 1.5: {e}")
            logger.warning("Falling back to dry-run mode")
            self._dry_run = True
            self._is_loaded = True

    def unload_models(self) -> None:
        if not self._is_loaded:
            return
        logger.info("Unloading ACE-Step 1.5 models...")
        if self._dit_handler is not None:
            del self._dit_handler
            self._dit_handler = None
        if self._llm_handler is not None:
            del self._llm_handler
            self._llm_handler = None
        self._is_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("ACE-Step 1.5 models unloaded")

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": "ace-step-1.5",
            "loaded": self._is_loaded,
            "dry_run": self._dry_run,
            "dit_initialized": self._dit_handler is not None,
            "llm_initialized": (
                self._llm_handler is not None
                and self._llm_handler.llm_initialized
            ),
            "default_duration": self._settings.acestep_default_duration,
            "max_duration": self._settings.acestep_max_duration,
            "sample_rate": self._settings.acestep_sample_rate,
        }

    async def generate_music(self, params: MusicGenerationParams) -> MusicGenerationResult:
        if not self._is_loaded:
            raise RuntimeError("ACE-Step 1.5 models not loaded")

        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        if self._dry_run:
            return await self._generate_dry_run(params, seed)

        return await asyncio.to_thread(self._generate_sync, params, seed)

    def _generate_sync(self, params: MusicGenerationParams, seed: int) -> MusicGenerationResult:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # Map our API params to ACE-Step 1.5 GenerationParams
        gen_params = GenerationParams(
            caption=params.prompt,
            lyrics=params.lyrics or "",
            instrumental=not bool(params.lyrics),
            duration=params.duration_seconds,
            seed=seed,
            # Turbo model defaults (acestep-v15-turbo uses shift=1.0)
            inference_steps=8,
            shift=1.0,  # Must match model variant: v15-turbo=1.0, v15-turbo-shift3=3.0
            # Enable Chain-of-Thought reasoning for better quality
            thinking=True,
            use_cot_metas=True,
            use_cot_caption=True,
            use_cot_language=True,
        )

        gen_config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,  # We manage our own seed
            seeds=[seed],
            audio_format="wav",  # We need raw audio for upload
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_music(
                dit_handler=self._dit_handler,
                llm_handler=self._llm_handler,
                params=gen_params,
                config=gen_config,
                save_dir=tmpdir,
            )

            if not result.success:
                raise RuntimeError(f"ACE-Step 1.5 generation failed: {result.error}")

            if not result.audios:
                raise RuntimeError("ACE-Step 1.5 returned no audio")

            # Read the generated audio file
            audio_info = result.audios[0]
            audio_path = audio_info.get("path", "")

            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
            elif audio_info.get("tensor") is not None:
                # Fallback: encode tensor directly to WAV
                audio_bytes = self._encode_wav(
                    audio_info["tensor"],
                    audio_info.get("sample_rate", self._settings.acestep_sample_rate),
                )
            else:
                raise RuntimeError("ACE-Step 1.5 generated no audio output")

        return MusicGenerationResult(
            audio_data=audio_bytes,
            duration_seconds=params.duration_seconds,
            sample_rate=audio_info.get("sample_rate", self._settings.acestep_sample_rate),
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
