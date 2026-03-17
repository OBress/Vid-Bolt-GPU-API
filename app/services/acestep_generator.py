"""ACE-Step 1.5 Music Generator Service.

Uses the ACE-Step 1.5 hybrid LM+DiT architecture for commercial-grade
music generation. Supports text-to-music with optional lyrics, Chain-of-Thought
reasoning via the 5Hz Language Model, and metadata control (BPM, key, time sig).

Target hardware: NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM)
- Uses acestep-5Hz-lm-4B (best quality LM, fits easily in VRAM)
- No CPU offloading needed
- Flash attention enabled

Ref: https://github.com/ace-step/ACE-Step-1.5
API: repos/ACE-Step-1.5/docs/en/INFERENCE.md
"""

import asyncio
import gc
import io
import logging
import os
import random
import tempfile
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

    # =========================================================================
    # Model Lifecycle
    # =========================================================================

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

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "repos", "ACE-Step-1.5")
            )
            checkpoint_dir = os.path.join(project_root, "checkpoints")

            # --- Initialize DiT Handler ---
            self._dit_handler = AceStepHandler()

            # acestep-v15-turbo: 8-step turbo distilled model (best speed/quality)
            config_path = "acestep-v15-turbo"

            logger.info(f"Initializing DiT handler (project_root={project_root}, config={config_path})")
            status_msg, success = self._dit_handler.initialize_service(
                project_root=project_root,
                config_path=config_path,
                device="cuda",
                use_flash_attention=True,
            )
            if not success:
                raise RuntimeError(f"DiT initialization failed: {status_msg}")
            logger.info(f"DiT handler initialized: {status_msg}")

            # --- Initialize LLM Handler ---
            # RTX PRO 6000 Blackwell has ~96GB VRAM → use 4B model for best quality
            self._llm_handler = LLMHandler()
            lm_model_path = "acestep-5Hz-lm-4B"

            logger.info(f"Initializing LLM handler (checkpoint_dir={checkpoint_dir}, lm_model={lm_model_path})")
            lm_status_msg, lm_success = self._llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend="vllm",
                device="cuda",
            )
            if not lm_success:
                # LLM is optional — DiT can work without it (no CoT reasoning)
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
        
        try:
            import torch
            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"  VRAM before ACE-Step unload: {vram_before:.2f}GB")
        except ImportError:
            torch = None
        
        # CRITICAL: Explicitly destroy vLLM/nanovllm engine to free GPU memory.
        # Without this, nanovllm's model runner + KV cache (~17.5GB) persist as zombie 
        # allocations because atexit.register(self.exit) only fires at process exit,
        # not when the Python reference is dropped.
        if self._llm_handler is not None:
            try:
                if (hasattr(self._llm_handler, 'llm') 
                    and self._llm_handler.llm is not None
                    and hasattr(self._llm_handler.llm, 'exit')):
                    logger.info("  Destroying nanovllm engine (freeing model runner + KV cache)...")
                    self._llm_handler.llm.exit()
            except Exception as e:
                logger.warning(f"  Failed to exit nanovllm engine: {e}")
            
            # Call LLMHandler's own unload (cleans up tokenizer, distributed state, etc.)
            try:
                if hasattr(self._llm_handler, 'unload'):
                    self._llm_handler.unload()
            except Exception as e:
                logger.warning(f"  Failed to unload LLM handler: {e}")
            
            del self._llm_handler
            self._llm_handler = None
        
        # Clean up DiT handler (moves models off GPU)
        if self._dit_handler is not None:
            try:
                # Try to move DiT model to CPU before deleting
                if hasattr(self._dit_handler, 'model') and self._dit_handler.model is not None:
                    if hasattr(self._dit_handler.model, 'cpu'):
                        self._dit_handler.model.cpu()
                if hasattr(self._dit_handler, 'vae') and self._dit_handler.vae is not None:
                    if hasattr(self._dit_handler.vae, 'cpu'):
                        self._dit_handler.vae.cpu()
                if hasattr(self._dit_handler, 'text_encoder') and self._dit_handler.text_encoder is not None:
                    if hasattr(self._dit_handler.text_encoder, 'cpu'):
                        self._dit_handler.text_encoder.cpu()
            except Exception as e:
                logger.warning(f"  Failed to offload DiT models to CPU: {e}")
            del self._dit_handler
            self._dit_handler = None
        
        self._is_loaded = False
        gc.collect()
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram_after = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"  VRAM after ACE-Step unload: {vram_after:.2f}GB")
        
        logger.info("ACE-Step 1.5 models unloaded successfully")

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": "ace-step-1.5",
            "loaded": self._is_loaded,
            "dry_run": self._dry_run,
            "dit_initialized": self._dit_handler is not None,
            "llm_initialized": (
                self._llm_handler is not None
                and getattr(self._llm_handler, "llm_initialized", False)
            ),
            "dit_model": "acestep-v15-turbo",
            "lm_model": "acestep-5Hz-lm-4B",
            "default_duration": self._settings.acestep_default_duration,
            "max_duration": self._settings.acestep_max_duration,
            "sample_rate": self._settings.acestep_sample_rate,
        }

    # =========================================================================
    # Music Generation
    # =========================================================================

    async def generate_music(self, params: MusicGenerationParams) -> MusicGenerationResult:
        if not self._is_loaded:
            raise RuntimeError("ACE-Step 1.5 models not loaded")

        seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

        if self._dry_run:
            return await self._generate_dry_run(params, seed)

        return await asyncio.to_thread(self._generate_sync, params, seed)

    def _generate_sync(self, params: MusicGenerationParams, seed: int) -> MusicGenerationResult:
        """Run ACE-Step 1.5 generation synchronously (called via asyncio.to_thread)."""
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # Map our API params to ACE-Step 1.5 GenerationParams
        # See: repos/ACE-Step-1.5/docs/en/INFERENCE.md
        gen_params = GenerationParams(
            # Text inputs
            caption=params.prompt,
            lyrics=params.lyrics or "",
            instrumental=not bool(params.lyrics),
            # Metadata (None/empty = auto-detect via LM CoT)
            duration=params.duration_seconds,
            bpm=params.bpm,
            keyscale=params.key_scale or "",
            timesignature=params.time_signature or "",
            vocal_language=params.vocal_language or "unknown",
            # Generation settings (turbo model defaults)
            inference_steps=8,
            seed=seed,
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
            audio_format="wav",
        )

        logger.info(
            f"Generating music: prompt={params.prompt!r}, "
            f"lyrics={'yes' if params.lyrics else 'no'}, "
            f"duration={params.duration_seconds}s, seed={seed}, "
            f"bpm={params.bpm}, key={params.key_scale}, time_sig={params.time_signature}"
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

            # Log generation info
            if result.extra_outputs:
                time_costs = result.extra_outputs.get("time_costs", {})
                if time_costs:
                    logger.info(f"Generation time costs: {time_costs}")

            # Read the generated audio file
            audio_info = result.audios[0]
            audio_path = audio_info.get("path", "")
            sample_rate = audio_info.get("sample_rate", self._settings.acestep_sample_rate)

            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                logger.info(f"Read audio file: {audio_path} ({len(audio_bytes)} bytes)")

            elif audio_info.get("tensor") is not None:
                # Fallback: encode tensor directly to WAV
                logger.info("No audio file found, encoding tensor to WAV")
                audio_bytes = self._encode_wav(audio_info["tensor"], sample_rate)

            else:
                raise RuntimeError("ACE-Step 1.5 generated no audio output")

        return MusicGenerationResult(
            audio_data=audio_bytes,
            duration_seconds=params.duration_seconds,
            sample_rate=sample_rate,
            seed=seed,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

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
        """Encode an audio tensor to WAV bytes."""
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
