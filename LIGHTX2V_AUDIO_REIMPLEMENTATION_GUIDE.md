# LightX2V & Audio Generation Re-Implementation Guide

This document provides everything needed to re-implement the LightX2V (LTX-2) video generation backend and audio generation systems (ACE-Step music, AudioGen SFX) after reverting the repository.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [Model Downloads](#3-model-downloads)
4. [Configuration](#4-configuration)
5. [LightX2V Video Generation](#5-lightx2v-video-generation)
6. [Audio Generation (Music & SFX)](#6-audio-generation-music--sfx)
7. [VRAM Management](#7-vram-management)
8. [API Routers](#8-api-routers)
9. [Internal Data Models](#9-internal-data-models)
10. [Docker Integration](#10-docker-integration)
11. [Dependencies](#11-dependencies)
12. [Known Issues & Troubleshooting](#12-known-issues--troubleshooting)

---

## 1. Overview

### Systems Covered

| System               | Model                   | VRAM  | License      | Purpose                                       |
| -------------------- | ----------------------- | ----- | ------------ | --------------------------------------------- |
| **LTX-2 Video**      | LTX-2 19B Distilled FP8 | ~40GB | Lightricks   | Image-to-Video (I2V), Keyframe Interpolation  |
| **Music Generation** | ACE-Step 1.5            | <4GB  | MIT          | Text-to-Music with lyrics support (10s-10min) |
| **Sound Effects**    | AudioGen Medium (1.5B)  | ~16GB | CC-BY-NC 4.0 | Text-to-SFX (1s-30s)                          |

### Key Architecture Patterns

- **LightX2V**: Unified pipeline for LTX-2 with FP8 per-tensor quantization
- **Direct Runner Flow**: Single `create_generator()` call, attribute patching for per-request params
- **Dynamic Audio Loading**: Load-on-demand in ALL mode, proactive unload before video tasks
- **Async Job Pattern**: Request returns 202, execution fetches generator at task time

---

## 2. Directory Structure

### New Files to Create

```
app/
├── routers/
│   ├── music_generation.py          # POST /api/v1/music/generate
│   └── sound_effect_generation.py   # POST /api/v1/sfx/generate
├── services/
│   ├── acestep_generator.py         # ACEStepGenerator class
│   ├── audiogen_generator.py        # AudioGenGenerator class
│   ├── ltx2_generator.py            # LTX2Generator class (LightX2V backend)
│   └── interfaces.py                # Add MusicGenerator, SoundEffectGenerator ABCs
├── models/
│   ├── music_generation.py          # MusicGenerateRequest, MusicGenerateResponse
│   ├── sound_effect_generation.py   # SoundEffectGenerateRequest, SoundEffectGenerateResponse
│   └── internal.py                  # Add MusicGenerationParams, SoundEffectParams, etc.
├── config.py                        # Add audio model paths and inference settings
└── main.py                          # Register music/sfx routers

LightX2V/                            # Vendored submodule (ModelTC/LightX2V fork)
└── lightx2v/                        # Core pipeline code

models/
├── ace-step-1.5/                    # ACE-Step weights (~4GB)
├── audiogen-medium/                 # AudioGen weights (~16GB)
└── ltx-2/
    ├── ltx-2-19b-distilled-fp8.safetensors    # Main FP8 checkpoint (~27GB)
    ├── ltx-2-spatial-upscaler-x2-1.0.safetensors
    ├── ltx-2-19b-distilled-lora-384.safetensors
    ├── transformer/config.json              # FROM Lightricks/LTX-2 HuggingFace
    └── gemma-3-12b-it-qat-q4_0-unquantized/ # Text encoder (~12GB)

scripts/
├── download-models.sh               # Model download script
└── docker-entrypoint.sh             # Handles FP8 model download on startup
```

---

## 3. Model Downloads

### scripts/download-models.sh

Download all required models from HuggingFace:

```bash
#!/bin/bash
# LTX-2 Components (~40GB total)

# Main FP8 checkpoint
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled-fp8.safetensors \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

# Spatial upsampler
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

# Distilled LoRA
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled-lora-384.safetensors \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

# Transformer config.json (CRITICAL - LightX2V reads from transformer/config.json)
mkdir -p "$MODELS_DIR/ltx-2/transformer"
huggingface-cli download Lightricks/LTX-2 \
    transformer/config.json \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

# Gemma Text Encoder (~12GB)
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$MODELS_DIR/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized" \
    --local-dir-use-symlinks False

# ACE-Step 1.5 (if model is available on HuggingFace)
# huggingface-cli download ace-step/ACE-Step-1.5 \
#     --local-dir "$MODELS_DIR/ace-step-1.5" \
#     --local-dir-use-symlinks False

# AudioGen - Auto-downloads from facebook/audiogen-medium on first use
```

---

## 4. Configuration

### app/config.py Additions

```python
class ModelPaths:
    """Hardcoded model paths relative to project root."""

    # LTX-2 (video generation)
    LTX2_CHECKPOINT = "models/ltx-2/ltx-2-19b-distilled-fp8.safetensors"
    LTX2_SPATIAL_UPSAMPLER = "models/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    LTX2_GEMMA_ROOT = "models/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized"
    LTX2_DISTILLED_LORA = "models/ltx-2/ltx-2-19b-distilled-lora-384.safetensors"

    # ACE-Step 1.5 (music generation)
    ACESTEP_MODEL = "models/ace-step-1.5"

    # AudioGen (sound effect generation)
    AUDIOGEN_MODEL = "models/audiogen-medium"


class InferenceConfig:
    """Hardcoded inference parameters."""

    # LTX-2 settings (Optimized for Distilled Model)
    LTX2_FP8_ENABLED = True  # FP8 for faster inference, ~20GB VRAM
    LTX2_NUM_INFERENCE_STEPS = 8  # Distilled model uses 8 predefined sigma values
    LTX2_CFG_GUIDANCE_SCALE = 1.0  # Distilled LoRA works best without CFG
    LTX2_DEFAULT_FRAME_RATE = 24.0
    LTX2_CONCURRENT_ENABLED = True
    LTX2_MAX_CONCURRENT_VIDEOS = 3
    LTX2_CONCURRENT_VRAM_BUDGET_GB = 72.0

    # ACE-Step (music) settings
    ACESTEP_DEFAULT_DURATION = 30.0   # seconds
    ACESTEP_MAX_DURATION = 600.0      # 10 minutes max
    ACESTEP_SAMPLE_RATE = 44100       # 44.1kHz output

    # AudioGen (sound effects) settings
    AUDIOGEN_DEFAULT_DURATION = 5.0   # seconds
    AUDIOGEN_MAX_DURATION = 30.0      # 30 seconds max
    AUDIOGEN_SAMPLE_RATE = 16000      # 16kHz output (AudioGen native)

    # Job timeouts (seconds)
    AUDIO_JOB_TIMEOUT = 600  # 10 minutes for audio jobs


class Settings(BaseSettings):
    # Add properties for audio
    @property
    def acestep_model_path(self) -> str:
        return ModelPaths.ACESTEP_MODEL

    @property
    def audiogen_model_path(self) -> str:
        return ModelPaths.AUDIOGEN_MODEL

    @property
    def acestep_default_duration(self) -> float:
        return InferenceConfig.ACESTEP_DEFAULT_DURATION

    @property
    def acestep_max_duration(self) -> float:
        return InferenceConfig.ACESTEP_MAX_DURATION

    @property
    def acestep_sample_rate(self) -> int:
        return InferenceConfig.ACESTEP_SAMPLE_RATE

    @property
    def audiogen_default_duration(self) -> float:
        return InferenceConfig.AUDIOGEN_DEFAULT_DURATION

    @property
    def audiogen_max_duration(self) -> float:
        return InferenceConfig.AUDIOGEN_MAX_DURATION

    @property
    def audiogen_sample_rate(self) -> int:
        return InferenceConfig.AUDIOGEN_SAMPLE_RATE

    @property
    def audio_job_timeout(self) -> int:
        return InferenceConfig.AUDIO_JOB_TIMEOUT

    @property
    def audio_dry_run(self) -> bool:
        return self.mock_mode
```

---

## 5. LightX2V Video Generation

### Core Constants

```python
# LightX2V LTX-2 distilled schedule sigma values (8-step)
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# FP8 skip blocks for quality preservation (input/output layers)
FP8_SKIP_BLOCK_INDEX = [0, 43, 44, 45, 46, 47]
```

### Environment Variables (Critical)

Set in `app/services/ltx2_generator.py` BEFORE importing LightX2V:

```python
import os
os.environ["DTYPE"] = "BF16"              # Precision for non-quantized components
os.environ["PROFILING_DEBUG_LEVEL"] = "0" # MUST be 0 for production (2 = 50% slowdown)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Pipeline Initialization Pattern

```python
from lightx2v.pipelines import LightX2VPipeline

# Load models (call once at startup)
def load_models(self):
    self._pipe = LightX2VPipeline(
        model_path=self.settings.ltx2_checkpoint_path,
        config_path="configs/ltx2/ltx2t_distilled_i2v_fp8.yaml",  # OR create config inline
        device="cuda",
    )

    # Mandatory attribute patching (if config.json is missing/incomplete)
    self._pipe.task = "i2v"  # Use "i2av" for Image-to-Audio-Video
    self._pipe.num_layers = 48
    self._pipe.num_attention_heads = 32
    self._pipe.attention_head_dim = 128
    self._pipe.rope_type = "split"
    self._pipe.attn_type = "torch_sdpa"  # Production default
    self._pipe.vae_scale_factors = [8, 32, 32]
    self._pipe.use_tiling_vae = True  # CRITICAL for 1080p

    # Initialize generator (locks config)
    self._pipe.create_generator()
```

### Per-Request Generation

```python
import types
import gc
import torch

def generate_video(self, params):
    # Update per-request params (attribute sync)
    self._pipe.prompt = params.prompt
    self._pipe.negative_prompt = params.negative_prompt
    self._pipe.seed = seed
    self._pipe.target_shape = (target_height, target_width)
    self._pipe.image_path = ",".join(keyframe_paths)
    self._pipe.image_strength = [1.0, 1.0]  # For start/end keyframes

    # Run pipeline
    result = self._pipe.runner.run_pipeline(input_info)

    # Memory barrier before VAE materialization
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Handle generator result (video is often a generator, not tensor)
    video_result = result.get("video")
    if isinstance(video_result, types.GeneratorType):
        video_tensor = next(video_result)  # Materialize
    else:
        video_tensor = video_result

    # Crop from padded dimensions (e.g., 1088 -> 1080)
    video_tensor = self._crop_and_trim_video(video_tensor, ...)

    # Encode to MP4
    from ltx2_media_io import encode_video
    return encode_video(video_tensor, ...)
```

### Critical Notes

1. **64-Pixel Alignment**: LTX-2 requires dimensions divisible by 64. Pad to 1088 for 1080p.
2. **16-Pixel VAE Constraint**: Internal latent space requires divisibility by 16.
3. **VAE Tiling**: Mandatory for 1080p (prevents ~26GB allocation OOM).
4. **Task Selection**: Use `"i2v"` for video-only (faster); `"i2av"` adds audio (2x slower).
5. **Sigma Schedule**: The distilled model uses predefined 8-step sigmas, not computed.

---

## 6. Audio Generation (Music & SFX)

### 6.1 Service Interfaces

Add to `app/services/interfaces.py`:

```python
class MusicGenerator(BaseModelGenerator):
    """Interface for music generators."""

    @abstractmethod
    async def generate_music(self, params: "MusicGenerationParams") -> "MusicGenerationResult":
        """Generate music from a text prompt and optional lyrics."""
        pass


class SoundEffectGenerator(BaseModelGenerator):
    """Interface for sound effect generators."""

    @abstractmethod
    async def generate_sound_effect(self, params: "SoundEffectParams") -> "SoundEffectResult":
        """Generate a sound effect from a text description."""
        pass
```

### 6.2 ACE-Step Generator (Music)

Create `app/services/acestep_generator.py`:

```python
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
            return

        logger.info("Loading ACE-Step 1.5 models...")
        try:
            from acestep.pipeline_ace_step import ACEStepPipeline

            model_path = Path(self._settings.acestep_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"ACE-Step model not found at {model_path}")

            self._model = ACEStepPipeline.from_pretrained(str(model_path), device="cuda")
            self._is_loaded = True
            logger.info("ACE-Step 1.5 models loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import ACE-Step: {e}")
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
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
```

### 6.3 AudioGen Generator (SFX)

Create `app/services/audiogen_generator.py`:

```python
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
            return

        logger.info("Loading AudioGen (facebook/audiogen-medium) models...")
        try:
            from audiocraft.models import AudioGen

            model_path = Path(self._settings.audiogen_model_path)
            if model_path.exists():
                self._model = AudioGen.get_pretrained(str(model_path))
            else:
                # Auto-download from HuggingFace
                self._model = AudioGen.get_pretrained('facebook/audiogen-medium')

            self._model.set_generation_params(duration=self._settings.audiogen_default_duration)
            self._is_loaded = True
            logger.info("AudioGen models loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import AudioCraft: {e}")
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
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
```

---

## 7. VRAM Management

### VRAMLoadMode Enum

Add to `app/services/model_manager.py`:

```python
from enum import Enum

class VRAMLoadMode(str, Enum):
    """VRAM loading mode - defines which models are loaded."""
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    VIDEO_GENERATION = "video_generation"
    AUDIO_CREATION = "audio_creation"  # NEW: ACE-Step + AudioGen
    ALL = "all"


class JobType(str, Enum):
    """Job type for scheduling purposes."""
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    VIDEO_GENERATION = "video_generation"
    MUSIC_GENERATION = "music_generation"           # NEW
    SOUND_EFFECT_GENERATION = "sound_effect_generation"  # NEW
```

### Audio Mode Switching

```python
def _switch_to_audio_creation_mode(self):
    """Switch to Audio Creation mode (ACE-Step + AudioGen)."""
    logger.info("Switching to AUDIO_CREATION mode...")

    # Unload all other models
    self._unload_zimage()
    self._unload_lightx2v()
    self._unload_ltx2()

    # Force GC
    gc.collect()
    torch.cuda.empty_cache()

    # Load audio models
    self._load_acestep()
    self._load_audiogen()

    self._current_mode = VRAMLoadMode.AUDIO_CREATION
    logger.info("AUDIO_CREATION mode active")

def ensure_mode_for_job(self, job_type: JobType) -> bool:
    """Ensure the system can handle the given job type."""

    if job_type == JobType.MUSIC_GENERATION or job_type == JobType.SOUND_EFFECT_GENERATION:
        if self._current_mode == VRAMLoadMode.AUDIO_CREATION:
            return True
        elif self._current_mode == VRAMLoadMode.ALL:
            # Dynamic loading in ALL mode
            if not self._audio_dynamic_loaded:
                self._load_acestep()
                self._load_audiogen()
                self._audio_dynamic_loaded = True
            return True
        else:
            return False  # Cannot switch automatically

    # ... existing logic for other job types
```

### Dynamic Audio Loading (ALL Mode)

```python
def _load_all_models(self):
    """Load all models into VRAM (ALL mode)."""
    logger.info("Loading all models into VRAM...")

    # Note: Audio models NOT loaded initially for VRAM headroom
    # They load dynamically when audio requests come in
    self._load_zimage()
    self._load_lightx2v()
    self._load_ltx2()

    # Audio loads on-demand via ensure_mode_for_job()
    self._audio_dynamic_loaded = False

    self._current_mode = VRAMLoadMode.ALL
```

### Model Manager Methods

```python
def get_music_generator(self) -> MusicGenerator:
    """Get the music generator (ACE-Step)."""
    if self._acestep_generator is None:
        raise RuntimeError("ACE-Step generator not loaded")
    return self._acestep_generator

def get_sound_effect_generator(self) -> SoundEffectGenerator:
    """Get the sound effect generator (AudioGen)."""
    if self._audiogen_generator is None:
        raise RuntimeError("AudioGen generator not loaded")
    return self._audiogen_generator
```

---

## 8. API Routers

### 8.1 Music Generation Router

Create `app/routers/music_generation.py`:

```python
"""Music generation endpoint."""

import logging
import time

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.models.common import ErrorResponse
from app.models.music_generation import MusicGenerateRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import MusicGenerationParams
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/music",
    tags=["Music Generation"],
)


@router.post(
    "/generate",
    response_model=AsyncJobResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "System busy"},
        503: {"model": ErrorResponse, "description": "Audio mode not active"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Generate Music",
    description="Generate music from a text prompt using ACE-Step 1.5.",
)
async def generate_music(
    request: Request,
    body: MusicGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    params = MusicGenerationParams(
        job_id=body.job_id,
        prompt=body.prompt,
        lyrics=body.lyrics,
        duration_seconds=body.duration_seconds,
        seed=body.seed,
    )

    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.MUSIC_GENERATION,
        task_func=_run_music_generation,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        model_manager=model_manager,
        storage=storage,
        params=params,
        save_url=body.save_url,
        is_mock=settings.mock_mode,
    )

    if not submitted:
        raise HTTPException(status_code=429, detail="System busy")

    return AsyncJobResponse(
        job_id=body.job_id,
        status_url=str(request.url_for("get_job_status", job_id=body.job_id)),
    )


async def _run_music_generation(
    model_manager: ModelManagerDep,
    storage: StorageDep,
    params: MusicGenerationParams,
    save_url: str,
    is_mock: bool = False,
) -> JobResult:
    start_time = time.time()

    # Get generator at execution time (supports dynamic loading)
    generator = model_manager.get_music_generator()
    result = await generator.generate_music(params)

    final_url = await storage.upload_to_url(
        data=result.audio_data,
        url=save_url,
        content_type="audio/wav",
    )

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata={
            "seed": result.seed,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
        }
    )
```

### 8.2 Sound Effect Router

Create `app/routers/sound_effect_generation.py`:

```python
"""Sound effect generation endpoint."""

import logging
import time

from fastapi import APIRouter, Request, HTTPException

from app.dependencies import APIKeyDep, StorageDep, JobManagerDep, ModelManagerDep, SettingsDep
from app.models.common import ErrorResponse
from app.models.sound_effect_generation import SoundEffectGenerateRequest
from app.models.job import AsyncJobResponse, JobResult
from app.models.internal import SoundEffectParams
from app.services.model_manager import JobType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/sfx",
    tags=["Sound Effect Generation"],
)


@router.post(
    "/generate",
    response_model=AsyncJobResponse,
    status_code=202,
    summary="Generate Sound Effect",
    description="Generate a sound effect from a text description using AudioGen.",
)
async def generate_sound_effect(
    request: Request,
    body: SoundEffectGenerateRequest,
    api_key: APIKeyDep,
    storage: StorageDep,
    job_manager: JobManagerDep,
    model_manager: ModelManagerDep,
    settings: SettingsDep,
) -> AsyncJobResponse:
    params = SoundEffectParams(
        job_id=body.job_id,
        prompt=body.prompt,
        duration_seconds=body.duration_seconds,
        seed=body.seed,
    )

    submitted = await job_manager.try_submit_job(
        job_id=body.job_id,
        job_type=JobType.SOUND_EFFECT_GENERATION,
        task_func=_run_sound_effect_generation,
        webhook_url=body.webhook_url,
        item_id=body.item_id,
        webhook_secret=body.webhook_secret,
        model_manager=model_manager,
        storage=storage,
        params=params,
        save_url=body.save_url,
        is_mock=settings.mock_mode,
    )

    if not submitted:
        raise HTTPException(status_code=429, detail="System busy")

    return AsyncJobResponse(
        job_id=body.job_id,
        status_url=str(request.url_for("get_job_status", job_id=body.job_id)),
    )


async def _run_sound_effect_generation(
    model_manager: ModelManagerDep,
    storage: StorageDep,
    params: SoundEffectParams,
    save_url: str,
    is_mock: bool = False,
) -> JobResult:
    start_time = time.time()

    generator = model_manager.get_sound_effect_generator()
    result = await generator.generate_sound_effect(params)

    final_url = await storage.upload_to_url(
        data=result.audio_data,
        url=save_url,
        content_type="audio/wav",
    )

    return JobResult(
        save_url=final_url,
        generation_time=round(time.time() - start_time, 2),
        metadata={
            "seed": result.seed,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
        }
    )
```

### 8.3 Register Routers in main.py

```python
from app.routers import music_generation, sound_effect_generation

app.include_router(music_generation.router)
app.include_router(sound_effect_generation.router)
```

---

## 9. Internal Data Models

Add to `app/models/internal.py`:

```python
# --- Audio Generation ---

@dataclass
class MusicGenerationParams:
    """Parameters for music generation (ACE-Step 1.5)."""
    job_id: str
    prompt: str  # Style/genre description
    lyrics: Optional[str]  # Optional lyrics for vocal generation
    duration_seconds: float
    seed: Optional[int]


@dataclass
class MusicGenerationResult:
    """Result of music generation."""
    audio_data: bytes
    duration_seconds: float
    sample_rate: int
    seed: int


@dataclass
class SoundEffectParams:
    """Parameters for sound effect generation (AudioGen)."""
    job_id: str
    prompt: str  # Sound description
    duration_seconds: float
    seed: Optional[int]


@dataclass
class SoundEffectResult:
    """Result of sound effect generation."""
    audio_data: bytes
    duration_seconds: float
    sample_rate: int
    seed: int
```

---

## 10. Docker Integration

### Dockerfile Additions

```dockerfile
# LightX2V - install from local vendored copy
COPY LightX2V /app/LightX2V
RUN pip install --no-cache-dir -e /app/LightX2V || echo "LightX2V installation skipped"

# sgl-kernel for FP8 quantized inference
RUN pip install --no-cache-dir sgl-kernel || echo "sgl-kernel installation skipped"

# AudioCraft (for AudioGen)
RUN pip install --no-cache-dir audiocraft || echo "audiocraft installation skipped"

# ACE-Step (if available as pip package)
# RUN pip install --no-cache-dir acestep || echo "acestep installation skipped"
```

### Environment Variables

```dockerfile
ENV DTYPE=BF16
ENV PROFILING_DEBUG_LEVEL=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace cache
ENV HF_HOME=/app/models
ENV HF_HUB_CACHE=/app/models/hub
```

---

## 11. Dependencies

### requirements.txt Additions

```
# Audio Generation
audiocraft>=1.0.0   # Facebook AudioGen (CC-BY-NC 4.0)
# acestep>=1.5.0    # ACE-Step music generation (MIT) - if pip installable
```

### LightX2V Dependencies

Key LightX2V dependencies (from `LightX2V/requirements.txt`):

- `torch>=2.9.0`
- `transformers>=4.40.0`
- `accelerate>=0.26.0`
- `safetensors>=0.4.0`
- `flash-attn` (optional, for flash attention)
- `sgl-kernel` (for FP8 inference)

---

## 12. Known Issues & Troubleshooting

### VAE OOM at 1080p

**Problem**: `torch.cuda.OutOfMemoryError` during VAE decode.

**Solution**: Enable VAE tiling BEFORE `create_generator()`:

```python
pipe.use_tiling_vae = True
```

### 'NoneType' object is not callable

**Problem**: Error in scheduler or post-processing.

**Causes**:

1. Missing `metrics_func` in profiling context
2. `return_result_tensor` not set

**Solution**: Ensure `input_info.return_result_tensor = True`

### Generator vs Tensor Result

**Problem**: `TypeError: 'generator' object has no attribute 'shape'`

**Solution**: Materialize the generator:

```python
if isinstance(video_result, types.GeneratorType):
    video_tensor = next(video_result)
```

### Sigma Schedule Inconsistency

**Problem**: Subsequent requests use wrong sigmas (46s vs 30s latency).

**Solution**: Preserve sigmas in `LTX2Scheduler.clear()` method.

### Task Selection (i2v vs i2av)

**Problem**: 2x slowdown with `i2av` task.

**Solution**: Use `pipe.task = "i2v"` for video-only.

### Profiling Overhead

**Problem**: 50% slowdown with debug profiling.

**Solution**: Set `PROFILING_DEBUG_LEVEL=0` in environment.

---

## Appendix: API Endpoint Summary

| Endpoint                 | Method | Description                         |
| ------------------------ | ------ | ----------------------------------- |
| `/api/v1/music/generate` | POST   | Generate music (ACE-Step 1.5)       |
| `/api/v1/sfx/generate`   | POST   | Generate sound effects (AudioGen)   |
| `/api/v1/ltx2/generate`  | POST   | Generate video (LTX-2 via LightX2V) |
| `/api/v1/ltx2/keyframe`  | POST   | Keyframe interpolation video        |
| `/api/v1/mode`           | PUT    | Switch VRAM mode                    |
| `/api/v1/jobs/{job_id}`  | GET    | Get job status                      |

---

_Document Version: 1.0_  
_Created: February 4, 2026_  
_For: Vid-Bolt GPU API Re-implementation_
