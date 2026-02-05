"""Interfaces for model generation services.

This module defines the abstract base classes that all model generators must implement.
This ensures a consistent API for the ModelManager and facilitates easy addition of new models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.models.internal import (
    ImageEditParams,
    ImageEditResult,
    ImageGenerationParams,
    ImageGenerationResult,
    KeyframeInterpolationParams,
    MusicGenerationParams,
    MusicGenerationResult,
    SoundEffectParams,
    SoundEffectResult,
    UpscaleParams,
    UpscaleResult,
    VideoGenerationParams,
    VideoGenerationResult,
)


class BaseModelGenerator(ABC):
    """Abstract base class for all model generators."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def load_models(self) -> None:
        """Load model components."""
        pass

    @abstractmethod
    def unload_models(self) -> None:
        """Unload models and free resources."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the generator."""
        pass

    @property
    @abstractmethod
    def _loaded(self) -> bool:
        """Check if models are loaded.
        
        Using _loaded naming to match existing property conventions in ModelManager.
        """
        pass


class ImageGenerator(BaseModelGenerator):
    """Interface for text-to-image generators."""

    @abstractmethod
    async def generate_image(self, params: ImageGenerationParams) -> ImageGenerationResult:
        """Generate an image from a text prompt."""
        pass
    
    async def generate_batch(
        self, params_list: List[ImageGenerationParams]
    ) -> List[ImageGenerationResult]:
        """Generate multiple images in a single batch.
        
        Default implementation runs serially. Subclasses can override for
        true batching (single forward pass).
        
        Args:
            params_list: List of generation parameters (must have same dimensions)
            
        Returns:
            List of generation results in same order as inputs
        """
        results = []
        for params in params_list:
            result = await self.generate_image(params)
            results.append(result)
        return results
    
    @abstractmethod
    async def load_lora(self, lora_name: str, weight: float = 1.0) -> None:
        """Load a LoRA adapter."""
        pass

    @abstractmethod
    async def unload_lora(self) -> None:
        """Unload the current LoRA adapter."""
        pass


class ImageEditor(BaseModelGenerator):
    """Interface for image editing generators."""

    @abstractmethod
    async def edit_image(self, params: ImageEditParams) -> ImageEditResult:
        """Edit an existing image based on prompt and mask."""
        pass


class VideoGenerator(BaseModelGenerator):
    """Interface for video generators."""

    @abstractmethod
    async def generate_video(self, params: VideoGenerationParams) -> VideoGenerationResult:
        """Generate a video from a start frame (I2V)."""
        pass

    @abstractmethod
    async def generate_keyframe_video(
        self, params: KeyframeInterpolationParams
    ) -> VideoGenerationResult:
        """Generate a video by interpolating between keyframes."""
        pass


class Upscaler(BaseModelGenerator):
    """Interface for video upscalers."""

    @abstractmethod
    async def upscale_video(self, params: UpscaleParams) -> UpscaleResult:
        """Upscale a video."""
        pass


class MusicGenerator(BaseModelGenerator):
    """Interface for music generators."""

    @abstractmethod
    async def generate_music(self, params: MusicGenerationParams) -> MusicGenerationResult:
        """Generate music from a text prompt and optional lyrics."""
        pass


class SoundEffectGenerator(BaseModelGenerator):
    """Interface for sound effect generators."""

    @abstractmethod
    async def generate_sound_effect(self, params: SoundEffectParams) -> SoundEffectResult:
        """Generate a sound effect from a text description."""
        pass
