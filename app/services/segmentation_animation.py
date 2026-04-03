"""Segmentation Animation Pipeline — Animated effects from single images.

Generates MP4 video by rendering animated visual effects on a segmented image.
Each frame interpolates operation parameters using easing functions, then
passes through the existing EffectsPipeline for rendering.

This module is CPU-only — SAM 3 runs once for segmentation, then all frame
generation is pure PIL/numpy/OpenCV.

Usage:
    pipeline = AnimationPipeline(image, masks, boxes, fps=30, duration=3.0)
    mp4_bytes = pipeline.render(operations)
"""

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.utils.video_encoding import encode_mp4_h264

logger = logging.getLogger(__name__)


# =============================================================================
# Easing Functions
# =============================================================================

class EasingFunctions:
    """Standard easing functions (from easings.net).
    
    All functions map t ∈ [0, 1] → t' ∈ [0, 1] (approximately).
    Some overshooting easings (back, elastic) may briefly exceed [0, 1].
    """

    @staticmethod
    def linear(t: float) -> float:
        return t

    @staticmethod
    def ease_in(t: float) -> float:
        """Quadratic ease-in: slow start, fast end."""
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        """Quadratic ease-out: fast start, slow end."""
        return 1.0 - (1.0 - t) ** 2

    @staticmethod
    def ease_in_out(t: float) -> float:
        """Smoothstep: smooth start and end."""
        return 3.0 * t * t - 2.0 * t * t * t

    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in: more dramatic slow start."""
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out: more dramatic slow end."""
        return 1.0 - (1.0 - t) ** 3

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out: professional motion feel."""
        if t < 0.5:
            return 4.0 * t * t * t
        return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0

    @staticmethod
    def ease_out_back(t: float) -> float:
        """Overshoots then settles: bouncy, playful feel."""
        c1 = 1.70158
        c3 = c1 + 1.0
        return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2

    @staticmethod
    def ease_out_elastic(t: float) -> float:
        """Springy oscillation: high-energy reveal."""
        if t == 0.0 or t == 1.0:
            return t
        c4 = (2.0 * math.pi) / 3.0
        return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0

    @staticmethod
    def ease_out_bounce(t: float) -> float:
        """Ball-bounce effect: fun, attention-grabbing."""
        n1 = 7.5625
        d1 = 2.75
        if t < 1.0 / d1:
            return n1 * t * t
        elif t < 2.0 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375

    # Registry for lookup by name
    REGISTRY = {}

    @classmethod
    def get(cls, name: str):
        """Get easing function by name."""
        if not cls.REGISTRY:
            cls.REGISTRY = {
                "linear": cls.linear,
                "ease_in": cls.ease_in,
                "ease_out": cls.ease_out,
                "ease_in_out": cls.ease_in_out,
                "ease_in_cubic": cls.ease_in_cubic,
                "ease_out_cubic": cls.ease_out_cubic,
                "ease_in_out_cubic": cls.ease_in_out_cubic,
                "ease_out_back": cls.ease_out_back,
                "ease_out_elastic": cls.ease_out_elastic,
                "ease_out_bounce": cls.ease_out_bounce,
            }
        return cls.REGISTRY.get(name, cls.linear)


# =============================================================================
# Animation Interpolator
# =============================================================================

class AnimationInterpolator:
    """Interpolates operation parameters over time using easing functions.
    
    Supports 6 animation modes:
        - transition: Linear A→B interpolation
        - draw: Progressive contour tracing (sets 'progress' param)
        - pulse: Oscillating min→max→min with configurable cycles
        - reveal: Directional mask reveal (left/right/top/bottom/radial/clockwise)
        - loop: Continuous oscillation (start→end→start→end...)
        - stagger: Per-object delay offset
    """

    def interpolate(
        self,
        op: Dict[str, Any],
        t: float,
        total_duration: float,
        frame_idx: int,
        total_frames: int,
    ) -> Dict[str, Any]:
        """Interpolate operation parameters at normalized time t.
        
        Args:
            op: Operation dict with optional 'animation' key
            t: Global progress 0.0 → 1.0
            total_duration: Total animation/video duration in seconds
            frame_idx: Current frame index
            total_frames: Total number of frames
            
        Returns:
            New operation dict with interpolated parameters
        """
        anim = op.get("animation")
        if not anim:
            return op

        result = copy.deepcopy(op)
        del result["animation"]

        mode = anim.get("mode", "transition")
        easing_name = anim.get("easing", "ease_out")
        delay = anim.get("delay", 0.0)
        duration = anim.get("duration", total_duration)
        easing_fn = EasingFunctions.get(easing_name)

        # Calculate local progress within this animation's window
        current_time = t * total_duration
        if current_time < delay:
            # Before animation starts: use start values
            return self._apply_values(result, anim.get("start", {}))
        elif current_time > delay + duration:
            # After animation ends: use end values
            return self._apply_values(result, anim.get("end", {}))

        # Within the animation window
        local_t = (current_time - delay) / max(0.001, duration)
        local_t = max(0.0, min(1.0, local_t))

        if mode == "transition":
            eased_t = easing_fn(local_t)
            return self._interpolate_transition(result, anim, eased_t)
        elif mode == "draw":
            eased_t = easing_fn(local_t)
            return self._interpolate_draw(result, eased_t)
        elif mode == "pulse":
            cycles = anim.get("cycles", 1)
            return self._interpolate_pulse(result, anim, local_t, easing_fn, cycles)
        elif mode == "reveal":
            eased_t = easing_fn(local_t)
            direction = anim.get("direction", "left")
            return self._interpolate_reveal(result, eased_t, direction)
        elif mode == "loop":
            cycles = anim.get("cycles", 2)
            return self._interpolate_loop(result, anim, local_t, easing_fn, cycles)
        elif mode == "stagger":
            stagger_delay = anim.get("stagger_delay", 0.2)
            eased_t = easing_fn(local_t)
            return self._interpolate_stagger(result, anim, eased_t, stagger_delay, total_duration)
        else:
            logger.warning(f"Unknown animation mode: {mode}, using transition")
            eased_t = easing_fn(local_t)
            return self._interpolate_transition(result, anim, eased_t)

    def _apply_values(self, op: dict, values: dict) -> dict:
        """Apply a set of parameter values to an operation."""
        result = copy.deepcopy(op)
        for key, value in values.items():
            result[key] = value
        return result

    def _interpolate_transition(self, op: dict, anim: dict, eased_t: float) -> dict:
        """Linearly interpolate between start and end parameter values."""
        start_vals = anim.get("start", {})
        end_vals = anim.get("end", {})
        result = copy.deepcopy(op)

        all_keys = set(list(start_vals.keys()) + list(end_vals.keys()))
        for key in all_keys:
            start_v = start_vals.get(key, op.get(key))
            end_v = end_vals.get(key, op.get(key))
            if start_v is None or end_v is None:
                continue
            result[key] = self._lerp(start_v, end_v, eased_t)

        return result

    def _interpolate_draw(self, op: dict, eased_t: float) -> dict:
        """Set 'progress' parameter for progressive contour drawing."""
        result = copy.deepcopy(op)
        result["progress"] = eased_t
        return result

    def _interpolate_pulse(
        self, op: dict, anim: dict, local_t: float, easing_fn, cycles: int
    ) -> dict:
        """Oscillate between start and end values."""
        start_vals = anim.get("start", {})
        end_vals = anim.get("end", {})
        result = copy.deepcopy(op)

        # Create oscillating t: goes 0→1→0→1→0... over cycles
        cycle_t = (local_t * cycles) % 1.0
        # Triangle wave: 0→1→0 per cycle
        if cycle_t <= 0.5:
            wave_t = cycle_t * 2.0
        else:
            wave_t = (1.0 - cycle_t) * 2.0
        eased_wave = easing_fn(wave_t)

        all_keys = set(list(start_vals.keys()) + list(end_vals.keys()))
        for key in all_keys:
            start_v = start_vals.get(key, op.get(key))
            end_v = end_vals.get(key, op.get(key))
            if start_v is None or end_v is None:
                continue
            result[key] = self._lerp(start_v, end_v, eased_wave)

        return result

    def _interpolate_reveal(self, op: dict, eased_t: float, direction: str) -> dict:
        """Add reveal mask info for directional wipe animation."""
        result = copy.deepcopy(op)
        result["_reveal_progress"] = eased_t
        result["_reveal_direction"] = direction
        return result

    def _interpolate_loop(
        self, op: dict, anim: dict, local_t: float, easing_fn, cycles: int
    ) -> dict:
        """Continuously loop between start and end values."""
        start_vals = anim.get("start", {})
        end_vals = anim.get("end", {})
        result = copy.deepcopy(op)

        # Sawtooth: 0→1→0→1... over cycles
        cycle_t = (local_t * cycles) % 1.0
        eased_cycle = easing_fn(cycle_t)

        all_keys = set(list(start_vals.keys()) + list(end_vals.keys()))
        for key in all_keys:
            start_v = start_vals.get(key, op.get(key))
            end_v = end_vals.get(key, op.get(key))
            if start_v is None or end_v is None:
                continue
            result[key] = self._lerp(start_v, end_v, eased_cycle)

        return result

    def _interpolate_stagger(
        self, op: dict, anim: dict, eased_t: float, stagger_delay: float, total_duration: float
    ) -> dict:
        """Add stagger metadata for per-object delay animations."""
        result = self._interpolate_transition(op, anim, eased_t)
        result["_stagger_delay"] = stagger_delay
        result["_stagger_total_duration"] = total_duration
        return result

    def _lerp(self, start, end, t: float):
        """Linearly interpolate between two values."""
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            result = start + (end - start) * t
            return int(result) if isinstance(start, int) and isinstance(end, int) else result
        elif isinstance(start, list) and isinstance(end, list):
            return [self._lerp(s, e, t) for s, e in zip(start, end)]
        else:
            # Non-interpolatable: snap at midpoint
            return end if t >= 0.5 else start


# =============================================================================
# Animation Pipeline
# =============================================================================

class AnimationPipeline:
    """Generates MP4 video from a single image + segmentation masks + animated operations.
    
    Architecture:
        1. For each frame (0..N), calculate normalized time t ∈ [0, 1]
        2. Interpolate all operation parameters at time t
        3. Apply camera operations (zoom/pan) via crop + scale
        4. Pass interpolated operations to EffectsPipeline for rendering
        5. Collect frames and encode to MP4 via OpenCV
    
    Args:
        image: Source PIL Image
        masks: List of binary masks from SAM 3 segmentation
        boxes: Bounding boxes per detected object
        fps: Frames per second (default: 30)
        duration: Total animation duration in seconds (default: 3.0)
    """

    MAX_FRAMES = 600  # 10s @ 60fps absolute cap

    def __init__(
        self,
        image: Image.Image,
        masks: List[np.ndarray],
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        labels: Optional[List[str]] = None,
        object_ids: Optional[List[int]] = None,
        fps: int = 30,
        duration: float = 3.0,
    ):
        self.source_image = image.convert("RGBA")
        self.width, self.height = self.source_image.size
        self.masks = masks
        self.boxes = boxes or []
        self.labels = labels
        self.object_ids = object_ids
        self.fps = fps
        self.duration = duration
        self.total_frames = min(int(fps * duration), self.MAX_FRAMES)
        self.interpolator = AnimationInterpolator()

        # Pre-compute mask centroid for camera targeting
        combined = np.zeros((self.height, self.width), dtype=bool)
        for m in masks:
            m_np = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m) if not isinstance(m, np.ndarray) else m
            while m_np.ndim > 2:
                m_np = m_np.squeeze(0)
            if m_np.shape == (self.height, self.width):
                combined |= (m_np > 0.5)

        if np.any(combined):
            ys, xs = np.where(combined)
            self.mask_center = (float(xs.mean()), float(ys.mean()))
        else:
            self.mask_center = (self.width / 2.0, self.height / 2.0)

        logger.info(
            f"AnimationPipeline initialized: {self.width}x{self.height}, "
            f"{self.total_frames} frames ({self.duration}s @ {self.fps}fps)"
        )

    def render(self, operations: List[Dict[str, Any]]) -> bytes:
        """Render all frames and encode to MP4.
        
        Args:
            operations: List of operation dicts, each with optional 'animation' config
            
        Returns:
            MP4 video as bytes
        """
        frames: List[np.ndarray] = []
        animated_ops = [
            op.get("type", "unknown")
            for op in operations
            if isinstance(op, dict) and op.get("animation")
        ]
        if not animated_ops:
            logger.warning(
                "Animation request contains no animation configs; output will be a static processed image repeated across frames",
                extra={
                    "operation_types": [
                        op.get("type", "unknown") for op in operations if isinstance(op, dict)
                    ],
                    "frame_count": self.total_frames,
                },
            )
        else:
            logger.info(
                "Animation request contains animated operations",
                extra={"animated_operation_types": animated_ops, "frame_count": self.total_frames},
            )

        for frame_idx in range(self.total_frames):
            t = frame_idx / max(1, self.total_frames - 1)

            # Interpolate all operations at time t
            frame_ops = []
            camera_ops = []
            for op in operations:
                interp_op = self.interpolator.interpolate(
                    op, t, self.duration, frame_idx, self.total_frames
                )
                if interp_op.get("type") in ("zoom", "pan"):
                    camera_ops.append(interp_op)
                else:
                    frame_ops.append(interp_op)

            # Start with a fresh copy of the source image
            working_image = self.source_image.copy()

            # Apply camera transformations (zoom/pan via crop + scale)
            working_image = self._apply_camera(working_image, camera_ops)

            # Apply visual effects using the existing EffectsPipeline
            from app.services.segmentation_effects import EffectsPipeline
            pipeline = EffectsPipeline(
                working_image,
                self.masks,
                self.boxes,
                labels=self.labels,
                object_ids=self.object_ids,
            )
            pipeline.apply(frame_ops)

            # Convert to RGB numpy array for video encoding
            frame_rgb = np.array(pipeline.image.convert("RGB"))
            frames.append(frame_rgb)

            if (frame_idx + 1) % 30 == 0 or frame_idx == self.total_frames - 1:
                logger.debug(f"Rendered frame {frame_idx + 1}/{self.total_frames}")

        logger.info(f"All {len(frames)} frames rendered, encoding MP4...")
        return self._encode_mp4(frames)

    def _apply_camera(self, image: Image.Image, camera_ops: List[dict]) -> Image.Image:
        """Apply zoom and pan by cropping and scaling the canvas.
        
        Zoom: crop a smaller region centered on the mask centroid, then scale up.
        Pan: offset the crop region.
        """
        w, h = image.size
        # Start with the full canvas
        crop_cx, crop_cy = w / 2.0, h / 2.0
        scale = 1.0
        pan_x, pan_y = 0.0, 0.0

        for op in camera_ops:
            if op["type"] == "zoom":
                scale = max(1.0, min(4.0, op.get("scale", 1.0)))
                target = op.get("target", "mask")
                if target == "mask":
                    crop_cx, crop_cy = self.mask_center
                elif target == "center":
                    crop_cx, crop_cy = w / 2.0, h / 2.0
                elif isinstance(target, list) and len(target) == 2:
                    crop_cx, crop_cy = float(target[0]), float(target[1])

            elif op["type"] == "pan":
                offset = op.get("offset", [0, 0])
                if isinstance(offset, list) and len(offset) == 2:
                    pan_x, pan_y = float(offset[0]), float(offset[1])

        if scale <= 1.0 and pan_x == 0 and pan_y == 0:
            return image

        # Calculate crop region
        crop_w = w / scale
        crop_h = h / scale
        cx = max(crop_w / 2, min(w - crop_w / 2, crop_cx + pan_x))
        cy = max(crop_h / 2, min(h - crop_h / 2, crop_cy + pan_y))

        left = int(cx - crop_w / 2)
        top = int(cy - crop_h / 2)
        right = int(cx + crop_w / 2)
        bottom = int(cy + crop_h / 2)

        # Clamp
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.LANCZOS)

    def _encode_mp4(self, frames: List[np.ndarray]) -> bytes:
        """Encode a list of RGB numpy frames to H.264 MP4 bytes."""
        if not frames:
            raise ValueError("No frames to encode")

        mp4_bytes, encode_info = encode_mp4_h264(frames, self.fps)
        logger.info(
            f"MP4 encoded: {encode_info['frame_count']} frames, {encode_info['width']}x{encode_info['height']}, "
            f"{len(mp4_bytes)} bytes ({len(mp4_bytes) / 1024 / 1024:.1f} MB)",
            extra={
                "codec": encode_info["codec"],
                "pixel_format": encode_info["pixel_format"],
                "movflags": encode_info["movflags"],
            },
        )
        return mp4_bytes


def interpolate_video_operations(
    operations: List[Dict[str, Any]],
    frame_idx: int,
    total_frames: int,
    total_duration: float,
) -> List[Dict[str, Any]]:
    """Interpolate operations for video-on-video temporal animation.
    
    Called from sam3_generator's video segmentation path to support
    animated effects on actual video input (not just static per-frame).
    
    Args:
        operations: List of operations, some with 'animation' configs
        frame_idx: Current frame index
        total_frames: Total video frame count
        total_duration: Total video duration in seconds
        
    Returns:
        List of operations with interpolated parameters for this frame
    """
    interpolator = AnimationInterpolator()
    t = frame_idx / max(1, total_frames - 1)

    result = []
    for op in operations:
        interp_op = interpolator.interpolate(op, t, total_duration, frame_idx, total_frames)
        result.append(interp_op)
    return result
