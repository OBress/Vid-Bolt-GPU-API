"""Segmentation Effects Pipeline - Composable post-processing for SAM 3 masks.

Applies ordered visual operations to images using SAM 3 segmentation masks.
All operations are CPU-based (PIL/numpy/OpenCV) — no GPU required.

Usage:
    pipeline = EffectsPipeline(image, masks)
    result = pipeline.apply(operations)
    result_bytes = pipeline.to_bytes(format="png")
"""

import copy
import io
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

logger = logging.getLogger(__name__)


class EffectsPipeline:
    """Applies composable visual operations using segmentation masks.
    
    The pipeline maintains:
        - `image`: the working image (RGBA numpy array)
        - `masks`: list of binary masks from SAM 3 (one per object)
        - `active_mask`: combined mask for the current selection target
        - `boxes`: bounding boxes per object (for text labels)
    
    Operations modify the image in-place through the active mask.
    """

    def __init__(
        self,
        image: Image.Image,
        masks: List[np.ndarray],
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        labels: Optional[List[str]] = None,
        object_ids: Optional[List[int]] = None,
        annotation_state: Optional[Dict[str, Any]] = None,
    ):
        """Initialize pipeline with source image and segmentation masks.
        
        Args:
            image: Source PIL image
            masks: List of binary masks (H, W), one per detected object
            boxes: Optional bounding boxes per object [(x1,y1,x2,y2)]
            labels: Optional string labels per object (from object_prompts)
            object_ids: Optional stable object IDs for tracked video objects
        """
        # Convert to RGBA for alpha support
        self.original = image.convert("RGBA")
        self.image = self.original.copy()
        self.width, self.height = self.image.size

        # Store per-object masks as boolean arrays
        self.object_masks: List[np.ndarray] = []
        for mask in masks:
            if hasattr(mask, 'cpu'):
                m = mask.cpu().numpy()
            elif not isinstance(mask, np.ndarray):
                m = np.array(mask)
            else:
                m = mask
            while m.ndim > 2:
                m = m.squeeze(0)
            self.object_masks.append(m > 0.5)

        # Combined mask of all detected objects
        self.combined_mask = np.zeros((self.height, self.width), dtype=bool)
        for m in self.object_masks:
            if m.shape == (self.height, self.width):
                self.combined_mask |= m

        # Active selection: "mask" = objects, "background" = inverse, "all" = everything
        self.active_target = "mask"
        self.active_mask = self.combined_mask.copy()

        # Store boxes for bounding box operations
        provided_boxes = list(boxes or [])
        self.boxes: List[Tuple[int, int, int, int]] = []
        for idx, mask in enumerate(self.object_masks):
            if idx < len(provided_boxes):
                self.boxes.append(tuple(int(v) for v in provided_boxes[idx]))
            else:
                self.boxes.append(self._box_from_mask(mask))

        # Store labels for per-object targeting via object_label
        self.object_labels: Optional[List[str]] = labels
        self._label_to_indices: Dict[str, List[int]] = {}
        if labels:
            for i, label in enumerate(labels):
                if label not in self._label_to_indices:
                    self._label_to_indices[label] = []
                self._label_to_indices[label].append(i)

        # Store stable IDs for object-aware video routing via object_id
        self.object_ids: Optional[List[int]] = object_ids
        self._id_to_indices: Dict[int, List[int]] = {}
        if object_ids:
            for i, object_id in enumerate(object_ids):
                self._id_to_indices.setdefault(int(object_id), []).append(i)

        self.active_object_indices: List[int] = list(range(len(self.object_masks)))
        self.annotation_state: Dict[str, Any] = annotation_state if annotation_state is not None else {}
        self.warnings: List[Dict[str, Any]] = []
        self._frame_label_rects: List[Tuple[int, int, int, int]] = []

    def apply(self, operations: List[Dict[str, Any]]) -> Image.Image:
        """Apply a list of operations sequentially.
        
        Args:
            operations: List of operation dicts, each with "type" and params
            
        Returns:
            Processed PIL Image
        """
        for i, op in enumerate(operations):
            op_type = op.get("type", "")
            logger.debug(f"Applying operation {i+1}/{len(operations)}: {op_type}")

            try:
                if op_type == "select":
                    self._op_select_v2(op)
                else:
                    self._apply_effect_operation(op)
            except Exception as e:
                logger.error(f"Error applying operation {op_type}: {e}")
                raise

        return self.image

    def _apply_effect_operation(self, op: Dict[str, Any]) -> None:
        """Dispatch effect operations, including animation-aware wrappers."""
        mode = op.get("_animation_mode")
        op_type = op.get("type", "")

        if mode == "stagger":
            self._apply_staggered_operation(op)
            return
        if op_type == "label":
            self._op_label(op)
            return
        if mode in {"reveal", "splash"}:
            self._apply_clipped_animation_operation(op)
            return

        self._dispatch_operation(op)

    def _dispatch_operation(self, op: Dict[str, Any]) -> None:
        """Dispatch a single non-select operation."""
        op_type = op.get("type", "")
        if op_type == "blur":
            self._op_blur(op)
        elif op_type == "pixelate":
            self._op_pixelate(op)
        elif op_type == "redact":
            self._op_redact(op)
        elif op_type == "color_overlay":
            self._op_color_overlay(op)
        elif op_type == "color_grade":
            self._op_color_grade(op)
        elif op_type == "opacity":
            self._op_opacity(op)
        elif op_type == "replace_color":
            self._op_replace_color(op)
        elif op_type == "remove_background":
            self._op_remove_background_v2(op)
        elif op_type == "replace_background":
            self._op_replace_background_v2(op)
        elif op_type == "greenscreen":
            self._op_greenscreen_v2(op)
        elif op_type == "outline":
            self._op_outline_v2(op)
        elif op_type == "text_label":
            logger.warning("text_label is deprecated and no longer supported; skipping")
        elif op_type == "bounding_box":
            self._op_bounding_box_v2(op)
        elif op_type == "spotlight":
            self._op_spotlight_v2(op)
        elif op_type == "bokeh":
            self._op_bokeh_v2(op)
        elif op_type == "glow":
            self._op_glow_v2(op)
        elif op_type == "shadow":
            self._op_shadow_v2(op)
        elif op_type == "vignette":
            self._op_vignette_v2(op)
        elif op_type == "grayscale":
            self._op_grayscale(op)
        elif op_type == "invert":
            self._op_invert(op)
        elif op_type == "sharpen":
            self._op_sharpen(op)
        elif op_type == "sepia":
            self._op_sepia(op)
        elif op_type == "posterize":
            self._op_posterize(op)
        elif op_type == "edge_detect":
            self._op_edge_detect(op)
        elif op_type == "emboss":
            self._op_emboss(op)
        elif op_type == "noise":
            self._op_noise(op)
        elif op_type == "sketch":
            self._op_sketch(op)
        elif op_type == "duotone":
            self._op_duotone(op)
        elif op_type == "halftone":
            self._op_halftone(op)
        elif op_type == "glitch":
            self._op_glitch(op)
        elif op_type == "motion_blur":
            self._op_motion_blur(op)
        elif op_type == "glass":
            self._op_glass(op)
        elif op_type == "feather":
            self._op_feather(op)
        elif op_type in {"zoom", "pan"}:
            return
        else:
            logger.warning(f"Unknown operation type: {op_type}, skipping")

    def to_bytes(self, format: str = "png") -> bytes:
        """Convert processed image to bytes.
        
        Args:
            format: Output format ("png" for RGBA support, "jpeg" for smaller size)
        """
        buf = io.BytesIO()
        if format.lower() == "jpeg":
            # JPEG doesn't support alpha, convert to RGB
            self.image.convert("RGB").save(buf, format="JPEG", quality=95)
        else:
            self.image.save(buf, format="PNG")
        return buf.getvalue()

    # =========================================================================
    # Selection
    # =========================================================================

    def _op_select(self, op: dict):
        """Switch which region subsequent operations apply to.
        
        Supports:
          - 'object_index' (int, 0-based): target a single object by index
          - 'object_label' (str): target all objects with this label
          - 'object_labels' (list[str]): target all objects matching any of these labels (union)
          - No index/label: target all objects (default)
        
        When using 'background' with a label/index, it inverts that specific mask
        (NOT the combined mask), so other objects may be included in the background.
        For background blur that excludes ALL objects, use 'background' without any label.
        """
        target = op.get("target", "mask")
        object_index = op.get("object_index", None)
        object_label = op.get("object_label", None)
        object_labels = op.get("object_labels", None)
        self.active_target = target

        # Resolve which mask indices to use
        selected_indices = None

        if object_label is not None and self._label_to_indices:
            # Single label → find all objects with this label
            selected_indices = self._label_to_indices.get(object_label, [])
            if not selected_indices:
                logger.warning(f"object_label '{object_label}' not found. Available: {list(self._label_to_indices.keys())}")
        elif object_labels is not None and self._label_to_indices:
            # Multiple labels → union of all matching objects
            selected_indices = []
            for lbl in object_labels:
                selected_indices.extend(self._label_to_indices.get(lbl, []))
            if not selected_indices:
                logger.warning(f"No objects found for labels {object_labels}. Available: {list(self._label_to_indices.keys())}")
        elif object_index is not None and isinstance(object_index, int):
            # Single index
            if 0 <= object_index < len(self.object_masks):
                selected_indices = [object_index]
            else:
                logger.warning(f"object_index {object_index} out of range (have {len(self.object_masks)} objects)")

        if selected_indices is not None and selected_indices:
            # Build a combined mask from selected objects
            combined = np.zeros((self.height, self.width), dtype=bool)
            for idx in selected_indices:
                if 0 <= idx < len(self.object_masks):
                    combined |= self.object_masks[idx]
            if target == "mask":
                self.active_mask = combined
            elif target == "background":
                self.active_mask = ~combined
            else:
                self.active_mask = np.ones((self.height, self.width), dtype=bool)
        else:
            # Default: all objects
            if target == "mask":
                self.active_mask = self.combined_mask.copy()
            elif target == "background":
                self.active_mask = ~self.combined_mask
            elif target == "all":
                self.active_mask = np.ones((self.height, self.width), dtype=bool)
            else:
                logger.warning(f"Unknown select target: {target}, defaulting to 'mask'")
                self.active_mask = self.combined_mask.copy()

    def _op_select_v2(self, op: dict):
        """Switch which region subsequent operations apply to with stable object routing."""
        target = op.get("target", "mask")
        object_id = op.get("object_id", None)
        object_ids = op.get("object_ids", None)
        object_index = op.get("object_index", None)
        object_label = op.get("object_label", None)
        object_labels = op.get("object_labels", None)
        self.active_target = target

        selected_indices = None
        explicit_selection = any(
            value is not None
            for value in (object_id, object_ids, object_index, object_label, object_labels)
        )

        if object_id is not None and self._id_to_indices:
            selected_indices = self._id_to_indices.get(int(object_id), [])
            if not selected_indices:
                logger.warning(f"object_id {object_id} not found. Available: {sorted(self._id_to_indices.keys())}")
        elif object_ids is not None and self._id_to_indices:
            selected_indices = []
            for obj_id in object_ids:
                selected_indices.extend(self._id_to_indices.get(int(obj_id), []))
            if not selected_indices:
                logger.warning(f"No objects found for object_ids {object_ids}. Available: {sorted(self._id_to_indices.keys())}")
        elif object_label is not None and self._label_to_indices:
            selected_indices = self._label_to_indices.get(object_label, [])
            if not selected_indices:
                logger.warning(f"object_label '{object_label}' not found. Available: {list(self._label_to_indices.keys())}")
        elif object_labels is not None and self._label_to_indices:
            selected_indices = []
            for lbl in object_labels:
                selected_indices.extend(self._label_to_indices.get(lbl, []))
            if not selected_indices:
                logger.warning(f"No objects found for labels {object_labels}. Available: {list(self._label_to_indices.keys())}")
        elif object_index is not None and isinstance(object_index, int):
            if 0 <= object_index < len(self.object_masks):
                selected_indices = [object_index]
            else:
                logger.warning(f"object_index {object_index} out of range (have {len(self.object_masks)} objects)")

        if selected_indices is not None:
            selected_indices = list(dict.fromkeys(selected_indices))

        if selected_indices is not None and selected_indices:
            combined = np.zeros((self.height, self.width), dtype=bool)
            for idx in selected_indices:
                if 0 <= idx < len(self.object_masks):
                    combined |= self.object_masks[idx]
            self.active_object_indices = selected_indices
            if target == "mask":
                self.active_mask = combined
            elif target == "background":
                self.active_mask = ~combined
            else:
                self.active_mask = np.ones((self.height, self.width), dtype=bool)
        elif explicit_selection:
            self.active_object_indices = []
            if target == "mask":
                self.active_mask = np.zeros((self.height, self.width), dtype=bool)
            else:
                self.active_mask = np.ones((self.height, self.width), dtype=bool)
        else:
            self.active_object_indices = list(range(len(self.object_masks)))
            if target == "mask":
                self.active_mask = self.combined_mask.copy()
            elif target == "background":
                self.active_mask = ~self.combined_mask
            elif target == "all":
                self.active_mask = np.ones((self.height, self.width), dtype=bool)
            else:
                logger.warning(f"Unknown select target: {target}, defaulting to 'mask'")
                self.active_mask = self.combined_mask.copy()

    def _selected_subject_mask(self) -> np.ndarray:
        """Return the currently selected subject mask for object-centric effects."""
        if self.active_target == "mask":
            return self.active_mask.copy()
        if self.active_target == "background":
            return ~self.active_mask
        return self.combined_mask.copy()

    def _iter_selected_object_indices(self) -> List[int]:
        """Return the object indices currently selected for object-wise effects."""
        return list(self.active_object_indices)

    @staticmethod
    def _box_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Derive a tight bounding box from a mask."""
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _object_centroid(self, idx: int) -> Tuple[float, float]:
        """Return centroid coordinates for a selected object."""
        if not (0 <= idx < len(self.object_masks)):
            return (0.0, 0.0)
        ys, xs = np.where(self.object_masks[idx])
        if len(xs) and len(ys):
            return (float(xs.mean()), float(ys.mean()))
        x1, y1, x2, y2 = self.boxes[idx] if idx < len(self.boxes) else (0, 0, 0, 0)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _sorted_object_indices(self, indices: List[int]) -> List[int]:
        """Sort objects in stable screen-space order for staggered rendering."""
        return sorted(
            indices,
            key=lambda idx: (
                round(self._object_centroid(idx)[0], 4),
                round(self._object_centroid(idx)[1], 4),
                idx,
            ),
        )

    def _apply_clipped_animation_operation(self, op: Dict[str, Any]) -> None:
        """Apply reveal/splash animations by clipping the active region per frame."""
        base_mask = self.active_mask.copy()
        progress = max(0.0, min(1.0, float(op.get("_animation_progress", 1.0))))
        if progress <= 0.0:
            return

        clip_mask = self._build_animation_clip_mask(base_mask, op)
        if not np.any(clip_mask):
            return

        original_mask = self.active_mask
        try:
            self.active_mask = base_mask & clip_mask
            clipped_op = copy.deepcopy(op)
            clipped_op["_clip_mask"] = clip_mask
            self._dispatch_operation(clipped_op)
        finally:
            self.active_mask = original_mask

    def _build_animation_clip_mask(self, base_mask: np.ndarray, op: Dict[str, Any]) -> np.ndarray:
        """Build a clip mask for reveal and splash animations."""
        mode = op.get("_animation_mode")
        progress = max(0.0, min(1.0, float(op.get("_animation_progress", 1.0))))
        if mode == "reveal":
            return self._build_reveal_mask(base_mask, progress, str(op.get("_animation_direction", "left")))
        if mode == "splash":
            return self._build_splash_mask(base_mask, progress, int(op.get("_animation_seed", 0)))
        return base_mask.copy()

    def _build_reveal_mask(
        self,
        base_mask: np.ndarray,
        progress: float,
        direction: str,
    ) -> np.ndarray:
        """Create a directional wipe mask over the selected region."""
        if progress >= 1.0:
            return base_mask.copy()
        if not np.any(base_mask):
            return np.zeros_like(base_mask, dtype=bool)

        ys, xs = np.where(base_mask)
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        width = max(1, max_x - min_x + 1)
        height = max(1, max_y - min_y + 1)

        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        if direction == "right":
            threshold = max_x - progress * width
            clip = x_coords >= threshold
        elif direction == "top":
            threshold = min_y + progress * height
            clip = y_coords <= threshold
        elif direction == "bottom":
            threshold = max_y - progress * height
            clip = y_coords >= threshold
        elif direction == "radial":
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            max_dist = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)
            clip = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) <= (progress * max_dist)
        elif direction == "clockwise":
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            angles = (np.arctan2(y_coords - cy, x_coords - cx) + math.pi / 2.0) % (2.0 * math.pi)
            clip = angles <= (progress * 2.0 * math.pi)
        else:
            threshold = min_x + progress * width
            clip = x_coords <= threshold

        return clip & base_mask

    def _build_splash_mask(self, base_mask: np.ndarray, progress: float, seed: int) -> np.ndarray:
        """Create an expanding organic fill mask over the selected region."""
        if progress >= 1.0:
            return base_mask.copy()
        if progress <= 0.0 or not np.any(base_mask):
            return np.zeros_like(base_mask, dtype=bool)

        ys, xs = np.where(base_mask)
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        width = max(1, max_x - min_x + 1)
        height = max(1, max_y - min_y + 1)
        max_radius = math.sqrt(width ** 2 + height ** 2)

        rng = np.random.RandomState(seed)
        sample_count = min(12, max(4, len(xs) // max(1, len(xs) // 8 + 1)))
        sample_indices = rng.choice(len(xs), size=sample_count, replace=len(xs) < sample_count)
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]

        splash = np.zeros_like(base_mask, dtype=np.uint8)
        for sample_index in np.atleast_1d(sample_indices):
            cx = float(xs[int(sample_index)])
            cy = float(ys[int(sample_index)])
            radius_x = max(4.0, progress * max_radius * rng.uniform(0.18, 0.32))
            radius_y = max(4.0, progress * max_radius * rng.uniform(0.18, 0.32))
            local = (((x_coords - cx) / radius_x) ** 2 + ((y_coords - cy) / radius_y) ** 2) <= 1.0
            splash[local & base_mask] = 255

        splash_img = Image.fromarray(splash).filter(
            ImageFilter.GaussianBlur(radius=max(1.0, progress * 18.0))
        )
        splash_arr = np.array(splash_img)
        return (splash_arr > 32) & base_mask

    def _apply_staggered_operation(self, op: Dict[str, Any]) -> None:
        """Apply an operation one selected object at a time with offset timing."""
        ordered_indices = self._sorted_object_indices(self._iter_selected_object_indices())
        if not ordered_indices:
            return

        original_target = self.active_target
        original_mask = self.active_mask.copy()
        original_indices = list(self.active_object_indices)

        try:
            for rank, object_index in enumerate(ordered_indices):
                staged_op = self._resolve_staggered_operation(op, rank)
                if staged_op is None:
                    continue
                self.active_target = "mask"
                self.active_object_indices = [object_index]
                self.active_mask = self.object_masks[object_index].copy()
                self._apply_effect_operation(staged_op)
        finally:
            self.active_target = original_target
            self.active_mask = original_mask
            self.active_object_indices = original_indices

    def _resolve_staggered_operation(self, op: Dict[str, Any], rank: int) -> Optional[Dict[str, Any]]:
        """Resolve the per-object local values for a staggered operation."""
        current_time = float(op.get("_animation_current_time", 0.0))
        duration = max(0.001, float(op.get("_animation_duration", op.get("_animation_total_duration", 1.0))))
        stagger_delay = float(op.get("_stagger_delay", 0.2))
        start_vals = copy.deepcopy(op.get("_animation_start_values", {}))
        end_vals = copy.deepcopy(op.get("_animation_end_values", {}))
        delay = stagger_delay * rank

        if current_time < delay and not start_vals:
            return None

        if current_time <= delay:
            eased_t = 0.0
        elif current_time >= delay + duration:
            eased_t = 1.0
        else:
            from app.services.segmentation_animation import EasingFunctions

            local_t = (current_time - delay) / duration
            easing_name = str(op.get("_animation_easing", "ease_out"))
            eased_t = EasingFunctions.get(easing_name)(local_t)

        staged_op = {
            key: copy.deepcopy(value)
            for key, value in op.items()
            if not key.startswith("_animation_") and not key.startswith("_stagger_")
        }
        all_keys = set(start_vals.keys()) | set(end_vals.keys())
        for key in all_keys:
            start_value = start_vals.get(key, staged_op.get(key))
            end_value = end_vals.get(key, staged_op.get(key))
            if start_value is None or end_value is None:
                continue
            staged_op[key] = self._lerp_value(start_value, end_value, eased_t)
        return staged_op

    def _lerp_value(self, start: Any, end: Any, t: float) -> Any:
        """Interpolate numbers and numeric lists."""
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            value = start + (end - start) * t
            return int(value) if isinstance(start, int) and isinstance(end, int) else value
        if isinstance(start, list) and isinstance(end, list):
            return [self._lerp_value(s, e, t) for s, e in zip(start, end)]
        return end if t >= 0.5 else start

    def _record_warning(self, warning: Dict[str, Any]) -> None:
        """Record a warning once per annotation state / pipeline lifetime."""
        warning_keys = self.annotation_state.setdefault("_warning_keys", set())
        warning_key = tuple(sorted((key, repr(value)) for key, value in warning.items()))
        if warning_key in warning_keys:
            return
        warning_keys.add(warning_key)
        self.warnings.append(warning)

    @staticmethod
    def _parse_color(value: Any, default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Normalize a color payload to RGBA."""
        if value is None:
            return default
        if not isinstance(value, (list, tuple)):
            return default
        try:
            color = tuple(int(max(0, min(255, channel))) for channel in value)
        except Exception:
            return default
        if len(color) == 3:
            return color + (255,)
        if len(color) == 4:
            return color
        return default

    @staticmethod
    def _parse_offset(value: Any) -> Tuple[int, int]:
        """Normalize label offset values."""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return (int(value[0]), int(value[1]))
            except Exception:
                return (0, 0)
        return (0, 0)

    @staticmethod
    def _clamp_rect(
        rect: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """Clamp a rectangle into the frame while preserving size."""
        x1, y1, x2, y2 = rect
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        x1 = min(max(0, x1), max(0, frame_width - width))
        y1 = min(max(0, y1), max(0, frame_height - height))
        return (x1, y1, min(frame_width, x1 + width), min(frame_height, y1 + height))

    @staticmethod
    def _rect_overlap_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
        """Compute overlap area between two rectangles."""
        overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return overlap_x * overlap_y

    @staticmethod
    def _rect_center(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Return rectangle center."""
        return ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)

    def _label_state_key(self, op: Dict[str, Any], idx: int) -> str:
        """Build a stable temporal key for a label/object pair."""
        stable_id = (
            self.object_ids[idx]
            if self.object_ids is not None and idx < len(self.object_ids)
            else idx
        )
        return f"label:{stable_id}:{op.get('text', '')}:{op.get('placement_hint', '')}"

    def _label_text_for_object(self, op: Dict[str, Any], idx: int) -> str:
        """Resolve label text for a specific object."""
        text = op.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        if self.object_labels and idx < len(self.object_labels):
            label = self.object_labels[idx]
            if isinstance(label, str) and label.strip():
                return label.strip()
        return f"Object {idx + 1}"

    def _auto_font_size(self, box: Tuple[int, int, int, int]) -> int:
        """Pick a readable default font size from object geometry."""
        box_w = max(1, box[2] - box[0])
        box_h = max(1, box[3] - box[1])
        return max(14, min(42, int(max(box_w, box_h) * 0.12)))

    def _load_label_font(self, op: Dict[str, Any], font_size: int) -> ImageFont.ImageFont:
        """Load a cached custom font or fall back to a bundled/default font."""
        font_path = op.get("_font_path")
        if font_path:
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except Exception as exc:
                logger.warning("Failed to load cached label font %s: %s", font_path, exc)
                self._record_warning(
                    {
                        "code": "FONT_FALLBACK",
                        "message": f"Failed to load font '{font_path}'. Falling back to default font.",
                        "font_path": font_path,
                    }
                )

        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            return ImageFont.load_default()

    def _measure_text(
        self,
        text: str,
        font: ImageFont.ImageFont,
        stroke_width: int,
    ) -> Tuple[int, int, Tuple[int, int, int, int]]:
        """Measure label text bbox."""
        measure_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        measure_draw = ImageDraw.Draw(measure_img)
        bbox = measure_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        return (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]), bbox)

    def _resolve_label_font(
        self,
        text: str,
        op: Dict[str, Any],
        box: Tuple[int, int, int, int],
    ) -> Tuple[ImageFont.ImageFont, int, Tuple[int, int, int, int]]:
        """Choose a font size that fits comfortably within the frame."""
        explicit_size = op.get("font_size")
        font_size = int(explicit_size) if explicit_size is not None else self._auto_font_size(box)
        font_size = max(10, min(80, font_size))
        stroke_width = max(0, int(op.get("stroke_width", 0)))
        padding = max(4, int(op.get("padding", max(8, round(font_size * 0.4)))))

        for _ in range(8):
            font = self._load_label_font(op, font_size)
            text_w, text_h, text_bbox = self._measure_text(text, font, stroke_width)
            if explicit_size is not None:
                return font, font_size, text_bbox
            if text_w + padding * 2 <= self.width * 0.65 and text_h + padding * 2 <= self.height * 0.25:
                return font, font_size, text_bbox
            if font_size <= 12:
                return font, font_size, text_bbox
            font_size = max(12, font_size - 2)

        font = self._load_label_font(op, font_size)
        _, _, text_bbox = self._measure_text(text, font, stroke_width)
        return font, font_size, text_bbox

    def _label_candidates(
        self,
        box: Tuple[int, int, int, int],
        card_size: Tuple[int, int],
        placement_hint: Optional[str],
        offset: Tuple[int, int],
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Generate candidate label rectangles around an object box."""
        card_w, card_h = card_size
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        margin = max(10, int(min(self.width, self.height) * 0.015))
        dx, dy = offset

        raw_candidates = {
            "top": (round(cx - card_w / 2 + dx), round(y1 - margin - card_h + dy), round(cx + card_w / 2 + dx), round(y1 - margin + dy)),
            "bottom": (round(cx - card_w / 2 + dx), round(y2 + margin + dy), round(cx + card_w / 2 + dx), round(y2 + margin + card_h + dy)),
            "left": (round(x1 - margin - card_w + dx), round(cy - card_h / 2 + dy), round(x1 - margin + dx), round(cy + card_h / 2 + dy)),
            "right": (round(x2 + margin + dx), round(cy - card_h / 2 + dy), round(x2 + margin + card_w + dx), round(cy + card_h / 2 + dy)),
            "top_left": (round(x1 - card_w * 0.2 + dx), round(y1 - margin - card_h + dy), round(x1 - card_w * 0.2 + card_w + dx), round(y1 - margin + dy)),
            "top_right": (round(x2 - card_w * 0.8 + dx), round(y1 - margin - card_h + dy), round(x2 - card_w * 0.8 + card_w + dx), round(y1 - margin + dy)),
            "bottom_left": (round(x1 - card_w * 0.2 + dx), round(y2 + margin + dy), round(x1 - card_w * 0.2 + card_w + dx), round(y2 + margin + card_h + dy)),
            "bottom_right": (round(x2 - card_w * 0.8 + dx), round(y2 + margin + dy), round(x2 - card_w * 0.8 + card_w + dx), round(y2 + margin + card_h + dy)),
        }

        preferred_order = list(raw_candidates.keys())
        if placement_hint in raw_candidates:
            preferred_order.remove(placement_hint)
            preferred_order.insert(0, placement_hint)

        return [
            (
                name,
                self._clamp_rect(raw_candidates[name], self.width, self.height),
            )
            for name in preferred_order
        ]

    def _score_label_candidate(
        self,
        candidate_name: str,
        rect: Tuple[int, int, int, int],
        occupied_rects: List[Tuple[int, int, int, int]],
        target_box: Tuple[int, int, int, int],
        placement_hint: Optional[str],
        previous_state: Optional[Dict[str, Any]],
    ) -> float:
        """Score candidate label placement; lower is better."""
        score = 0.0
        target_center = self._rect_center(target_box)
        rect_center = self._rect_center(rect)
        score += math.dist(target_center, rect_center) * 0.08
        score += self._rect_overlap_area(rect, target_box) * 12.0
        score += sum(self._rect_overlap_area(rect, other) * 30.0 for other in occupied_rects)

        if placement_hint and candidate_name != placement_hint:
            score += 18.0

        if previous_state:
            prev_rect_raw = previous_state.get("rect")
            prev_candidate = previous_state.get("placement")
            if isinstance(prev_rect_raw, (list, tuple)) and len(prev_rect_raw) == 4:
                prev_rect = tuple(int(v) for v in prev_rect_raw)
                score += math.dist(self._rect_center(prev_rect), rect_center) * 0.2
            if prev_candidate == candidate_name:
                score -= 10.0

        frame_margin_penalty = (
            (0 if rect[0] > 0 else 4)
            + (0 if rect[1] > 0 else 4)
            + (0 if rect[2] < self.width else 4)
            + (0 if rect[3] < self.height else 4)
        )
        score += frame_margin_penalty
        return score

    def _choose_label_layout(
        self,
        op: Dict[str, Any],
        idx: int,
        box: Tuple[int, int, int, int],
        card_size: Tuple[int, int],
    ) -> Tuple[str, Tuple[int, int, int, int]]:
        """Select the best label placement with temporal stability."""
        placement_hint = op.get("placement_hint")
        offset = self._parse_offset(op.get("offset"))
        candidates = self._label_candidates(box, card_size, placement_hint, offset)
        state_key = self._label_state_key(op, idx)
        previous_positions = self.annotation_state.setdefault("label_positions", {})
        previous_state = previous_positions.get(state_key)
        best_name = "top"
        best_rect = candidates[0][1]
        best_score = float("inf")

        for candidate_name, rect in candidates:
            score = self._score_label_candidate(
                candidate_name,
                rect,
                self._frame_label_rects,
                box,
                placement_hint,
                previous_state,
            )
            if score < best_score:
                best_score = score
                best_name = candidate_name
                best_rect = rect

        if previous_state:
            prev_rect_raw = previous_state.get("rect")
            if isinstance(prev_rect_raw, (list, tuple)) and len(prev_rect_raw) == 4:
                prev_rect = tuple(int(v) for v in prev_rect_raw)
                blended_rect = tuple(
                    int(round(prev_rect[i] * 0.55 + best_rect[i] * 0.45))
                    for i in range(4)
                )
                best_rect = self._clamp_rect(blended_rect, self.width, self.height)

        previous_positions[state_key] = {
            "rect": list(best_rect),
            "placement": best_name,
        }
        self._frame_label_rects.append(best_rect)
        return best_name, best_rect

    @staticmethod
    def _anchor_between_rects(
        source_rect: Tuple[int, int, int, int],
        target_rect: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Choose leader-line anchor points between two rectangles."""
        source_center = EffectsPipeline._rect_center(source_rect)
        target_center = EffectsPipeline._rect_center(target_rect)

        source_anchor = (
            int(round(min(max(target_center[0], source_rect[0]), source_rect[2]))),
            int(round(min(max(target_center[1], source_rect[1]), source_rect[3]))),
        )
        target_anchor = (
            int(round(min(max(source_center[0], target_rect[0]), target_rect[2]))),
            int(round(min(max(source_center[1], target_rect[1]), target_rect[3]))),
        )
        return source_anchor, target_anchor

    @staticmethod
    def _apply_clip_to_overlay(overlay: Image.Image, clip_mask: np.ndarray) -> Image.Image:
        """Clip an RGBA overlay by a boolean mask."""
        overlay_arr = np.array(overlay)
        overlay_arr[~clip_mask, 3] = 0
        return Image.fromarray(overlay_arr)

    def _build_label_overlay(
        self,
        op: Dict[str, Any],
        box: Tuple[int, int, int, int],
        rect: Tuple[int, int, int, int],
        text: str,
        font: ImageFont.ImageFont,
        text_bbox: Tuple[int, int, int, int],
        font_size: int,
    ) -> Image.Image:
        """Render the label card, text, and optional leader line."""
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        opacity = max(0.0, min(1.0, float(op.get("opacity", 1.0))))
        text_color = self._parse_color(op.get("color"), (255, 255, 255, 255))
        background_color = self._parse_color(op.get("background_color"), (20, 28, 38, 210))
        border_color = self._parse_color(op.get("border_color"), (255, 255, 255, 160))
        stroke_color = self._parse_color(op.get("stroke_color"), (0, 0, 0, 255))
        stroke_width = max(0, int(op.get("stroke_width", 0)))
        padding = max(4, int(op.get("padding", max(8, round(font_size * 0.4)))))
        corner_radius = max(0, int(op.get("corner_radius", max(8, round(font_size * 0.45)))))

        shadow_cfg = op.get("shadow", True)
        if shadow_cfg:
            if isinstance(shadow_cfg, dict):
                shadow_color = self._parse_color(shadow_cfg.get("color"), (0, 0, 0, 110))
                shadow_offset = self._parse_offset(shadow_cfg.get("offset", [3, 4]))
                shadow_blur = max(0, int(shadow_cfg.get("blur", 8)))
            else:
                shadow_color = (0, 0, 0, 110)
                shadow_offset = (3, 4)
                shadow_blur = 8
            shadow_overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_overlay, "RGBA")
            shadow_rect = (
                rect[0] + shadow_offset[0],
                rect[1] + shadow_offset[1],
                rect[2] + shadow_offset[0],
                rect[3] + shadow_offset[1],
            )
            shadow_draw.rounded_rectangle(shadow_rect, radius=corner_radius, fill=shadow_color)
            overlay = Image.alpha_composite(
                overlay,
                shadow_overlay.filter(ImageFilter.GaussianBlur(radius=shadow_blur)),
            )
            draw = ImageDraw.Draw(overlay, "RGBA")

        leader_line = bool(op.get("leader_line", True))
        if leader_line:
            source_anchor, target_anchor = self._anchor_between_rects(box, rect)
            line_width = max(2, int(round(font_size * 0.12)))
            draw.line([source_anchor, target_anchor], fill=border_color, width=line_width)
            dot_radius = max(2, line_width)
            draw.ellipse(
                [
                    source_anchor[0] - dot_radius,
                    source_anchor[1] - dot_radius,
                    source_anchor[0] + dot_radius,
                    source_anchor[1] + dot_radius,
                ],
                fill=border_color,
            )

        draw.rounded_rectangle(rect, radius=corner_radius, fill=background_color)
        if border_color[3] > 0:
            draw.rounded_rectangle(rect, radius=corner_radius, outline=border_color, width=max(1, stroke_width or 1))

        text_pos = (
            rect[0] + padding - text_bbox[0],
            rect[1] + padding - text_bbox[1],
        )
        draw.text(
            text_pos,
            text,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )

        if opacity < 1.0:
            overlay_arr = np.array(overlay, dtype=np.uint8)
            overlay_arr[:, :, 3] = np.clip(
                overlay_arr[:, :, 3].astype(np.float32) * opacity,
                0,
                255,
            ).astype(np.uint8)
            overlay = Image.fromarray(overlay_arr)

        if op.get("_animation_mode") in {"reveal", "splash"}:
            overlay_mask = np.array(overlay)[:, :, 3] > 0
            clip_mask = self._build_animation_clip_mask(overlay_mask, op)
            overlay = self._apply_clip_to_overlay(overlay, clip_mask)

        return overlay

    def _op_label(self, op: Dict[str, Any]) -> None:
        """Render a smart label for each selected object."""
        selected_indices = self._sorted_object_indices(self._iter_selected_object_indices())
        if not selected_indices:
            return

        combined_overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        for idx in selected_indices:
            if not (0 <= idx < len(self.boxes)):
                continue
            box = self.boxes[idx]
            if box[2] <= box[0] or box[3] <= box[1]:
                box = self._box_from_mask(self.object_masks[idx])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            text = self._label_text_for_object(op, idx)
            font, font_size, text_bbox = self._resolve_label_font(text, op, box)
            stroke_width = max(0, int(op.get("stroke_width", 0)))
            text_w, text_h, _ = self._measure_text(text, font, stroke_width)
            padding = max(4, int(op.get("padding", max(8, round(font_size * 0.4)))))
            card_size = (
                min(self.width, text_w + padding * 2),
                min(self.height, text_h + padding * 2),
            )
            _, rect = self._choose_label_layout(op, idx, box, card_size)
            overlay = self._build_label_overlay(op, box, rect, text, font, text_bbox, font_size)
            combined_overlay = Image.alpha_composite(combined_overlay, overlay)

        self.image = Image.alpha_composite(self.image, combined_overlay)

    # =========================================================================
    # Blur / Privacy
    # =========================================================================

    def _op_blur(self, op: dict):
        """Apply Gaussian blur to the selected region (edge-aware).

        To prevent bleed from non-selected pixels into the blurred region,
        the non-selected area is filled with the average color of the selected
        region before blurring, then composited back.
        """
        strength = float(op.get("strength", 25))
        radius = max(0.0, strength)
        if radius <= 0.0:
            return
        mask = self.active_mask

        # Create a copy where the non-target region is filled with the
        # average color of the target region, preventing edge bleed
        img_arr = np.array(self.image)
        fill_img = img_arr.copy()
        if np.any(mask):
            avg_color = img_arr[mask, :].mean(axis=0).astype(np.uint8)
            fill_img[~mask] = avg_color

        blurred = Image.fromarray(fill_img).filter(ImageFilter.GaussianBlur(radius=radius))
        self._composite_with_mask(blurred)

    def _op_pixelate(self, op: dict):
        """Apply mosaic pixelation to the selected region."""
        block_size = op.get("block_size", 15)
        block_size = max(2, min(100, block_size))

        img_arr = np.array(self.image)
        mask = self.active_mask

        # Pixelate by downscaling and upscaling
        small_w = max(1, self.width // block_size)
        small_h = max(1, self.height // block_size)
        pixelated = self.image.resize((small_w, small_h), Image.NEAREST)
        pixelated = pixelated.resize((self.width, self.height), Image.NEAREST)
        pix_arr = np.array(pixelated)

        # Apply only to masked region
        mask_3d = np.stack([mask] * 4, axis=-1) if img_arr.shape[2] == 4 else np.stack([mask] * 3, axis=-1)
        img_arr[mask_3d] = pix_arr[mask_3d]
        self.image = Image.fromarray(img_arr)

    def _op_redact(self, op: dict):
        """Fill selected region with a solid color."""
        color = tuple(op.get("color", [0, 0, 0]))
        if len(color) == 3:
            color = color + (255,)

        img_arr = np.array(self.image)
        mask = self.active_mask
        img_arr[mask] = color
        self.image = Image.fromarray(img_arr)

    # =========================================================================
    # Color & Appearance
    # =========================================================================

    def _op_color_overlay(self, op: dict):
        """Apply semi-transparent color overlay to selected region."""
        color = tuple(op.get("color", [255, 0, 0, 128]))
        if len(color) == 3:
            color = color + (128,)

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        overlay_arr[self.active_mask] = color
        overlay = Image.fromarray(overlay_arr)
        self.image = Image.alpha_composite(self.image, overlay)

    def _op_color_grade(self, op: dict):
        """Adjust brightness, contrast, saturation of selected region."""
        brightness = op.get("brightness", 0)
        contrast = op.get("contrast", 0)
        saturation = op.get("saturation", 0)

        graded = self.image.copy()

        if brightness != 0:
            factor = 1.0 + brightness / 100.0
            graded = ImageEnhance.Brightness(graded).enhance(factor)
        if contrast != 0:
            factor = 1.0 + contrast / 100.0
            graded = ImageEnhance.Contrast(graded).enhance(factor)
        if saturation != 0:
            factor = 1.0 + saturation / 100.0
            graded = ImageEnhance.Color(graded).enhance(factor)

        self._composite_with_mask(graded)

    def _op_opacity(self, op: dict):
        """Adjust opacity of selected region."""
        value = op.get("value", 0.5)
        value = max(0.0, min(1.0, value))

        img_arr = np.array(self.image)
        mask = self.active_mask
        img_arr[mask, 3] = int(value * 255)
        self.image = Image.fromarray(img_arr)

    def _op_replace_color(self, op: dict):
        """Shift hue and adjust saturation of selected region."""
        hue_shift = op.get("hue_shift", 0)
        sat_scale = op.get("saturation_scale", 1.0)

        img_arr = np.array(self.image)
        # Work in HSV space
        rgb_img = Image.fromarray(img_arr[:, :, :3])
        hsv_arr = np.array(rgb_img.convert("HSV"))

        mask = self.active_mask
        # Shift hue (0-255 range in PIL HSV)
        hue_shift_scaled = int(hue_shift * 255 / 360)
        hsv_arr[mask, 0] = (hsv_arr[mask, 0].astype(int) + hue_shift_scaled) % 256
        # Scale saturation
        hsv_arr[mask, 1] = np.clip(hsv_arr[mask, 1].astype(float) * sat_scale, 0, 255).astype(np.uint8)

        recolored = Image.fromarray(hsv_arr, "HSV").convert("RGB")
        recolored_arr = np.array(recolored)
        img_arr[mask, :3] = recolored_arr[mask]
        self.image = Image.fromarray(img_arr)

    # =========================================================================
    # Compositing
    # =========================================================================

    def _op_remove_background(self, op: dict):
        """Make background transparent (keep only masked objects)."""
        img_arr = np.array(self.image)
        # Set alpha to 0 for background pixels
        img_arr[~self.combined_mask, 3] = 0
        self.image = Image.fromarray(img_arr)

    def _op_replace_background(self, op: dict):
        """Replace background with a solid color or tiled image."""
        color = op.get("color", None)
        # image_url support would need async download — handled at router level
        bg_image_data = op.get("_bg_image_data", None)

        if bg_image_data is not None:
            bg = Image.open(io.BytesIO(bg_image_data)).convert("RGBA")
            bg = bg.resize((self.width, self.height), Image.LANCZOS)
        elif color is not None:
            c = tuple(color)
            if len(c) == 3:
                c = c + (255,)
            bg = Image.new("RGBA", (self.width, self.height), c)
        else:
            bg = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))

        img_arr = np.array(self.image)
        bg_arr = np.array(bg)
        # Replace background pixels
        bg_mask = ~self.combined_mask
        img_arr[bg_mask] = bg_arr[bg_mask]
        self.image = Image.fromarray(img_arr)

    def _op_greenscreen(self, op: dict):
        """Replace background with green screen color."""
        img_arr = np.array(self.image)
        bg_mask = ~self.combined_mask
        img_arr[bg_mask] = [0, 177, 64, 255]  # Standard green screen
        self.image = Image.fromarray(img_arr)

    # =========================================================================
    # Drawing & Annotation
    # =========================================================================

    def _op_outline(self, op: dict):
        """Draw smooth, anti-aliased contour lines around detected objects.
        
        Supports 'progress' param (0.0-1.0) for progressive draw animation.
        Uses Gaussian blur on the contour mask for smooth edges.
        """
        color = tuple(op.get("color", [0, 255, 0, 255]))
        if len(color) == 3:
            color = color + (255,)
        thickness = op.get("thickness", 3)
        progress = max(0.0, min(1.0, op.get("progress", 1.0)))
        # Anti-aliasing blur radius — proportional to thickness for consistent smoothness
        smooth_radius = max(1, thickness // 2 + 1)

        for obj_mask in self.object_masks:
            if not np.any(obj_mask):
                continue
            contour = self._find_contour_pixels(obj_mask, thickness)

            # If progress < 1.0, only show partial contour (for draw animation)
            if progress < 1.0:
                contour = self._partial_contour(contour, progress)

            # Create a grayscale contour mask and smooth it with Gaussian blur
            contour_gray = (contour.astype(np.float32) * 255).astype(np.uint8)
            contour_img = Image.fromarray(contour_gray, "L")
            contour_img = contour_img.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
            contour_smooth = np.array(contour_img)  # 0-255 alpha values

            # Create RGBA overlay using the smooth alpha
            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            overlay_arr = np.array(overlay)
            # Set color channels where contour has any value
            mask_any = contour_smooth > 0
            overlay_arr[mask_any, 0] = color[0]
            overlay_arr[mask_any, 1] = color[1]
            overlay_arr[mask_any, 2] = color[2]
            # Alpha = min(contour_smooth, color_alpha) for smooth falloff
            overlay_arr[mask_any, 3] = np.minimum(contour_smooth[mask_any], color[3])
            overlay = Image.fromarray(overlay_arr)
            self.image = Image.alpha_composite(self.image, overlay)

    # text_label operation removed — no longer supported

    def _op_bounding_box(self, op: dict):
        """Draw bounding boxes around detected objects."""
        color = tuple(op.get("color", [255, 0, 0, 255]))
        if len(color) == 3:
            color = color + (255,)
        thickness = op.get("thickness", 2)

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for box in self.boxes:
            x1, y1, x2, y2 = box
            for t in range(thickness):
                draw.rectangle(
                    [x1 - t, y1 - t, x2 + t, y2 + t],
                    outline=color,
                )

        self.image = Image.alpha_composite(self.image, overlay)

    # =========================================================================
    # Creative Effects
    # =========================================================================

    def _op_spotlight(self, op: dict):
        """Darken everything except the selected objects."""
        darkness = op.get("darkness", 0.7)
        darkness = max(0.0, min(1.0, darkness))

        dark_overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        dark_arr = np.array(dark_overlay)
        alpha = int(darkness * 255)
        bg_mask = ~self.combined_mask
        dark_arr[bg_mask] = [0, 0, 0, alpha]
        dark_overlay = Image.fromarray(dark_arr)
        self.image = Image.alpha_composite(self.image, dark_overlay)

    def _op_bokeh(self, op: dict):
        """Simulate depth-of-field blur on background."""
        strength = op.get("strength", 15)
        radius = max(1, int(strength))

        blurred = self.image.filter(ImageFilter.GaussianBlur(radius=radius))
        # Replace only background pixels with blurred version
        img_arr = np.array(self.image)
        blur_arr = np.array(blurred)
        bg_mask = ~self.combined_mask
        channels = img_arr.shape[2]
        mask_expanded = np.stack([bg_mask] * channels, axis=-1)
        img_arr[mask_expanded] = blur_arr[mask_expanded]
        self.image = Image.fromarray(img_arr)

    def _op_glow(self, op: dict):
        """Add glow/bloom around the edges of selected objects."""
        color = tuple(op.get("color", [255, 255, 255]))
        radius = op.get("radius", 15)
        intensity = op.get("intensity", 0.7)

        if len(color) == 3:
            color_rgba = color + (int(intensity * 255),)
        else:
            color_rgba = color

        # Create glow by dilating the mask and then blurring
        mask_img = Image.fromarray((self.combined_mask * 255).astype(np.uint8), "L")
        # Dilate by blurring and thresholding at a lower value
        dilated = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
        dilated_arr = np.array(dilated)

        # Glow = dilated area minus original mask
        glow_mask = (dilated_arr > 20) & ~self.combined_mask

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)

        # Apply glow with gradient intensity based on distance from mask
        glow_intensity = dilated_arr[glow_mask].astype(float) / 255.0 * intensity
        overlay_arr[glow_mask, 0] = color[0]
        overlay_arr[glow_mask, 1] = color[1]
        overlay_arr[glow_mask, 2] = color[2]
        overlay_arr[glow_mask, 3] = (glow_intensity * 255).astype(np.uint8)

        overlay = Image.fromarray(overlay_arr)
        self.image = Image.alpha_composite(self.image, overlay)

    def _op_shadow(self, op: dict):
        """Add drop shadow to selected objects."""
        offset = op.get("offset", [5, 5])
        blur_radius = op.get("blur", 10)
        color = tuple(op.get("color", [0, 0, 0, 160]))
        if len(color) == 3:
            color = color + (160,)

        # Create shadow mask (shift combined mask by offset)
        shadow = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        shadow_arr = np.array(shadow)

        ox, oy = int(offset[0]), int(offset[1])
        shifted_mask = np.zeros_like(self.combined_mask)
        # Clip the shift bounds
        src_y_start = max(0, -oy)
        src_y_end = min(self.height, self.height - oy)
        src_x_start = max(0, -ox)
        src_x_end = min(self.width, self.width - ox)
        dst_y_start = max(0, oy)
        dst_y_end = min(self.height, self.height + oy)
        dst_x_start = max(0, ox)
        dst_x_end = min(self.width, self.width + ox)

        h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        if h > 0 and w > 0:
            shifted_mask[dst_y_start:dst_y_start+h, dst_x_start:dst_x_start+w] = \
                self.combined_mask[src_y_start:src_y_start+h, src_x_start:src_x_start+w]

        # Shadow only outside the object
        shadow_only = shifted_mask & ~self.combined_mask
        shadow_arr[shadow_only] = color
        shadow = Image.fromarray(shadow_arr)

        if blur_radius > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Composite shadow below the current image
        result = shadow.copy()
        result = Image.alpha_composite(result, self.image)
        self.image = result

    def _op_vignette(self, op: dict):
        """Apply vignette effect centered on detected objects."""
        strength = op.get("strength", 0.5)
        strength = max(0.0, min(1.0, strength))

        # Find center of detected objects
        if np.any(self.combined_mask):
            ys, xs = np.where(self.combined_mask)
            cy, cx = ys.mean(), xs.mean()
        else:
            cy, cx = self.height / 2, self.width / 2

        # Create radial gradient
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        max_dist = np.sqrt(self.width**2 + self.height**2) / 2
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        vignette = np.clip(dist / max_dist * strength, 0, 1)

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        overlay_arr[:, :, 3] = (vignette * 255).astype(np.uint8)
        overlay = Image.fromarray(overlay_arr)
        self.image = Image.alpha_composite(self.image, overlay)

    def _op_remove_background_v2(self, op: dict):
        """Make everything outside the selected subject transparent."""
        img_arr = np.array(self.image)
        subject_mask = self._selected_subject_mask()
        img_arr[~subject_mask, 3] = 0
        self.image = Image.fromarray(img_arr)

    def _op_replace_background_v2(self, op: dict):
        """Replace everything outside the selected subject with a solid color or image."""
        color = op.get("color", None)
        bg_image_data = op.get("_bg_image_data", None)

        if bg_image_data is not None:
            bg = Image.open(io.BytesIO(bg_image_data)).convert("RGBA")
            bg = bg.resize((self.width, self.height), Image.LANCZOS)
        elif color is not None:
            c = tuple(color)
            if len(c) == 3:
                c = c + (255,)
            bg = Image.new("RGBA", (self.width, self.height), c)
        else:
            bg = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))

        img_arr = np.array(self.image)
        bg_arr = np.array(bg)
        bg_mask = ~self._selected_subject_mask()
        img_arr[bg_mask] = bg_arr[bg_mask]
        self.image = Image.fromarray(img_arr)

    def _op_greenscreen_v2(self, op: dict):
        """Replace everything outside the selected subject with standard green-screen color."""
        img_arr = np.array(self.image)
        bg_mask = ~self._selected_subject_mask()
        img_arr[bg_mask] = [0, 177, 64, 255]
        self.image = Image.fromarray(img_arr)

    def _op_outline_v2(self, op: dict):
        """Draw outlines only around the currently selected objects."""
        color = tuple(op.get("color", [0, 255, 0, 255]))
        if len(color) == 3:
            color = color + (255,)
        thickness = op.get("thickness", 3)
        progress = max(0.0, min(1.0, op.get("progress", 1.0)))
        smooth_radius = max(1, thickness // 2 + 1)
        clip_mask = op.get("_clip_mask")

        for idx in self._iter_selected_object_indices():
            if not (0 <= idx < len(self.object_masks)):
                continue
            obj_mask = self.object_masks[idx]
            if not np.any(obj_mask):
                continue
            contour = self._find_contour_pixels(obj_mask, thickness)
            if progress < 1.0:
                contour = self._partial_contour(contour, progress)

            contour_gray = (contour.astype(np.float32) * 255).astype(np.uint8)
            contour_img = Image.fromarray(contour_gray, "L")
            contour_img = contour_img.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
            contour_smooth = np.array(contour_img)
            if isinstance(clip_mask, np.ndarray):
                contour_smooth = np.where(clip_mask, contour_smooth, 0)

            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            overlay_arr = np.array(overlay)
            mask_any = contour_smooth > 0
            overlay_arr[mask_any, 0] = color[0]
            overlay_arr[mask_any, 1] = color[1]
            overlay_arr[mask_any, 2] = color[2]
            overlay_arr[mask_any, 3] = np.minimum(contour_smooth[mask_any], color[3])
            overlay = Image.fromarray(overlay_arr)
            self.image = Image.alpha_composite(self.image, overlay)

    def _op_bounding_box_v2(self, op: dict):
        """Draw bounding boxes only around the currently selected objects."""
        color = tuple(op.get("color", [255, 0, 0, 255]))
        if len(color) == 3:
            color = color + (255,)
        thickness = op.get("thickness", 2)
        progress = max(0.0, min(1.0, float(op.get("progress", 1.0))))
        clip_mask = op.get("_clip_mask")

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for idx in self._iter_selected_object_indices():
            if not (0 <= idx < len(self.boxes)):
                continue
            x1, y1, x2, y2 = self.boxes[idx]
            for t in range(thickness):
                inset_box = [x1 - t, y1 - t, x2 + t, y2 + t]
                if progress >= 1.0:
                    draw.rectangle(inset_box, outline=color)
                elif progress > 0.0:
                    self._draw_partial_rectangle(draw, inset_box, color, progress)

        if isinstance(clip_mask, np.ndarray):
            overlay = self._apply_clip_to_overlay(overlay, clip_mask)

        self.image = Image.alpha_composite(self.image, overlay)

    def _op_spotlight_v2(self, op: dict):
        """Darken everything except the selected subject."""
        darkness = max(0.0, min(1.0, op.get("darkness", 0.7)))
        subject_mask = self._selected_subject_mask()

        dark_overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        dark_arr = np.array(dark_overlay)
        dark_arr[~subject_mask] = [0, 0, 0, int(darkness * 255)]
        dark_overlay = Image.fromarray(dark_arr)
        self.image = Image.alpha_composite(self.image, dark_overlay)

    def _op_bokeh_v2(self, op: dict):
        """Apply depth-of-field blur outside the selected subject."""
        radius = max(1, int(op.get("strength", 15)))
        subject_mask = self._selected_subject_mask()

        blurred = self.image.filter(ImageFilter.GaussianBlur(radius=radius))
        img_arr = np.array(self.image)
        blur_arr = np.array(blurred)
        bg_mask = ~subject_mask
        mask_expanded = np.stack([bg_mask] * img_arr.shape[2], axis=-1)
        img_arr[mask_expanded] = blur_arr[mask_expanded]
        self.image = Image.fromarray(img_arr)

    def _op_glow_v2(self, op: dict):
        """Add glow around the selected subject."""
        color = tuple(op.get("color", [255, 255, 255]))
        radius = op.get("radius", 15)
        intensity = op.get("intensity", 0.7)
        subject_mask = self._selected_subject_mask()

        mask_img = Image.fromarray((subject_mask * 255).astype(np.uint8), "L")
        dilated = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
        dilated_arr = np.array(dilated)
        glow_mask = (dilated_arr > 20) & ~subject_mask

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        glow_intensity = dilated_arr[glow_mask].astype(float) / 255.0 * intensity
        overlay_arr[glow_mask, 0] = color[0]
        overlay_arr[glow_mask, 1] = color[1]
        overlay_arr[glow_mask, 2] = color[2]
        overlay_arr[glow_mask, 3] = (glow_intensity * 255).astype(np.uint8)
        overlay = Image.fromarray(overlay_arr)
        self.image = Image.alpha_composite(self.image, overlay)

    def _op_shadow_v2(self, op: dict):
        """Add a drop shadow to the selected subject."""
        offset = op.get("offset", [5, 5])
        blur_radius = op.get("blur", 10)
        color = tuple(op.get("color", [0, 0, 0, 160]))
        if len(color) == 3:
            color = color + (160,)
        subject_mask = self._selected_subject_mask()

        shadow = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        shadow_arr = np.array(shadow)

        ox, oy = int(offset[0]), int(offset[1])
        shifted_mask = np.zeros_like(subject_mask)
        src_y_start = max(0, -oy)
        src_y_end = min(self.height, self.height - oy)
        src_x_start = max(0, -ox)
        src_x_end = min(self.width, self.width - ox)
        dst_y_start = max(0, oy)
        dst_y_end = min(self.height, self.height + oy)
        dst_x_start = max(0, ox)
        dst_x_end = min(self.width, self.width + ox)

        h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        if h > 0 and w > 0:
            shifted_mask[dst_y_start:dst_y_start+h, dst_x_start:dst_x_start+w] = \
                subject_mask[src_y_start:src_y_start+h, src_x_start:src_x_start+w]

        shadow_only = shifted_mask & ~subject_mask
        shadow_arr[shadow_only] = color
        shadow = Image.fromarray(shadow_arr)

        if blur_radius > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        self.image = Image.alpha_composite(shadow, self.image)

    def _op_vignette_v2(self, op: dict):
        """Apply vignette centered on the selected subject."""
        strength = max(0.0, min(1.0, op.get("strength", 0.5)))
        subject_mask = self._selected_subject_mask()

        if np.any(subject_mask):
            ys, xs = np.where(subject_mask)
            cy, cx = ys.mean(), xs.mean()
        else:
            cy, cx = self.height / 2, self.width / 2

        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        max_dist = np.sqrt(self.width**2 + self.height**2) / 2
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        vignette = np.clip(dist / max_dist * strength, 0, 1)

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        overlay_arr[:, :, 3] = (vignette * 255).astype(np.uint8)
        overlay = Image.fromarray(overlay_arr)
        self.image = Image.alpha_composite(self.image, overlay)

    # =========================================================================
    # Filters
    # =========================================================================

    def _op_grayscale(self, op: dict):
        """Convert selection to grayscale with blend intensity."""
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        img_arr = np.array(self.image)
        mask = self.active_mask

        luminance = np.dot(img_arr[mask, :3].astype(np.float32), [0.299, 0.587, 0.114])
        gray_rgb = np.stack([luminance] * 3, axis=-1)
        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity) + gray_rgb * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_invert(self, op: dict):
        """Invert colors of selection with blend intensity."""
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        img_arr = np.array(self.image)
        mask = self.active_mask
        inverted = 255 - img_arr[mask, :3]
        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity) + inverted.astype(np.float32) * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_sharpen(self, op: dict):
        """Sharpen edges within selection."""
        strength = max(0.0, min(10.0, op.get("strength", 2.0)))

        sharpened = self.image.filter(ImageFilter.SHARPEN)
        if strength > 1.0:
            enhancer = ImageEnhance.Sharpness(self.image)
            sharpened = enhancer.enhance(strength)
        self._composite_with_mask(sharpened)

    def _op_sepia(self, op: dict):
        """Apply sepia tone to selection."""
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        img_arr = np.array(self.image).astype(np.float32)
        mask = self.active_mask
        r, g, b = img_arr[mask, 0], img_arr[mask, 1], img_arr[mask, 2]

        sepia_r = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
        sepia_g = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
        sepia_b = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)

        img_arr[mask, 0] = (r * (1 - intensity) + sepia_r * intensity).astype(np.uint8)
        img_arr[mask, 1] = (g * (1 - intensity) + sepia_g * intensity).astype(np.uint8)
        img_arr[mask, 2] = (b * (1 - intensity) + sepia_b * intensity).astype(np.uint8)
        self.image = Image.fromarray(img_arr.astype(np.uint8))

    def _op_posterize(self, op: dict):
        """Reduce color levels within selection."""
        levels = max(2, min(32, int(op.get("levels", 4))))

        img_arr = np.array(self.image)
        mask = self.active_mask
        factor = 256.0 / levels
        img_arr[mask, :3] = (np.floor(img_arr[mask, :3].astype(np.float32) / factor) * factor).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_edge_detect(self, op: dict):
        """Apply edge detection to selection with blend."""
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        edges = self.image.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges)
        img_arr = np.array(self.image)
        mask = self.active_mask
        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity)
            + edges_arr[mask, :3].astype(np.float32) * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_emboss(self, op: dict):
        """Apply emboss relief effect to selection."""
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        embossed = self.image.filter(ImageFilter.EMBOSS)
        embossed_arr = np.array(embossed)
        img_arr = np.array(self.image)
        mask = self.active_mask
        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity)
            + embossed_arr[mask, :3].astype(np.float32) * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_noise(self, op: dict):
        """Add noise/grain to selection."""
        amount = max(0.0, min(1.0, op.get("amount", 0.3)))
        noise_type = op.get("noise_type", "gaussian")
        seed = op.get("seed", None)

        rng = np.random.RandomState(seed)
        img_arr = np.array(self.image)
        mask = self.active_mask
        h, w = img_arr.shape[:2]

        if noise_type == "grain":
            noise_mono = rng.normal(0, 255 * amount * 0.5, (h, w)).astype(np.float32)
            noise_rgb = np.stack([noise_mono] * 3, axis=-1)
        else:  # gaussian
            noise_rgb = rng.normal(0, 255 * amount * 0.5, (h, w, 3)).astype(np.float32)

        pixels = img_arr[mask, :3].astype(np.float32)
        noise_pixels = noise_rgb[mask]
        img_arr[mask, :3] = np.clip(pixels + noise_pixels, 0, 255).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_sketch(self, op: dict):
        """Convert selection to pencil sketch effect."""
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))
        detail = max(1, min(10, int(op.get("detail", 5))))

        img_arr = np.array(self.image)
        mask = self.active_mask

        # Create sketch: grayscale → invert → blur → blend (dodge)
        gray = np.dot(img_arr[:, :, :3].astype(np.float32), [0.299, 0.587, 0.114])
        inv_gray = 255.0 - gray
        blur_radius = detail * 3
        inv_blur = np.array(
            Image.fromarray(inv_gray.astype(np.uint8), "L").filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )
        ).astype(np.float32)

        # Color dodge blend
        sketch_val = np.clip(gray / (256.0 - inv_blur + 1e-6) * 256.0, 0, 255)
        sketch_rgb = np.stack([sketch_val] * 3, axis=-1).astype(np.uint8)

        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity)
            + sketch_rgb[mask].astype(np.float32) * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    # =========================================================================
    # Artistic
    # =========================================================================

    def _op_duotone(self, op: dict):
        """Map selection to a two-color palette."""
        color_dark = tuple(op.get("color_dark", [20, 0, 80]))
        color_light = tuple(op.get("color_light", [255, 200, 100]))
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        img_arr = np.array(self.image)
        mask = self.active_mask
        luminance = np.dot(img_arr[mask, :3].astype(np.float32), [0.299, 0.587, 0.114]) / 255.0

        duo = np.zeros_like(img_arr[mask, :3], dtype=np.float32)
        for c in range(3):
            duo[:, c] = color_dark[c] * (1 - luminance) + color_light[c] * luminance

        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity) + duo * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_halftone(self, op: dict):
        """Apply halftone dot pattern to selection."""
        dot_size = max(2, min(30, int(op.get("dot_size", 6))))
        intensity = max(0.0, min(1.0, op.get("intensity", 1.0)))

        img_arr = np.array(self.image)
        mask = self.active_mask

        gray = np.dot(img_arr[:, :, :3].astype(np.float32), [0.299, 0.587, 0.114])
        halftone = np.zeros_like(gray)

        for y in range(0, self.height, dot_size):
            for x in range(0, self.width, dot_size):
                block = gray[y:y+dot_size, x:x+dot_size]
                if block.size == 0:
                    continue
                avg = block.mean() / 255.0
                radius = int(avg * dot_size * 0.5)
                cy, cx = y + dot_size // 2, x + dot_size // 2
                yy, xx = np.ogrid[max(0,cy-radius):min(self.height,cy+radius+1),
                                  max(0,cx-radius):min(self.width,cx+radius+1)]
                dist = (yy - cy)**2 + (xx - cx)**2
                circle_mask = dist <= radius**2
                halftone[max(0,cy-radius):min(self.height,cy+radius+1),
                         max(0,cx-radius):min(self.width,cx+radius+1)][circle_mask] = 255

        ht_rgb = np.stack([halftone] * 3, axis=-1).astype(np.float32)
        img_arr[mask, :3] = (
            img_arr[mask, :3].astype(np.float32) * (1 - intensity) + ht_rgb[mask] * intensity
        ).astype(np.uint8)
        self.image = Image.fromarray(img_arr)

    def _op_glitch(self, op: dict):
        """RGB channel shift + scanline glitch effect."""
        intensity_val = max(0.0, min(1.0, op.get("intensity", 0.5)))
        rgb_shift = max(0, min(30, int(op.get("rgb_shift", 10))))
        seed = op.get("seed", 42)

        rng = np.random.RandomState(seed)
        img_arr = np.array(self.image)
        mask = self.active_mask
        glitched = img_arr.copy()
        shift = int(rgb_shift * intensity_val)

        if shift > 0:
            # Shift red channel right
            glitched[:, shift:, 0] = img_arr[:, :-shift, 0]
            # Shift blue channel left
            glitched[:, :-shift, 2] = img_arr[:, shift:, 2]

        # Add random scanlines
        num_lines = max(1, int(20 * intensity_val))
        for _ in range(num_lines):
            y = rng.randint(0, self.height)
            line_h = rng.randint(1, max(2, int(4 * intensity_val)))
            x_off = rng.randint(-shift * 2, shift * 2 + 1) if shift > 0 else 0
            y_end = min(y + line_h, self.height)
            if x_off > 0:
                glitched[y:y_end, x_off:, :3] = img_arr[y:y_end, :-x_off or None, :3]
            elif x_off < 0:
                glitched[y:y_end, :x_off, :3] = img_arr[y:y_end, -x_off:, :3]

        channels = img_arr.shape[2]
        mask_exp = np.stack([mask] * channels, axis=-1)
        img_arr[mask_exp] = glitched[mask_exp]
        self.image = Image.fromarray(img_arr)

    # =========================================================================
    # Distortion
    # =========================================================================

    def _op_motion_blur(self, op: dict):
        """Apply directional motion blur to selection."""
        angle = op.get("angle", 0) % 360
        strength = max(0, min(50, int(op.get("strength", 15))))
        if strength <= 0:
            return

        # Create motion blur kernel
        kernel_size = strength * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = strength
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        for i in range(-strength, strength + 1):
            x = int(center + i * cos_a)
            y = int(center + i * sin_a)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        kernel /= max(1, kernel.sum())

        kernel_img = ImageFilter.Kernel(
            size=(kernel_size, kernel_size),
            kernel=kernel.flatten().tolist(),
            scale=1,
            offset=0,
        )

        try:
            blurred = self.image.filter(kernel_img)
        except Exception:
            # Fallback: use simple directional blur via multiple box blurs
            blurred = self.image.filter(ImageFilter.GaussianBlur(radius=strength))

        self._composite_with_mask(blurred)

    def _op_glass(self, op: dict):
        """Apply frosted glass distortion to selection."""
        strength = max(1, min(30, int(op.get("strength", 8))))
        scale = max(1, min(20, int(op.get("scale", 4))))

        img_arr = np.array(self.image)
        mask = self.active_mask
        result = img_arr.copy()

        rng = np.random.RandomState(op.get("seed", 0))
        h, w = img_arr.shape[:2]

        # Create displacement map
        disp_y = (rng.rand(h // scale + 1, w // scale + 1) * 2 - 1) * strength
        disp_x = (rng.rand(h // scale + 1, w // scale + 1) * 2 - 1) * strength

        # Upscale displacement to full resolution
        disp_y_img = Image.fromarray(disp_y.astype(np.float32)).resize((w, h), Image.BILINEAR)
        disp_x_img = Image.fromarray(disp_x.astype(np.float32)).resize((w, h), Image.BILINEAR)
        disp_y_full = np.array(disp_y_img)
        disp_x_full = np.array(disp_x_img)

        ys, xs = np.where(mask)
        src_y = np.clip((ys + disp_y_full[ys, xs]).astype(int), 0, h - 1)
        src_x = np.clip((xs + disp_x_full[ys, xs]).astype(int), 0, w - 1)
        result[ys, xs] = img_arr[src_y, src_x]

        self.image = Image.fromarray(result)

    # =========================================================================
    # Mask Processing
    # =========================================================================

    def _op_feather(self, op: dict):
        """Soften mask edges with Gaussian feathering.
        
        Modifies the active mask to have soft (anti-aliased) edges,
        then re-composites the original and processed image.
        """
        radius = max(1, min(50, int(op.get("radius", 10))))

        # Create soft-edge mask by blurring the binary mask
        mask_img = Image.fromarray(
            (self.active_mask * 255).astype(np.uint8), "L"
        )
        soft_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Update active mask to soft version (float)
        self.active_mask = np.array(soft_mask).astype(np.float32) / 255.0 > 0.01
        logger.debug(f"Feathered mask with radius {radius}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _composite_with_mask(self, modified: Image.Image, feather_radius: int = 3):
        """Alpha-blend modified image into self.image using active_mask.
        
        Applies automatic edge feathering (Gaussian blur on the mask boundary)
        to eliminate jagged pixel edges on all operations.
        """
        img_arr = np.array(self.image).astype(np.float32)
        mod_arr = np.array(modified).astype(np.float32)

        # Create a soft mask by blurring the binary mask edges
        mask_uint8 = (self.active_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8, "L")
        if feather_radius > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        alpha = np.array(mask_img).astype(np.float32) / 255.0  # 0.0-1.0

        # Expand alpha to match channel count
        channels = img_arr.shape[2]
        alpha_expanded = np.stack([alpha] * channels, axis=-1)

        # Alpha-blend: result = modified * alpha + original * (1 - alpha)
        blended = mod_arr * alpha_expanded + img_arr * (1.0 - alpha_expanded)
        self.image = Image.fromarray(blended.astype(np.uint8))

    @staticmethod
    def _draw_partial_rectangle(
        draw: ImageDraw.ImageDraw,
        box: List[int],
        color: Tuple[int, int, int, int],
        progress: float,
    ) -> None:
        """Draw a rectangle progressively around its perimeter."""
        x1, y1, x2, y2 = box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        perimeter = (2 * width) + (2 * height)
        remaining = perimeter * max(0.0, min(1.0, progress))
        if remaining <= 0:
            return

        segments = [
            ((x1, y1), (x2, y1), width),
            ((x2, y1), (x2, y2), height),
            ((x2, y2), (x1, y2), width),
            ((x1, y2), (x1, y1), height),
        ]
        for start, end, length in segments:
            if remaining <= 0:
                break
            segment_fraction = min(1.0, remaining / max(1.0, length))
            partial_end = (
                int(round(start[0] + (end[0] - start[0]) * segment_fraction)),
                int(round(start[1] + (end[1] - start[1]) * segment_fraction)),
            )
            draw.line([start, partial_end], fill=color, width=1)
            remaining -= length * segment_fraction

    def _find_contour_pixels(self, mask: np.ndarray, thickness: int = 3) -> np.ndarray:
        """Find contour pixels of a binary mask using morphological operations."""
        from scipy import ndimage

        struct = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.binary_dilation(mask, structure=struct, iterations=thickness)
        eroded = ndimage.binary_erosion(mask, structure=struct, iterations=max(1, thickness // 2))
        contour = dilated & ~eroded
        return contour

    def _partial_contour(self, contour: np.ndarray, progress: float) -> np.ndarray:
        """Return only a percentage of contour pixels for draw animation.
        
        Traces the contour clockwise from the topmost point, revealing
        only `progress` fraction of the total pixels.
        """
        ys, xs = np.where(contour)
        if len(ys) == 0:
            return contour

        # Find centroid and compute angles for clockwise ordering
        cy, cx = ys.mean(), xs.mean()
        angles = np.arctan2(ys - cy, xs - cx)
        order = np.argsort(angles)

        # Keep only first N% of ordered contour pixels
        n_show = max(1, int(len(order) * progress))
        partial_mask = np.zeros_like(contour)
        show_idx = order[:n_show]
        partial_mask[ys[show_idx], xs[show_idx]] = True
        return partial_mask


def apply_effects_to_frame(
    frame: np.ndarray,
    masks: List[np.ndarray],
    operations: List[Dict[str, Any]],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    labels: Optional[List[str]] = None,
    object_ids: Optional[List[int]] = None,
    annotation_state: Optional[Dict[str, Any]] = None,
    warnings_out: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """Convenience function for video frame processing.
    
    Args:
        frame: BGR or RGB numpy array (H, W, 3)
        masks: List of binary masks
        operations: List of operation dicts
        boxes: Optional bounding boxes
        labels: Optional per-object labels
        object_ids: Optional stable per-object IDs
        
    Returns:
        Processed frame as numpy array (H, W, 3) in RGB
    """
    # Convert numpy frame to PIL
    image = Image.fromarray(frame)

    pipeline = EffectsPipeline(
        image,
        masks,
        boxes,
        labels=labels,
        object_ids=object_ids,
        annotation_state=annotation_state,
    )
    result = pipeline.apply(operations)
    if warnings_out is not None:
        warnings_out.extend(pipeline.warnings)

    # Convert back to RGB numpy array
    return np.array(result.convert("RGB"))
