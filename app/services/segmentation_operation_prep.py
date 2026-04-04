"""Preparation and validation helpers for segmentation operations."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.exceptions import ValidationError
from app.services.font_cache import get_font_cache_service
from app.services.segmentation_animation import AnimationInterpolator
from app.services.storage import StorageService


@dataclass
class PreparedOperations:
    """Prepared operation payload and any non-fatal warnings."""

    operations: Optional[List[Dict[str, Any]]]
    warnings: List[Dict[str, Any]]


@dataclass
class _SelectionState:
    target: str = "mask"
    explicit_object_scope: bool = False


_OBJECT_SCOPE_KEYS = ("object_id", "object_ids", "object_index", "object_label", "object_labels")
_DRAW_COMPATIBLE_TYPES = {"outline", "bounding_box"}
_NON_GEOMETRIC_ANIMATION_TYPES = {"select", "zoom", "pan", "feather"}
_NON_STAGGERABLE_TYPES = {"select", "zoom", "pan", "feather"}


async def prepare_segmentation_operations(
    storage: StorageService,
    operations: Optional[List[Dict[str, Any]]],
    *,
    validate_animations: bool = False,
) -> PreparedOperations:
    """Hydrate remote assets and validate operation compatibility."""
    if not operations:
        return PreparedOperations(operations=operations, warnings=[])

    hydrated = copy.deepcopy(operations)
    warnings: List[Dict[str, Any]] = []
    font_cache = get_font_cache_service()

    for index, op in enumerate(hydrated):
        if not isinstance(op, dict):
            continue

        if op.get("type") == "replace_background":
            image_url = op.get("image_url")
            if image_url:
                bg_image_data = await storage.download_from_url(image_url)
                if not _validate_image_like_payload(bg_image_data):
                    raise ValidationError("replace_background.image_url is not a valid image")
                op["_bg_image_data"] = bg_image_data

        if op.get("type") == "label":
            font_url = op.get("font_url")
            if font_url:
                cached_font = await font_cache.ensure_font_cached(storage, font_url)
                if cached_font.path is not None:
                    op["_font_path"] = str(cached_font.path)
                elif cached_font.warning:
                    warnings.append(
                        {
                            "code": "FONT_FALLBACK",
                            "message": cached_font.warning,
                            "operation_index": index,
                            "font_url": font_url,
                        }
                    )

    _validate_operation_sequence(hydrated, validate_animations=validate_animations)
    return PreparedOperations(operations=hydrated, warnings=warnings)


def _validate_image_like_payload(data: bytes) -> bool:
    """Validate image data by checking magic bytes."""
    if len(data) < 8:
        return False
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def _validate_operation_sequence(
    operations: List[Dict[str, Any]],
    *,
    validate_animations: bool,
) -> None:
    """Validate sequencing and animation compatibility for operations."""
    selection = _SelectionState()
    interpolator = AnimationInterpolator()

    for op in operations:
        if not isinstance(op, dict):
            continue

        op_type = op.get("type")
        if op_type == "select":
            selection = _selection_state_from_op(op)
            if op.get("animation"):
                raise ValidationError("select operations do not support animation")
            continue

        if op_type == "label" and selection.target != "mask":
            raise ValidationError("label operations require an object selection (target='mask')")

        if not validate_animations or not op.get("animation"):
            continue

        mode = (op.get("animation") or {}).get("mode", "transition")
        if op_type in _NON_GEOMETRIC_ANIMATION_TYPES and mode in {"draw", "reveal", "stagger", "splash"}:
            raise ValidationError(
                f"{op_type} does not support the '{mode}' animation mode"
            )

        if mode == "draw" and op_type not in _DRAW_COMPATIBLE_TYPES:
            raise ValidationError(
                f"draw animation is only supported for outline and bounding_box, not {op_type}"
            )

        if mode == "stagger":
            if selection.target != "mask":
                raise ValidationError(
                    f"stagger animation requires target='mask'; current target is '{selection.target}'"
                )
            if op_type in _NON_STAGGERABLE_TYPES:
                raise ValidationError(f"{op_type} does not support stagger animation")

        if mode in {"reveal", "splash"} and op_type in _NON_GEOMETRIC_ANIMATION_TYPES:
            raise ValidationError(f"{op_type} does not support the '{mode}' animation mode")

        if mode in {"transition", "pulse", "loop", "stagger"}:
            description = interpolator.describe(op)
            if not description["animated_keys"]:
                raise ValidationError(
                    f"{op_type} animation does not resolve any animatable parameters. "
                    "Include numeric parameters on the operation or explicit animation.start/animation.end values."
                )


def _selection_state_from_op(op: Dict[str, Any]) -> _SelectionState:
    """Resolve selection routing state from a select operation."""
    explicit_object_scope = any(op.get(key) is not None for key in _OBJECT_SCOPE_KEYS)
    return _SelectionState(
        target=op.get("target", "mask"),
        explicit_object_scope=explicit_object_scope,
    )
