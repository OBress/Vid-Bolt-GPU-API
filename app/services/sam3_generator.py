"""SAM 3 Generator - Segment Anything Model 3 service.

Provides image segmentation (text + visual prompts) and video object tracking
using Meta's SAM 3 model (848M params, ~4-10GB VRAM).

Implements the Segmenter interface for integration with the ModelManager.

API Reference (from sam3/model/sam3_image_processor.py):
  - Sam3Processor.set_image(image) -> state dict
  - Sam3Processor.set_text_prompt(prompt, state) -> state (with state["masks"], state["boxes"], state["scores"])
  - Sam3Processor.add_geometric_prompt(box, label, state) -> state  (box in normalized [cx, cy, w, h])
  - Sam3Processor.set_confidence_threshold(threshold, state) -> state
  - Sam3Processor.reset_all_prompts(state)

Video API (from sam3/model/sam3_base_predictor.py):
  - handle_request({"type": "start_session", "resource_path": ...}) -> {"session_id": ...}
  - handle_request({"type": "add_prompt", "session_id": ..., "frame_index": 0, "text": ...,
      "points": ..., "point_labels": ..., "bounding_boxes": ..., "bounding_box_labels": ...})
  - handle_stream_request({"type": "propagate_in_video", "session_id": ...,
      "propagation_direction": "forward"|"backward"|"both"}) -> yields per-frame results
  - handle_request({"type": "close_session", "session_id": ...})
"""

import asyncio
import base64
import gc
import importlib.util
import io
import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.models.internal import (
    ImageAnimationParams,
    ImageAnimationResult,
    ImageSegmentationParams,
    ImageSegmentationResult,
    VideoSegmentationParams,
    VideoSegmentationResult,
)
from app.services.interfaces import Segmenter
from app.utils.video_encoding import encode_mp4_h264

logger = logging.getLogger(__name__)


class SAM3Generator(Segmenter):
    """SAM 3 segmentation generator for images and videos.
    
    Wraps the sam3 library's image model and video predictor to provide
    unified segmentation capabilities.
    
    Attributes:
        _image_model: SAM 3 image model instance
        _image_processor: SAM 3 image processor for prompting
        _video_predictor: SAM 3 video predictor for tracking
        _is_loaded: Whether models are loaded
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None
        self._is_loaded = False
        self._dry_run = settings.sam3_dry_run
        self._max_objects = settings.sam3_max_objects
        self._max_frames = settings.sam3_video_max_frames
        self._device = settings.sam3_device
        self._image_model_version = settings.sam3_image_model_version
        self._video_model_version = settings.sam3_video_model_version
        self._video_use_fa3: Optional[bool] = None

    @property
    def _loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded

    @staticmethod
    def _resolve_video_use_fa3() -> Tuple[bool, str]:
        """Decide whether SAM 3 video should use FA3 on this runtime."""
        if importlib.util.find_spec("flash_attn_interface") is None:
            return False, "flash_attn_interface is not installed"

        try:
            import torch
        except ImportError:
            return False, "torch is unavailable during FA3 capability detection"

        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return False, "CUDA is not available"

        try:
            device_name = cuda.get_device_name(0)
        except Exception:
            device_name = "unknown GPU"

        if "blackwell" in device_name.lower():
            return False, f"detected Blackwell GPU ({device_name})"

        return True, f"flash_attn_interface is installed for GPU {device_name}"

    def load_models(self) -> None:
        """Load SAM 3 image and video models."""
        if self._is_loaded:
            logger.info("SAM 3 models already loaded")
            return

        if self._dry_run:
            logger.info("SAM 3: Dry run mode - skipping model loading")
            self._is_loaded = True
            return

        try:
            from sam3.model_builder import build_sam3_image_model, build_sam3_predictor, download_ckpt_from_hf
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info(f"Loading SAM image model ({self._image_model_version})...")
            image_model_kwargs = {}
            if self._image_model_version == "sam3.1":
                image_model_kwargs["checkpoint_path"] = download_ckpt_from_hf(version="sam3.1")
                image_model_kwargs["load_from_HF"] = False
            elif self._image_model_version != "sam3":
                raise ValueError(
                    f"Unsupported SAM image model version: {self._image_model_version!r}"
                )
            self._image_model = build_sam3_image_model(**image_model_kwargs)
            self._image_processor = Sam3Processor(self._image_model)
            logger.info(f"SAM image model loaded ({self._image_model_version})")

            logger.info(f"Loading SAM video predictor ({self._video_model_version})...")
            self._video_use_fa3, fa3_reason = self._resolve_video_use_fa3()
            if self._video_use_fa3:
                logger.info(
                    "Using FlashAttention 3 for SAM video predictor: %s",
                    fa3_reason,
                )
            else:
                logger.info(
                    "Using PyTorch SDPA for SAM video predictor: %s",
                    fa3_reason,
                )
            self._video_predictor = build_sam3_predictor(
                version=self._video_model_version,
                use_fa3=self._video_use_fa3,
            )
            logger.info(f"SAM video predictor loaded ({self._video_model_version})")

            self._is_loaded = True
            logger.info(
                f"SAM models fully loaded (image={self._image_model_version}, "
                f"video={self._video_model_version})"
            )

        except ImportError as e:
            logger.error(f"SAM 3 package not installed: {e}")
            logger.error("Install with: pip install -e repos/sam3")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM 3 models: {e}")
            raise

    def unload_models(self) -> None:
        """Unload SAM 3 models and free VRAM."""
        logger.info("Unloading SAM 3 models...")

        # Shutdown video predictor sessions
        if self._video_predictor is not None:
            try:
                self._video_predictor.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down video predictor: {e}")

        self._image_model = None
        self._image_processor = None
        self._video_predictor = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        self._is_loaded = False
        logger.info("SAM 3 models unloaded")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the SAM 3 generator."""
        return {
            "model": "sam3",
            "loaded": self._is_loaded,
            "dry_run": self._dry_run,
            "device": self._device,
            "max_objects": self._max_objects,
            "max_frames": self._max_frames,
            "image_model_version": self._image_model_version,
            "video_model_version": self._video_model_version,
            "video_use_fa3": self._video_use_fa3,
            "video_attention_backend": "fa3" if self._video_use_fa3 else "torch_sdpa",
        }

    @staticmethod
    def _merge_warnings(*warning_lists: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Combine warning lists while preserving order."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for warnings in warning_lists:
            if not warnings:
                continue
            for warning in warnings:
                warning_key = tuple(sorted((key, repr(value)) for key, value in warning.items()))
                if warning_key in seen:
                    continue
                seen.add(warning_key)
                merged.append(warning)
        return merged or None

    # --- Image Segmentation ---

    async def segment_image(self, params: ImageSegmentationParams) -> ImageSegmentationResult:
        """Segment objects in an image using text or visual prompts."""
        if self._dry_run:
            return self._mock_image_segmentation(params)

        return await asyncio.to_thread(self._segment_image_sync, params)

    def _segment_image_sync(self, params: ImageSegmentationParams) -> ImageSegmentationResult:
        """Synchronous image segmentation (runs in thread).
        
        Supports all SAM 3 image prompting modes:
          - Text prompt: open-vocabulary detection
          - Box prompts (simple): all positive
          - Box prompts (labeled): positive/negative include/exclude
          - Point prompts: via tiny geometric boxes
          - Configurable confidence threshold
        """
        import torch
        from PIL import Image

        # Load image from bytes
        image = Image.open(io.BytesIO(params.input_image_data)).convert("RGB")
        width, height = image.size

        # Set confidence threshold if different from default
        if params.confidence_threshold != 0.5:
            self._image_processor.set_confidence_threshold(params.confidence_threshold)
        else:
            # Reset to default in case it was changed by a previous request
            self._image_processor.set_confidence_threshold(0.5)

        # Set up the image in the processor (returns state dict with backbone features)
        inference_state = self._image_processor.set_image(image)

        masks_list = []
        boxes_list = []
        scores_list = []
        labels_list = []  # Track which label each mask belongs to

        # === Named object prompts (per-object targeting) ===
        if params.object_prompts:
            logger.info(f"Running {len(params.object_prompts)} named object prompts")
            for obj_prompt in params.object_prompts:
                obj_label = obj_prompt["label"]
                obj_text = obj_prompt["text"]
                logger.info(f"  Segmenting '{obj_label}': '{obj_text}'")

                self._image_processor.reset_all_prompts(inference_state)
                inference_state = self._image_processor.set_text_prompt(
                    prompt=obj_text,
                    state=inference_state,
                )

                # Track how many masks we had before this prompt
                prev_count = len(masks_list)
                self._extract_masks_from_state(
                    inference_state, width, height, params.max_objects,
                    masks_list, boxes_list, scores_list,
                )
                # Tag all new masks with this label
                new_count = len(masks_list) - prev_count
                labels_list.extend([obj_label] * new_count)
                logger.info(f"  Found {new_count} objects for '{obj_label}'")

        # === Text prompt segmentation ===
        elif params.text_prompt:
            logger.info(f"Running text prompt segmentation: '{params.text_prompt}'")
            self._image_processor.reset_all_prompts(inference_state)
            inference_state = self._image_processor.set_text_prompt(
                prompt=params.text_prompt,
                state=inference_state,
            )
            self._extract_masks_from_state(inference_state, width, height, params.max_objects,
                                           masks_list, boxes_list, scores_list)

        # === Labeled box prompts (positive/negative) ===
        elif params.box_prompts_labeled:
            logger.info(f"Running labeled box prompt segmentation with {len(params.box_prompts_labeled)} boxes")
            self._image_processor.reset_all_prompts(inference_state)

            for (box_xyxy, label) in params.box_prompts_labeled[:params.max_objects]:
                norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=label,
                )

            self._extract_masks_from_state(inference_state, width, height, params.max_objects,
                                           masks_list, boxes_list, scores_list)

        # === Simple box prompts (all positive) ===
        elif params.box_prompts:
            logger.info(f"Running box prompt segmentation with {len(params.box_prompts)} boxes")
            self._image_processor.reset_all_prompts(inference_state)

            for box_xyxy in params.box_prompts[:params.max_objects]:
                norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=True,
                )

            self._extract_masks_from_state(inference_state, width, height, params.max_objects,
                                           masks_list, boxes_list, scores_list)

        # === Point prompts (via tiny geometric boxes) ===
        elif params.point_prompts:
            logger.info(f"Running point prompt segmentation with {len(params.point_prompts)} points")
            self._image_processor.reset_all_prompts(inference_state)

            for point in params.point_prompts[:params.max_objects]:
                px, py = point
                # Create small normalized box around the point (2% of image size)
                norm_box = [px / width, py / height, 0.02, 0.02]
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=True,
                )

            self._extract_masks_from_state(inference_state, width, height, params.max_objects,
                                           masks_list, boxes_list, scores_list)

        # === Apply effects pipeline or return raw masks ===
        if params.output_type == "image" and params.operations:
            logger.info(f"Applying {len(params.operations)} visual operations")
            from app.services.segmentation_effects import EffectsPipeline

            # Get raw masks as numpy arrays for effects pipeline
            raw_masks = self._get_raw_masks_from_state(inference_state)

            # For object_prompts, we need to re-gather raw masks from each prompt
            if params.object_prompts:
                raw_masks = self._get_raw_masks_multi_prompt(params, image, inference_state)

            pipeline = EffectsPipeline(
                image,
                raw_masks,
                boxes=boxes_list,
                labels=labels_list if labels_list else None,
                annotation_state={},
            )
            pipeline.apply(params.operations)
            processed_bytes = pipeline.to_bytes(format="png")
            warnings = self._merge_warnings(params.operation_warnings, pipeline.warnings)

            logger.info(f"Effects applied: {len(processed_bytes)} bytes processed image")
            return ImageSegmentationResult(
                masks_data=processed_bytes,
                boxes=boxes_list,
                scores=scores_list,
                object_count=len(masks_list),
                width=width,
                height=height,
                content_type="image/png",
                labels=labels_list if labels_list else None,
                warnings=warnings,
                model_version=self._image_model_version,
            )
        else:
            # Default: return raw masks as base64 JSON
            encoded_masks = []
            for mask_bytes in masks_list:
                encoded_masks.append(base64.b64encode(mask_bytes).decode("utf-8"))

            masks_json = json.dumps(encoded_masks).encode("utf-8")
            logger.info(f"Segmentation complete: {len(encoded_masks)} masks, {len(masks_json)} bytes output")

            return ImageSegmentationResult(
                masks_data=masks_json,
                boxes=boxes_list,
                scores=scores_list,
                object_count=len(masks_list),
                width=width,
                height=height,
                content_type="application/json",
                labels=labels_list if labels_list else None,
                warnings=self._merge_warnings(params.operation_warnings),
                model_version=self._image_model_version,
            )

    def _xyxy_to_norm_cxcywh(self, box_xyxy, width: int, height: int) -> list:
        """Convert [x1, y1, x2, y2] pixel coords to normalized [cx, cy, w, h] for SAM 3."""
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        return [cx / width, cy / height, bw / width, bh / height]

    def _extract_masks_from_state(self, state, width, height, max_objects,
                                   masks_list, boxes_list, scores_list):
        """Extract masks, boxes, and scores from SAM 3 inference state dict."""
        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        if masks is not None and len(masks) > 0:
            count = min(len(masks), max_objects)
            logger.info(f"Found {len(masks)} objects, keeping {count}")

            for i in range(count):
                mask_png = self._mask_to_png(masks[i], width, height)
                masks_list.append(mask_png)

                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    boxes_list.append(tuple(int(v) for v in box[:4]))
                else:
                    boxes_list.append((0, 0, width, height))

                if scores is not None and i < len(scores):
                    score = scores[i]
                    if hasattr(score, 'item'):
                        score = score.item()
                    scores_list.append(float(score))
                else:
                    scores_list.append(1.0)
        else:
            logger.info("No objects found matching prompts")

    def _get_raw_masks_from_state(self, state) -> list:
        """Extract raw mask tensors/arrays from SAM 3 inference state for effects pipeline."""
        import numpy as np

        masks = state.get("masks")
        if masks is None:
            return []

        raw_masks = []
        for mask in masks:
            if hasattr(mask, 'cpu'):
                m = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray):
                m = mask
            else:
                m = np.array(mask)
            while m.ndim > 2:
                m = m.squeeze(0)
            raw_masks.append(m)
        return raw_masks

    def _get_raw_masks_multi_prompt(self, params, image, inference_state) -> list:
        """Re-run each object prompt to collect raw masks for the effects pipeline.
        
        When using object_prompts, we need to re-run each text prompt to gather
        the raw mask arrays (as opposed to the PNG-encoded versions already collected).
        """
        import numpy as np

        all_raw_masks = []
        for obj_prompt in params.object_prompts:
            self._image_processor.reset_all_prompts(inference_state)
            inference_state = self._image_processor.set_text_prompt(
                prompt=obj_prompt["text"],
                state=inference_state,
            )
            masks = inference_state.get("masks")
            if masks is not None:
                for mask in masks[:params.max_objects]:
                    if hasattr(mask, 'cpu'):
                        m = mask.cpu().numpy()
                    elif isinstance(mask, np.ndarray):
                        m = mask
                    else:
                        m = np.array(mask)
                    while m.ndim > 2:
                        m = m.squeeze(0)
                    all_raw_masks.append(m)
        return all_raw_masks

    def _mask_to_png(self, mask, width: int, height: int) -> bytes:
        """Convert a mask tensor/array to PNG bytes."""
        import numpy as np
        from PIL import Image

        # Handle different mask formats
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            mask_np = np.array(mask)

        # Squeeze extra dimensions (masks can be [1, 1, H, W] or [1, H, W])
        while mask_np.ndim > 2:
            mask_np = mask_np.squeeze(0)

        # Convert to uint8 binary mask (0 or 255)
        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255

        # Create PIL image and save as PNG
        mask_img = Image.fromarray(mask_uint8, mode="L")
        if mask_img.size != (width, height):
            mask_img = mask_img.resize((width, height), Image.NEAREST)

        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        return buffer.getvalue()

    def _mask_data_to_numpy(self, mask_data) -> "np.ndarray":
        """Convert predictor output mask data into a 2D numpy array."""
        import numpy as np

        if hasattr(mask_data, "cpu"):
            mask_np = mask_data.cpu().numpy()
        else:
            mask_np = np.array(mask_data)

        while mask_np.ndim > 2:
            mask_np = mask_np.squeeze(0)
        return mask_np

    def _mask_numpy_to_base64(self, mask_np) -> str:
        """Encode a binary mask array as a base64 PNG."""
        from PIL import Image

        mask_uint8 = (mask_np > 0.5).astype("uint8") * 255
        mask_img = Image.fromarray(mask_uint8, mode="L")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _coerce_output_sequence(value) -> List[Any]:
        """Normalize SAM outputs into a plain Python list.

        SAM sometimes returns numpy arrays for fields like `out_obj_ids`. Using
        Python truthiness on those arrays raises `ValueError`, so we normalize
        everything explicitly instead of relying on `or []`.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if converted is None:
                return []
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]

        try:
            return list(value)
        except TypeError:
            return [value]

    def _video_box_xywh_to_xyxy_pixels(
        self,
        box_xywh,
        frame_width: int,
        frame_height: int,
    ) -> List[int]:
        """Convert normalized [x, y, w, h] video boxes to pixel [x1, y1, x2, y2]."""
        x, y, w, h = [float(v) for v in box_xywh]
        x1 = int(round(x * frame_width))
        y1 = int(round(y * frame_height))
        x2 = int(round((x + w) * frame_width))
        y2 = int(round((y + h) * frame_height))
        return [x1, y1, x2, y2]

    # --- Video Segmentation ---

    async def segment_video(self, params: VideoSegmentationParams) -> VideoSegmentationResult:
        """Track and segment objects across video frames."""
        if self._dry_run:
            return self._mock_video_segmentation(params)

        return await asyncio.to_thread(self._segment_video_sync_v2, params)

    def _segment_video_sync(self, params: VideoSegmentationParams) -> VideoSegmentationResult:
        """Synchronous video segmentation (runs in thread).
        
        Supports all SAM 3 video prompting modes:
          - Text prompt: open-vocabulary object detection + tracking
          - Point prompts: click coordinates on initial frame
          - Box prompts: bounding boxes on initial frame
          - Configurable propagation direction (forward/backward/both)
          - Configurable confidence threshold
          - Configurable prompt frame index
        """
        import numpy as np

        # Write video bytes to temp file for SAM 3 video predictor
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(params.input_video_data)
            tmp_path = tmp.name

        session_id = None
        try:
            # 1. Start a video session
            logger.info(f"Starting video session for: {tmp_path}")
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=tmp_path,
                )
            )
            session_id = response["session_id"]
            logger.info(f"Video session started: {session_id}")

            # 2. Add prompts on the specified frame
            frame_index = params.prompt_frame_index
            add_prompt_request = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                output_prob_thresh=params.confidence_threshold,
            )

            # Add text prompt if provided
            if params.text_prompt:
                add_prompt_request["text"] = params.text_prompt
                logger.info(f"Adding text prompt: '{params.text_prompt}' on frame {frame_index}")

            # Add point prompts if provided
            if params.point_prompts:
                add_prompt_request["points"] = params.point_prompts
                add_prompt_request["point_labels"] = params.point_labels or [1] * len(params.point_prompts)
                logger.info(f"Adding {len(params.point_prompts)} point prompts on frame {frame_index}")

            # Add box prompts if provided
            if params.box_prompts:
                add_prompt_request["bounding_boxes"] = params.box_prompts
                add_prompt_request["bounding_box_labels"] = params.box_labels or [1] * len(params.box_prompts)
                logger.info(f"Adding {len(params.box_prompts)} box prompts on frame {frame_index}")

            response = self._video_predictor.handle_request(request=add_prompt_request)
            logger.info(f"Prompts added on frame {frame_index}")

            # 3. Propagate through video frames (streaming API)
            tracked_ids = set()
            frame_count = 0

            propagation_direction = params.propagation_direction
            wants_video_output = params.output_format == "video" and params.operations
            logger.info(f"Propagating {propagation_direction} through video frames (max {params.max_frames})...")

            if wants_video_output:
                # === Video output mode: apply effects per-frame and encode MP4 ===
                import cv2
                from app.services.segmentation_effects import apply_effects_to_frame

                # Open source video to read frames
                cap = cv2.VideoCapture(tmp_path)
                source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"Source video: {frame_width}x{frame_height} @ {source_fps} FPS, {total_frames} frames")

                # Read all source frames into memory (bounded by max_frames)
                source_frames = {}
                for fi in range(min(total_frames, params.max_frames)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    source_frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()

                # Collect per-frame masks from SAM 3
                per_frame_masks = {}
                for frame_output in self._video_predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction=propagation_direction,
                        max_frame_num_to_track=params.max_frames,
                        output_prob_thresh=params.confidence_threshold,
                    )
                ):
                    frame_idx = frame_output["frame_index"]
                    outputs = frame_output.get("outputs", {})
                    frame_count += 1

                    if frame_count > params.max_frames:
                        break

                    masks_for_frame = []
                    for obj_id_key, mask_data in outputs.items():
                        obj_id = int(obj_id_key) if isinstance(obj_id_key, str) else obj_id_key
                        tracked_ids.add(obj_id)

                        if hasattr(mask_data, 'cpu'):
                            mask_np = mask_data.cpu().numpy()
                        else:
                            mask_np = np.array(mask_data)
                        while mask_np.ndim > 2:
                            mask_np = mask_np.squeeze(0)
                        masks_for_frame.append(mask_np)

                    per_frame_masks[frame_idx] = masks_for_frame

                # Check if any operations have animation configs
                has_animation = params.operations and any(
                    isinstance(op, dict) and "animation" in op for op in params.operations
                )
                total_video_frames = len(source_frames)
                video_duration = total_video_frames / max(1, source_fps)

                def iter_processed_frames():
                    for fi in sorted(source_frames.keys()):
                        frame_rgb = source_frames[fi]
                        masks = per_frame_masks.get(fi, [])
                        if masks and params.operations:
                            # Interpolate operations temporally if they have animation configs
                            if has_animation:
                                from app.services.segmentation_animation import interpolate_video_operations
                                frame_ops = interpolate_video_operations(
                                    params.operations, fi, total_video_frames, video_duration
                                )
                            else:
                                frame_ops = params.operations
                            yield apply_effects_to_frame(frame_rgb, masks, frame_ops)
                        else:
                            yield frame_rgb

                result_data, encode_info = encode_mp4_h264(iter_processed_frames(), source_fps)

                logger.info(
                    f"Video effects applied: {frame_count} frames, {len(result_data)} bytes output",
                    extra={
                        "codec": encode_info["codec"],
                        "pixel_format": encode_info["pixel_format"],
                        "movflags": encode_info["movflags"],
                    },
                )

                return VideoSegmentationResult(
                    result_data=result_data,
                    output_format="video",
                    frame_count=frame_count,
                    object_count=len(tracked_ids),
                    tracked_ids=sorted(tracked_ids),
                    prompt_to_obj_ids={},
                    object_id_to_prompt_label={},
                    model_version=self._video_model_version,
                )

            else:
                # === Default: masks_json output ===
                frame_results = {}

                for frame_output in self._video_predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction=propagation_direction,
                        max_frame_num_to_track=params.max_frames,
                        output_prob_thresh=params.confidence_threshold,
                    )
                ):
                    frame_idx = frame_output["frame_index"]
                    outputs = frame_output.get("outputs", {})
                    frame_count += 1

                    if frame_count > params.max_frames:
                        break

                    frame_masks = {}
                    for obj_id_key, mask_data in outputs.items():
                        obj_id = int(obj_id_key) if isinstance(obj_id_key, str) else obj_id_key
                        tracked_ids.add(obj_id)

                        if hasattr(mask_data, 'cpu'):
                            mask_np = mask_data.cpu().numpy()
                        else:
                            mask_np = np.array(mask_data)

                        while mask_np.ndim > 2:
                            mask_np = mask_np.squeeze(0)

                        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                        from PIL import Image
                        mask_img = Image.fromarray(mask_uint8, mode="L")
                        buf = io.BytesIO()
                        mask_img.save(buf, format="PNG")
                        frame_masks[str(obj_id)] = base64.b64encode(buf.getvalue()).decode("utf-8")

                    frame_results[str(frame_idx)] = frame_masks

                logger.info(f"Video segmentation complete: {frame_count} frames, {len(tracked_ids)} tracked objects")

                result_json = json.dumps({
                    "frames": frame_results,
                    "tracked_ids": sorted(tracked_ids),
                    "frame_count": frame_count,
                    "text_prompt": params.text_prompt,
                    "propagation_direction": propagation_direction,
                }).encode("utf-8")

                return VideoSegmentationResult(
                    result_data=result_json,
                    output_format=params.output_format,
                    frame_count=frame_count,
                    object_count=len(tracked_ids),
                    tracked_ids=sorted(tracked_ids),
                    prompt_to_obj_ids={},
                    object_id_to_prompt_label={},
                    model_version=self._video_model_version,
                )

        finally:
            # 4. Close the session to free GPU memory
            if session_id is not None:
                try:
                    self._video_predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=session_id,
                        )
                    )
                    logger.info(f"Closed video session: {session_id}")
                except Exception as e:
                    logger.warning(f"Error closing video session {session_id}: {e}")

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # --- Image Animation (Image→Video) ---

    def _segment_video_sync_v2(self, params: VideoSegmentationParams) -> VideoSegmentationResult:
        """Video segmentation with stable API-level object IDs and prompt mappings."""
        import cv2

        from app.services.segmentation_effects import apply_effects_to_frame

        prompt_specs = params.object_prompts or []
        if not prompt_specs:
            if params.text_prompt:
                prompt_specs = [{"label": params.text_prompt, "text": params.text_prompt}]
            else:
                prompt_specs = [{"label": "visual_prompt", "text": None}]

        wants_video_output = params.output_format == "video" and params.operations

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(params.input_video_data)
            tmp_path = tmp.name

        source_fps = 30.0
        frame_width = 0
        frame_height = 0
        total_frames = 0
        source_frames = {}

        try:
            cap = cv2.VideoCapture(tmp_path)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if wants_video_output:
                for fi in range(min(total_frames, params.max_frames)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    source_frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()

            logger.info(
                f"Video source loaded for segmentation: {frame_width}x{frame_height} @ "
                f"{source_fps} FPS, {total_frames} frames"
            )

            prompt_to_obj_ids: Dict[str, List[int]] = {
                spec["label"]: [] for spec in prompt_specs
            }
            object_id_to_prompt_label: Dict[int, str] = {}
            frame_records: Dict[int, Dict[int, Dict[str, Any]]] = {}
            tracked_ids = set()
            next_global_id = 1

            def assign_global_id(local_obj_id: int, prompt_label: str) -> int:
                nonlocal next_global_id
                global_id = next_global_id
                next_global_id += 1
                prompt_to_obj_ids.setdefault(prompt_label, []).append(global_id)
                object_id_to_prompt_label[global_id] = prompt_label
                logger.info(
                    f"Mapped local SAM object {local_obj_id} to API object {global_id} "
                    f"for prompt '{prompt_label}'"
                )
                return global_id

            for prompt_spec in prompt_specs:
                session_id = None
                local_to_global: Dict[int, int] = {}
                prompt_label = prompt_spec["label"]
                prompt_text = prompt_spec.get("text")

                try:
                    start_response = self._video_predictor.handle_request(
                        request=dict(
                            type="start_session",
                            resource_path=tmp_path,
                        )
                    )
                    session_id = start_response["session_id"]
                    logger.info(f"Video session started for prompt '{prompt_label}': {session_id}")

                    add_prompt_request = dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=params.prompt_frame_index,
                        output_prob_thresh=params.confidence_threshold,
                    )

                    if prompt_text:
                        add_prompt_request["text"] = prompt_text
                    if params.point_prompts:
                        add_prompt_request["points"] = params.point_prompts
                        add_prompt_request["point_labels"] = params.point_labels or [1] * len(params.point_prompts)
                    if params.box_prompts:
                        add_prompt_request["bounding_boxes"] = params.box_prompts
                        add_prompt_request["bounding_box_labels"] = params.box_labels or [1] * len(params.box_prompts)

                    add_prompt_response = self._video_predictor.handle_request(request=add_prompt_request)
                    prompt_outputs = add_prompt_response.get("outputs") or {}
                    for local_obj_id in self._coerce_output_sequence(
                        prompt_outputs.get("out_obj_ids")
                    ):
                        local_id = int(local_obj_id)
                        local_to_global[local_id] = assign_global_id(local_id, prompt_label)

                    for frame_output in self._video_predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                            propagation_direction=params.propagation_direction,
                            max_frame_num_to_track=params.max_frames,
                            output_prob_thresh=params.confidence_threshold,
                        )
                    ):
                        frame_idx = frame_output["frame_index"]
                        outputs = frame_output.get("outputs") or {}

                        out_obj_ids = self._coerce_output_sequence(outputs.get("out_obj_ids"))
                        out_masks = self._coerce_output_sequence(outputs.get("out_binary_masks"))
                        out_scores = self._coerce_output_sequence(outputs.get("out_probs"))
                        out_boxes = self._coerce_output_sequence(outputs.get("out_boxes_xywh"))

                        frame_bucket = frame_records.setdefault(frame_idx, {})
                        for idx, local_obj_id in enumerate(out_obj_ids):
                            local_id = int(local_obj_id)
                            global_id = local_to_global.get(local_id)
                            if global_id is None:
                                global_id = assign_global_id(local_id, prompt_label)
                                local_to_global[local_id] = global_id

                            tracked_ids.add(global_id)

                            mask_np = self._mask_data_to_numpy(out_masks[idx])
                            score = float(out_scores[idx]) if idx < len(out_scores) else 1.0
                            if idx < len(out_boxes):
                                box_xyxy = self._video_box_xywh_to_xyxy_pixels(
                                    out_boxes[idx], frame_width, frame_height
                                )
                            else:
                                box_xyxy = [0, 0, frame_width, frame_height]

                            frame_bucket[global_id] = {
                                "mask": mask_np,
                                "box": box_xyxy,
                                "score": score,
                                "label": prompt_label,
                            }

                finally:
                    if session_id is not None:
                        try:
                            self._video_predictor.handle_request(
                                request=dict(
                                    type="close_session",
                                    session_id=session_id,
                                )
                            )
                            logger.info(f"Closed prompt-scoped video session: {session_id}")
                        except Exception as e:
                            logger.warning(f"Error closing prompt-scoped session {session_id}: {e}")

            frame_count = len(frame_records)
            tracked_ids_sorted = sorted(tracked_ids)

            if wants_video_output:
                has_animation = params.operations and any(
                    isinstance(op, dict) and "animation" in op for op in params.operations
                )
                total_video_frames = len(source_frames)
                video_duration = total_video_frames / max(1, source_fps)
                annotation_state: Dict[str, Any] = {}
                collected_warnings: List[Dict[str, Any]] = []

                def iter_processed_frames():
                    for fi in sorted(source_frames.keys()):
                        frame_rgb = source_frames[fi]
                        frame_objects = frame_records.get(fi, {})

                        if frame_objects and params.operations:
                            if has_animation:
                                from app.services.segmentation_animation import interpolate_video_operations

                                frame_ops = interpolate_video_operations(
                                    params.operations, fi, total_video_frames, video_duration
                                )
                            else:
                                frame_ops = params.operations

                            ordered_ids = sorted(frame_objects.keys())
                            masks = [frame_objects[obj_id]["mask"] for obj_id in ordered_ids]
                            boxes = [tuple(frame_objects[obj_id]["box"]) for obj_id in ordered_ids]
                            labels = [frame_objects[obj_id]["label"] for obj_id in ordered_ids]
                            yield apply_effects_to_frame(
                                frame_rgb,
                                masks,
                                frame_ops,
                                boxes=boxes,
                                labels=labels,
                                object_ids=ordered_ids,
                                annotation_state=annotation_state,
                                warnings_out=collected_warnings,
                            )
                        else:
                            yield frame_rgb

                result_data, encode_info = encode_mp4_h264(iter_processed_frames(), source_fps)
                warnings = self._merge_warnings(params.operation_warnings, collected_warnings)

                logger.info(
                    f"Video effects applied with stable object routing: "
                    f"{frame_count} tracked frames, {len(tracked_ids_sorted)} objects",
                    extra={
                        "codec": encode_info["codec"],
                        "pixel_format": encode_info["pixel_format"],
                        "movflags": encode_info["movflags"],
                    },
                )

                return VideoSegmentationResult(
                    result_data=result_data,
                    output_format="video",
                    frame_count=frame_count,
                    object_count=len(tracked_ids_sorted),
                    tracked_ids=tracked_ids_sorted,
                    prompt_to_obj_ids=prompt_to_obj_ids,
                    object_id_to_prompt_label=object_id_to_prompt_label,
                    warnings=warnings,
                    model_version=self._video_model_version,
                )

            frame_results: Dict[str, Dict[str, Any]] = {}
            for frame_idx, frame_objects in sorted(frame_records.items()):
                serialized_frame = {}
                for object_id, record in sorted(frame_objects.items()):
                    mask_b64 = self._mask_numpy_to_base64(record["mask"])
                    if params.include_tracking_metadata:
                        serialized_frame[str(object_id)] = {
                            "mask": mask_b64,
                            "box": record["box"],
                            "score": record["score"],
                            "label": record["label"],
                        }
                    else:
                        serialized_frame[str(object_id)] = mask_b64
                frame_results[str(frame_idx)] = serialized_frame

            result_json = json.dumps({
                "frames": frame_results,
                "tracked_ids": tracked_ids_sorted,
                "frame_count": frame_count,
                "prompt_to_obj_ids": prompt_to_obj_ids,
                "object_id_to_prompt_label": {str(k): v for k, v in object_id_to_prompt_label.items()},
                "propagation_direction": params.propagation_direction,
                "include_tracking_metadata": params.include_tracking_metadata,
                "model_version": self._video_model_version,
            }).encode("utf-8")

            return VideoSegmentationResult(
                result_data=result_json,
                output_format=params.output_format,
                frame_count=frame_count,
                object_count=len(tracked_ids_sorted),
                tracked_ids=tracked_ids_sorted,
                prompt_to_obj_ids=prompt_to_obj_ids,
                object_id_to_prompt_label=object_id_to_prompt_label,
                warnings=self._merge_warnings(params.operation_warnings),
                model_version=self._video_model_version,
            )

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def animate_image(self, params: ImageAnimationParams) -> ImageAnimationResult:
        """Generate an animated video from a segmented image.
        
        1. Run SAM 3 segmentation on the image (get masks)
        2. Pass image + masks + animated operations to AnimationPipeline
        3. AnimationPipeline renders N frames with interpolated effects
        4. Return MP4 bytes
        """
        if self._dry_run:
            return self._mock_image_animation(params)

        return await asyncio.to_thread(self._animate_image_sync, params)

    def _animate_image_sync(self, params: ImageAnimationParams) -> ImageAnimationResult:
        """Synchronous image animation (runs in thread)."""
        import numpy as np
        from PIL import Image
        from app.services.segmentation_animation import AnimationPipeline

        # 1. Open image
        image = Image.open(io.BytesIO(params.input_image_data)).convert("RGB")
        width, height = image.size
        logger.info(f"Animate: image loaded {width}x{height}")

        if params.confidence_threshold != 0.5:
            self._image_processor.set_confidence_threshold(params.confidence_threshold)
        else:
            self._image_processor.set_confidence_threshold(0.5)

        # 2. Run SAM 3 segmentation to get masks
        inference_state = self._image_processor.set_image(image)

        raw_masks: List[np.ndarray] = []
        boxes_list: List[Tuple[int, int, int, int]] = []
        labels_list: List[str] = []

        def collect_state_masks(state, label: Optional[str] = None) -> int:
            masks = state.get("masks")
            boxes = state.get("boxes")
            if masks is None or len(masks) == 0:
                return 0

            count = min(len(masks), params.max_objects)
            for i in range(count):
                raw_masks.append(self._mask_data_to_numpy(masks[i]))

                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, "tolist"):
                        box = box.tolist()
                    boxes_list.append(tuple(int(v) for v in box[:4]))
                else:
                    boxes_list.append((0, 0, width, height))

                if label is not None:
                    labels_list.append(label)

            return count

        if params.object_prompts:
            logger.info(f"Animate: running {len(params.object_prompts)} named object prompts")
            for obj_prompt in params.object_prompts:
                obj_label = obj_prompt["label"]
                obj_text = obj_prompt["text"]
                self._image_processor.reset_all_prompts(inference_state)
                prompt_state = self._image_processor.set_text_prompt(
                    prompt=obj_text,
                    state=inference_state,
                )
                found = collect_state_masks(prompt_state, label=obj_label)
                logger.info(f"Animate: found {found} objects for '{obj_label}'")
        elif params.text_prompt:
            self._image_processor.reset_all_prompts(inference_state)
            inference_state = self._image_processor.set_text_prompt(params.text_prompt, inference_state)
            collect_state_masks(inference_state)
        elif params.box_prompts_labeled:
            self._image_processor.reset_all_prompts(inference_state)
            for (box_xyxy, label) in params.box_prompts_labeled[:params.max_objects]:
                norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=label,
                )
            collect_state_masks(inference_state)
        elif params.box_prompts:
            self._image_processor.reset_all_prompts(inference_state)
            for box_xyxy in params.box_prompts[:params.max_objects]:
                norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=True,
                )
            collect_state_masks(inference_state)
        elif params.point_prompts:
            self._image_processor.reset_all_prompts(inference_state)
            for point in params.point_prompts[:params.max_objects]:
                px, py = point
                norm_box = [px / width, py / height, 0.02, 0.02]
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=True,
                )
            collect_state_masks(inference_state)

        object_count = len(raw_masks)
        logger.info(f"Animate: segmented {object_count} objects, rendering animation...")

        if object_count == 0:
            logger.warning("No objects found for animation, using full-frame mask")
            full_mask = np.ones((height, width), dtype=bool)
            raw_masks = [full_mask]
            boxes_list = [(0, 0, width, height)]

        # 3. Build and render animation
        pipeline = AnimationPipeline(
            image=image,
            masks=raw_masks,
            boxes=boxes_list,
            labels=labels_list if labels_list else None,
            fps=params.fps,
            duration=params.duration_seconds,
        )

        operations = params.operations or []
        mp4_bytes = pipeline.render(operations)
        warnings = self._merge_warnings(params.operation_warnings, getattr(pipeline, "warnings", None))

        total_frames = min(int(params.fps * params.duration_seconds), 600)
        logger.info(
            f"Animation complete: {total_frames} frames, "
            f"{len(mp4_bytes)} bytes MP4"
        )

        return ImageAnimationResult(
            video_data=mp4_bytes,
            width=width,
            height=height,
            duration_seconds=params.duration_seconds,
            fps=params.fps,
            frame_count=total_frames,
            object_count=object_count,
            labels=labels_list if labels_list else None,
            warnings=warnings,
            model_version=self._image_model_version,
        )

    def _mock_image_animation(self, params: ImageAnimationParams) -> ImageAnimationResult:
        """Generate mock animation result for dry-run testing."""
        from PIL import Image
        image = Image.open(io.BytesIO(params.input_image_data))
        width, height = image.size
        total_frames = min(int(params.fps * params.duration_seconds), 600)
        labels = [prompt["label"] for prompt in params.object_prompts] if params.object_prompts else None
        object_count = len(labels) if labels else 1

        return ImageAnimationResult(
            video_data=b"mock_mp4_data",
            width=width,
            height=height,
            duration_seconds=params.duration_seconds,
            fps=params.fps,
            frame_count=total_frames,
            object_count=object_count,
            labels=labels,
            warnings=self._merge_warnings(params.operation_warnings),
            model_version=self._image_model_version,
        )

    # --- Mock/Dry-Run Methods ---

    def _mock_image_segmentation(self, params: ImageSegmentationParams) -> ImageSegmentationResult:
        """Generate mock segmentation results for testing."""
        import numpy as np
        from PIL import Image
        from app.services.segmentation_effects import EffectsPipeline

        image = Image.open(io.BytesIO(params.input_image_data))
        width, height = image.size

        # Create a simple mock mask (center rectangle)
        mock_mask = Image.new("L", (width, height), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mock_mask)
        x1, y1 = width // 4, height // 4
        x2, y2 = 3 * width // 4, 3 * height // 4
        draw.rectangle([x1, y1, x2, y2], fill=255)

        buf = io.BytesIO()
        mock_mask.save(buf, format="PNG")
        mask_bytes = buf.getvalue()
        object_specs = params.object_prompts or [{"label": params.text_prompt or "object"}]
        raw_masks = [np.array(mock_mask) > 0 for _ in object_specs]
        boxes = [(x1, y1, x2, y2) for _ in object_specs]
        scores = [0.95 for _ in object_specs]
        labels = [spec["label"] for spec in object_specs] if params.object_prompts else None

        if params.output_type == "image" and params.operations:
            pipeline = EffectsPipeline(
                image.convert("RGB"),
                raw_masks,
                boxes=boxes,
                labels=labels,
                annotation_state={},
            )
            pipeline.apply(params.operations)
            result_bytes = pipeline.to_bytes(format="png")
            return ImageSegmentationResult(
                masks_data=result_bytes,
                boxes=boxes,
                scores=scores,
                object_count=len(raw_masks),
                width=width,
                height=height,
                content_type="image/png",
                labels=labels,
                warnings=self._merge_warnings(params.operation_warnings, pipeline.warnings),
                model_version=self._image_model_version,
            )

        encoded = [base64.b64encode(mask_bytes).decode("utf-8") for _ in object_specs]
        masks_json = json.dumps(encoded).encode("utf-8")

        return ImageSegmentationResult(
            masks_data=masks_json,
            boxes=boxes,
            scores=scores,
            object_count=len(object_specs),
            width=width,
            height=height,
            labels=labels,
            warnings=self._merge_warnings(params.operation_warnings),
            model_version=self._image_model_version,
        )

    def _mock_video_segmentation(self, params: VideoSegmentationParams) -> VideoSegmentationResult:
        """Generate mock video segmentation results for testing."""
        prompt_specs = params.object_prompts or []
        if not prompt_specs:
            if params.text_prompts:
                prompt_specs = [{"label": prompt, "text": prompt} for prompt in params.text_prompts]
            elif params.text_prompt:
                prompt_specs = [{"label": params.text_prompt, "text": params.text_prompt}]
            else:
                prompt_specs = [{"label": "visual_prompt", "text": None}]

        prompt_to_obj_ids: Dict[str, List[int]] = {}
        object_id_to_prompt_label: Dict[int, str] = {}
        for index, prompt_spec in enumerate(prompt_specs, start=1):
            prompt_to_obj_ids[prompt_spec["label"]] = [index]
            object_id_to_prompt_label[index] = prompt_spec["label"]

        if params.output_format == "video":
            return VideoSegmentationResult(
                result_data=b"mock_mp4_data",
                output_format="video",
                frame_count=2,
                object_count=len(prompt_specs),
                tracked_ids=sorted(object_id_to_prompt_label.keys()),
                prompt_to_obj_ids=prompt_to_obj_ids,
                object_id_to_prompt_label=object_id_to_prompt_label,
                warnings=self._merge_warnings(params.operation_warnings),
                model_version=self._video_model_version,
            )

        frame_payload: Dict[str, Dict[str, Any]] = {}
        for frame_idx in range(2):
            objects = {}
            for object_id, label in object_id_to_prompt_label.items():
                if params.include_tracking_metadata:
                    objects[str(object_id)] = {
                        "mask": "mock_mask_base64",
                        "box": [10 * object_id, 10 * object_id, 40, 40],
                        "score": 0.95,
                        "label": label,
                    }
                else:
                    objects[str(object_id)] = "mock_mask_base64"
            frame_payload[str(frame_idx)] = objects

        result_json = json.dumps({
            "frames": frame_payload,
            "tracked_ids": sorted(object_id_to_prompt_label.keys()),
            "frame_count": 2,
            "prompt_to_obj_ids": prompt_to_obj_ids,
            "object_id_to_prompt_label": {str(k): v for k, v in object_id_to_prompt_label.items()},
            "propagation_direction": params.propagation_direction,
            "include_tracking_metadata": params.include_tracking_metadata,
            "model_version": self._video_model_version,
        }).encode("utf-8")

        return VideoSegmentationResult(
            result_data=result_json,
            output_format=params.output_format,
            frame_count=2,
            object_count=len(prompt_specs),
            tracked_ids=sorted(object_id_to_prompt_label.keys()),
            prompt_to_obj_ids=prompt_to_obj_ids,
            object_id_to_prompt_label=object_id_to_prompt_label,
            warnings=self._merge_warnings(params.operation_warnings),
            model_version=self._video_model_version,
        )
