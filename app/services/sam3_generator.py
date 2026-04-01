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

    @property
    def _loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded

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
            from sam3.model_builder import build_sam3_image_model, build_sam3_predictor
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info("Loading SAM 3 image model...")
            self._image_model = build_sam3_image_model()
            self._image_processor = Sam3Processor(self._image_model)
            logger.info("SAM 3 image model loaded")

            logger.info("Loading SAM 3.1 video predictor (Object Multiplex)...")
            self._video_predictor = build_sam3_predictor(version="sam3.1")
            logger.info("SAM 3.1 video predictor loaded (multiplex: up to 16 objects/pass)")

            self._is_loaded = True
            logger.info("SAM 3 / 3.1 models fully loaded (~3.5 GB VRAM)")

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
        }

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

            pipeline = EffectsPipeline(image, raw_masks, boxes=boxes_list, labels=labels_list if labels_list else None)
            pipeline.apply(params.operations)
            processed_bytes = pipeline.to_bytes(format="png")

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

    # --- Video Segmentation ---

    async def segment_video(self, params: VideoSegmentationParams) -> VideoSegmentationResult:
        """Track and segment objects across video frames."""
        if self._dry_run:
            return self._mock_video_segmentation(params)

        return await asyncio.to_thread(self._segment_video_sync, params)

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

                # Apply effects and encode MP4
                out_path = tmp_path + "_out.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, source_fps, (frame_width, frame_height))

                # Check if any operations have animation configs
                has_animation = params.operations and any(
                    isinstance(op, dict) and "animation" in op for op in params.operations
                )
                total_video_frames = len(source_frames)
                video_duration = total_video_frames / max(1, source_fps)

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
                        processed = apply_effects_to_frame(frame_rgb, masks, frame_ops)
                        out.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                    else:
                        out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                out.release()

                # Read the output video bytes
                with open(out_path, "rb") as f:
                    result_data = f.read()

                try:
                    os.unlink(out_path)
                except OSError:
                    pass

                logger.info(f"Video effects applied: {frame_count} frames, {len(result_data)} bytes output")

                return VideoSegmentationResult(
                    result_data=result_data,
                    output_format="video",
                    frame_count=frame_count,
                    object_count=len(tracked_ids),
                    tracked_ids=sorted(tracked_ids),
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

        # 2. Run SAM 3 segmentation to get masks
        inference_state = self._image_processor.set_image(image)

        if params.text_prompt:
            inference_state = self._image_processor.set_text_prompt(params.text_prompt, inference_state)
        elif params.box_prompts_labeled:
            for (box_xyxy, label) in params.box_prompts_labeled[:params.max_objects]:
                norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=label,
                )
        elif params.box_prompts:
            for box_xyxy in params.box_prompts[:params.max_objects]:
                norm_box = self._xyxy_to_norm_cxcywh(box_xyxy, width, height)
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=True,
                )
        elif params.point_prompts:
            for point in params.point_prompts[:params.max_objects]:
                px, py = point
                norm_box = [px / width, py / height, 0.02, 0.02]
                inference_state = self._image_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=True,
                )

        # Get raw masks and boxes
        raw_masks = self._get_raw_masks_from_state(inference_state)
        boxes_list = []
        boxes = inference_state.get("boxes")
        if boxes is not None:
            for box in boxes:
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                boxes_list.append(tuple(int(v) for v in box[:4]))

        object_count = len(raw_masks)
        logger.info(f"Animate: segmented {object_count} objects, rendering animation...")

        if object_count == 0:
            logger.warning("No objects found for animation, using full-frame mask")
            full_mask = np.ones((height, width), dtype=bool)
            raw_masks = [full_mask]

        # 3. Build and render animation
        pipeline = AnimationPipeline(
            image=image,
            masks=raw_masks,
            boxes=boxes_list,
            fps=params.fps,
            duration=params.duration_seconds,
        )

        operations = params.operations or []
        mp4_bytes = pipeline.render(operations)

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
        )

    def _mock_image_animation(self, params: ImageAnimationParams) -> ImageAnimationResult:
        """Generate mock animation result for dry-run testing."""
        from PIL import Image
        image = Image.open(io.BytesIO(params.input_image_data))
        width, height = image.size
        total_frames = min(int(params.fps * params.duration_seconds), 600)

        return ImageAnimationResult(
            video_data=b"mock_mp4_data",
            width=width,
            height=height,
            duration_seconds=params.duration_seconds,
            fps=params.fps,
            frame_count=total_frames,
            object_count=1,
        )

    # --- Mock/Dry-Run Methods ---

    def _mock_image_segmentation(self, params: ImageSegmentationParams) -> ImageSegmentationResult:
        """Generate mock segmentation results for testing."""
        from PIL import Image

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

        encoded = [base64.b64encode(mask_bytes).decode("utf-8")]
        masks_json = json.dumps(encoded).encode("utf-8")

        return ImageSegmentationResult(
            masks_data=masks_json,
            boxes=[(x1, y1, x2, y2)],
            scores=[0.95],
            object_count=1,
            width=width,
            height=height,
        )

    def _mock_video_segmentation(self, params: VideoSegmentationParams) -> VideoSegmentationResult:
        """Generate mock video segmentation results for testing."""
        result_json = json.dumps({
            "frames": {
                "0": {"1": "mock_mask_base64"},
                "1": {"1": "mock_mask_base64"},
            },
            "tracked_ids": [1],
            "frame_count": 2,
            "text_prompt": params.text_prompt,
            "propagation_direction": params.propagation_direction,
        }).encode("utf-8")

        return VideoSegmentationResult(
            result_data=result_json,
            output_format=params.output_format,
            frame_count=2,
            object_count=1,
            tracked_ids=[1],
        )
