"""SAM 3 Generator - Segment Anything Model 3 service.

Provides image segmentation (text + visual prompts) and video object tracking
using Meta's SAM 3 model (848M params, ~4-10GB VRAM).

Implements the Segmenter interface for integration with the ModelManager.
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
            from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info("Loading SAM 3 image model...")
            self._image_model = build_sam3_image_model()
            self._image_processor = Sam3Processor(self._image_model)
            logger.info("SAM 3 image model loaded")

            logger.info("Loading SAM 3 video predictor...")
            self._video_predictor = build_sam3_video_predictor()
            logger.info("SAM 3 video predictor loaded")

            self._is_loaded = True
            logger.info("SAM 3 models fully loaded (~3.5 GB VRAM)")

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
        """Synchronous image segmentation (runs in thread)."""
        from PIL import Image

        # Load image from bytes
        image = Image.open(io.BytesIO(params.input_image_data))
        width, height = image.size

        # Set up the image in the processor
        inference_state = self._image_processor.set_image(image)

        masks_list = []
        boxes_list = []
        scores_list = []

        # Text prompt segmentation
        if params.text_prompt:
            output = self._image_processor.set_text_prompt(
                state=inference_state,
                prompt=params.text_prompt,
            )
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]

            # Limit to max_objects
            count = min(len(scores), params.max_objects)

            for i in range(count):
                # Convert mask to PNG bytes
                mask_png = self._mask_to_png(masks[i], width, height)
                masks_list.append(mask_png)
                # Convert box tensor to tuple
                box = boxes[i]
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                boxes_list.append(tuple(int(v) for v in box[:4]))
                score = scores[i]
                if hasattr(score, 'item'):
                    score = score.item()
                scores_list.append(float(score))

        # Box prompt segmentation
        elif params.box_prompts:
            for box in params.box_prompts[:params.max_objects]:
                output = self._image_processor.set_box_prompt(
                    state=inference_state,
                    box=list(box),
                )
                if output.get("masks") is not None and len(output["masks"]) > 0:
                    mask_png = self._mask_to_png(output["masks"][0], width, height)
                    masks_list.append(mask_png)
                    boxes_list.append(box)
                    score = output.get("scores", [1.0])[0]
                    if hasattr(score, 'item'):
                        score = score.item()
                    scores_list.append(float(score))

        # Point prompt segmentation
        elif params.point_prompts:
            for point in params.point_prompts[:params.max_objects]:
                output = self._image_processor.set_point_prompt(
                    state=inference_state,
                    point=list(point),
                    label=1,  # Positive point
                )
                if output.get("masks") is not None and len(output["masks"]) > 0:
                    mask_png = self._mask_to_png(output["masks"][0], width, height)
                    masks_list.append(mask_png)
                    # Derive box from mask if not provided
                    boxes_list.append((0, 0, width, height))
                    score = output.get("scores", [1.0])[0]
                    if hasattr(score, 'item'):
                        score = score.item()
                    scores_list.append(float(score))

        # Encode masks as base64 JSON
        encoded_masks = []
        for mask_bytes in masks_list:
            encoded_masks.append(base64.b64encode(mask_bytes).decode("utf-8"))

        masks_json = json.dumps(encoded_masks).encode("utf-8")

        return ImageSegmentationResult(
            masks_data=masks_json,
            boxes=boxes_list,
            scores=scores_list,
            object_count=len(masks_list),
            width=width,
            height=height,
        )

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

        # Squeeze extra dimensions
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
        """Synchronous video segmentation (runs in thread)."""
        import numpy as np
        
        # Write video bytes to temp file for SAM 3 video predictor
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(params.input_video_data)
            tmp_path = tmp.name

        try:
            # Start a video session
            response = self._video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=tmp_path,
                )
            )
            session_id = response["session_id"]

            # Add text prompt on the first frame
            response = self._video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=params.text_prompt,
                )
            )

            outputs = response.get("outputs", {})
            
            # Collect tracking results
            tracked_ids = set()
            frame_results = {}
            frame_count = 0

            # Process outputs from the propagation
            for frame_idx, frame_output in outputs.items():
                frame_count += 1
                if frame_count > params.max_frames:
                    break

                frame_masks = {}
                for obj_id, mask_data in frame_output.items():
                    tracked_ids.add(int(obj_id))
                    # Convert mask to base64 PNG
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

            result_json = json.dumps({
                "frames": frame_results,
                "tracked_ids": sorted(tracked_ids),
                "frame_count": frame_count,
                "text_prompt": params.text_prompt,
            }).encode("utf-8")

            return VideoSegmentationResult(
                result_data=result_json,
                output_format=params.output_format,
                frame_count=frame_count,
                object_count=len(tracked_ids),
                tracked_ids=sorted(tracked_ids),
            )

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

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
        }).encode("utf-8")

        return VideoSegmentationResult(
            result_data=result_json,
            output_format=params.output_format,
            frame_count=2,
            object_count=1,
            tracked_ids=[1],
        )
