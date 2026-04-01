"""Segmentation Effects Pipeline - Composable post-processing for SAM 3 masks.

Applies ordered visual operations to images using SAM 3 segmentation masks.
All operations are CPU-based (PIL/numpy/OpenCV) — no GPU required.

Usage:
    pipeline = EffectsPipeline(image, masks)
    result = pipeline.apply(operations)
    result_bytes = pipeline.to_bytes(format="png")
"""

import io
import logging
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
    ):
        """Initialize pipeline with source image and segmentation masks.
        
        Args:
            image: Source PIL image
            masks: List of binary masks (H, W), one per detected object
            boxes: Optional bounding boxes per object [(x1,y1,x2,y2)]
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

        # Store boxes for text labels
        self.boxes = boxes or []

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
                    self._op_select(op)
                elif op_type == "blur":
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
                    self._op_remove_background(op)
                elif op_type == "replace_background":
                    self._op_replace_background(op)
                elif op_type == "greenscreen":
                    self._op_greenscreen(op)
                elif op_type == "outline":
                    self._op_outline(op)
                elif op_type == "text_label":
                    pass  # Removed: text labels are no longer supported
                elif op_type == "bounding_box":
                    self._op_bounding_box(op)
                elif op_type == "spotlight":
                    self._op_spotlight(op)
                elif op_type == "bokeh":
                    self._op_bokeh(op)
                elif op_type == "glow":
                    self._op_glow(op)
                elif op_type == "shadow":
                    self._op_shadow(op)
                elif op_type == "vignette":
                    self._op_vignette(op)
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
                elif op_type == "zoom":
                    pass  # Handled by AnimationPipeline camera layer
                elif op_type == "pan":
                    pass  # Handled by AnimationPipeline camera layer
                else:
                    logger.warning(f"Unknown operation type: {op_type}, skipping")
            except Exception as e:
                logger.error(f"Error applying operation {op_type}: {e}")
                raise

        return self.image

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
        """Switch which region subsequent operations apply to."""
        target = op.get("target", "mask")
        self.active_target = target
        if target == "mask":
            self.active_mask = self.combined_mask.copy()
        elif target == "background":
            self.active_mask = ~self.combined_mask
        elif target == "all":
            self.active_mask = np.ones((self.height, self.width), dtype=bool)
        else:
            logger.warning(f"Unknown select target: {target}, defaulting to 'mask'")
            self.active_mask = self.combined_mask.copy()

    # =========================================================================
    # Blur / Privacy
    # =========================================================================

    def _op_blur(self, op: dict):
        """Apply Gaussian blur to the selected region (edge-aware).
        
        To prevent bleed from non-selected pixels into the blurred region,
        the non-selected area is filled with the average color of the selected
        region before blurring, then composited back.
        """
        strength = op.get("strength", 25)
        radius = max(1, int(strength))
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
        strength = max(1, min(50, int(op.get("strength", 15))))

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

    def _composite_with_mask(self, modified: Image.Image):
        """Replace pixels in self.image with modified image where active_mask is True."""
        img_arr = np.array(self.image)
        mod_arr = np.array(modified)
        mask = self.active_mask
        channels = img_arr.shape[2]
        mask_expanded = np.stack([mask] * channels, axis=-1)
        img_arr[mask_expanded] = mod_arr[mask_expanded]
        self.image = Image.fromarray(img_arr)

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
) -> np.ndarray:
    """Convenience function for video frame processing.
    
    Args:
        frame: BGR or RGB numpy array (H, W, 3)
        masks: List of binary masks
        operations: List of operation dicts
        boxes: Optional bounding boxes
        
    Returns:
        Processed frame as numpy array (H, W, 3) in RGB
    """
    # Convert numpy frame to PIL
    image = Image.fromarray(frame)

    pipeline = EffectsPipeline(image, masks, boxes)
    result = pipeline.apply(operations)

    # Convert back to RGB numpy array
    return np.array(result.convert("RGB"))
