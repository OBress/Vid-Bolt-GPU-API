"""Placeholder image and video generation using Pillow and moviepy."""

import io
import logging
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class PlaceholderGenerator:
    """Generate placeholder images and videos for mock mode."""

    # Color palettes for different edit types
    EDIT_TINTS = {
        "inpaint": (255, 100, 100, 80),  # Red tint
        "outpaint": (255, 165, 0, 80),  # Orange tint
        "style_transfer": (100, 150, 255, 80),  # Blue tint
        "remove_background": (100, 255, 100, 80),  # Green tint
        "upscale": (200, 100, 255, 80),  # Purple tint
    }

    def create_image(
        self,
        width: int,
        height: int,
        prompt: str,
        job_id: str,
        seed: int,
    ) -> bytes:
        """Create a placeholder image with gradient background and text.

        Args:
            width: Image width
            height: Image height
            prompt: Generation prompt
            job_id: Job identifier
            seed: Random seed used

        Returns:
            PNG image as bytes
        """
        # Use seed for reproducible random colors
        rng = random.Random(seed)

        # Generate two random colors for gradient
        color1 = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
        color2 = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))

        # Create image with gradient
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)

        # Draw vertical gradient
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Prepare text
        truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

        # Draw text with outline for visibility
        self._draw_centered_text(
            draw,
            width,
            height,
            [
                (job_id[:36], 16),
                (truncated_prompt, 24),
                (f"{width}x{height}", 16),
                (f"MOCK - Seed: {seed}", 14),
            ],
        )

        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        logger.info(
            f"Created placeholder image",
            extra={"width": width, "height": height, "seed": seed},
        )

        return buffer.getvalue()

    def create_edited_image(
        self,
        input_image_data: bytes,
        edit_type: str,
        prompt: str,
        job_id: str,
        seed: int,
    ) -> tuple[bytes, int, int, int, int]:
        """Create a placeholder edited image with color tint.

        Args:
            input_image_data: Original image as bytes
            edit_type: Type of edit to simulate
            prompt: Edit prompt
            job_id: Job identifier
            seed: Random seed

        Returns:
            Tuple of (image_bytes, original_width, original_height, output_width, output_height)
        """
        # Load input image
        input_image = Image.open(io.BytesIO(input_image_data))
        original_width, original_height = input_image.size

        # Convert to RGB if necessary
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Handle upscale - double dimensions
        if edit_type == "upscale":
            output_width = original_width * 2
            output_height = original_height * 2
            input_image = input_image.resize((output_width, output_height), Image.Resampling.LANCZOS)
        else:
            output_width = original_width
            output_height = original_height

        # Apply color tint overlay
        tint_color = self.EDIT_TINTS.get(edit_type, (100, 100, 100, 80))
        overlay = Image.new("RGBA", (output_width, output_height), tint_color)
        input_image = input_image.convert("RGBA")
        input_image = Image.alpha_composite(input_image, overlay)
        input_image = input_image.convert("RGB")

        # Add watermark text
        draw = ImageDraw.Draw(input_image)
        watermark = f"EDITED: {edit_type}"

        # Draw watermark in corner
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Position in bottom-right with padding
        bbox = draw.textbbox((0, 0), watermark, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = output_width - text_width - 20
        y = output_height - text_height - 20

        # Draw with outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + dx, y + dy), watermark, fill="black", font=font)
        draw.text((x, y), watermark, fill="white", font=font)

        # Save to bytes
        buffer = io.BytesIO()
        input_image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        logger.info(
            f"Created placeholder edited image",
            extra={
                "edit_type": edit_type,
                "original_size": f"{original_width}x{original_height}",
                "output_size": f"{output_width}x{output_height}",
            },
        )

        return buffer.getvalue(), original_width, original_height, output_width, output_height

    def create_video(
        self,
        input_image_data: bytes,
        prompt: str,
        job_id: str,
        duration_seconds: float,
        fps: int,
        seed: int,
    ) -> tuple[bytes, int, int]:
        """Create a placeholder video with static image and text overlay.

        Args:
            input_image_data: Source image as bytes
            prompt: Motion prompt
            job_id: Job identifier
            duration_seconds: Video duration
            fps: Frames per second
            seed: Random seed

        Returns:
            Tuple of (video_bytes, width, height)
        """
        from moviepy.editor import ImageClip, TextClip, CompositeVideoClip

        # Load input image
        input_image = Image.open(io.BytesIO(input_image_data))
        width, height = input_image.size

        # Convert to RGB if necessary
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Add text overlay to image
        draw = ImageDraw.Draw(input_image)
        truncated_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt

        # Draw text in center
        self._draw_centered_text(
            draw,
            width,
            height,
            [
                (f"Job: {job_id[:8]}...", 14),
                (truncated_prompt, 18),
                (f"MOCK VIDEO - {duration_seconds}s @ {fps}fps", 14),
            ],
        )

        # Save image to temp file for moviepy
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            input_image.save(tmp_img, format="PNG")
            tmp_img_path = tmp_img.name

        try:
            # Create video clip from static image
            clip = ImageClip(tmp_img_path, duration=duration_seconds)

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
                tmp_vid_path = tmp_vid.name

            clip.write_videofile(
                tmp_vid_path,
                fps=fps,
                codec="libx264",
                audio=False,
                verbose=False,
                logger=None,
            )

            # Read video bytes
            video_bytes = Path(tmp_vid_path).read_bytes()

            logger.info(
                f"Created placeholder video",
                extra={
                    "duration": duration_seconds,
                    "fps": fps,
                    "size": f"{width}x{height}",
                },
            )

            return video_bytes, width, height

        finally:
            # Cleanup temp files
            Path(tmp_img_path).unlink(missing_ok=True)
            if "tmp_vid_path" in locals():
                Path(tmp_vid_path).unlink(missing_ok=True)

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        lines: list[tuple[str, int]],
    ) -> None:
        """Draw centered text lines with outline.

        Args:
            draw: ImageDraw instance
            width: Image width
            height: Image height
            lines: List of (text, font_size) tuples
        """
        # Calculate total height
        y_offset = height // 2 - (len(lines) * 30) // 2

        for text, font_size in lines:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2

            # Draw outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                draw.text((x + dx, y_offset + dy), text, fill="black", font=font)

            # Draw text
            draw.text((x, y_offset), text, fill="white", font=font)
            y_offset += font_size + 10
