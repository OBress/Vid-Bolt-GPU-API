"""Video processing utilities.

This module provides utilities for post-processing generated videos,
including trimming to exact durations after frame-rounded generation.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)


def trim_video_to_duration(
    video_data: bytes,
    target_duration: float,
    preserve_audio: bool = True,
) -> bytes:
    """Trim video to exact target duration.
    
    LTX-2 generates videos with frame counts following the 8k+1 pattern.
    This function trims the generated video to the exact requested duration.
    
    Args:
        video_data: Raw video bytes (MP4 format expected)
        target_duration: Target duration in seconds
        preserve_audio: Whether to preserve audio in the trimmed video
        
    Returns:
        Trimmed video as bytes
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip
    
    # Write input to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as input_file:
        input_file.write(video_data)
        input_path = input_file.name
    
    try:
        # Load and trim video
        clip = VideoFileClip(input_path)
        
        # Get actual duration
        actual_duration = clip.duration
        
        if actual_duration <= target_duration:
            # Video is already shorter or equal to target, return as-is
            logger.info(
                f"Video duration ({actual_duration:.2f}s) <= target ({target_duration:.2f}s), "
                f"no trimming needed"
            )
            clip.close()
            return video_data
        
        # Trim to target duration
        trimmed_clip = clip.subclip(0, target_duration)
        
        # Write to temp output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
            output_path = output_file.name
        
        # Write with audio if present and requested
        has_audio = clip.audio is not None and preserve_audio
        trimmed_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac" if has_audio else None,
            audio=has_audio,
            verbose=False,
            logger=None,
        )
        
        # Read trimmed video
        with open(output_path, "rb") as f:
            trimmed_data = f.read()
        
        logger.info(
            f"Trimmed video from {actual_duration:.2f}s to {target_duration:.2f}s"
        )
        
        # Cleanup
        clip.close()
        trimmed_clip.close()
        Path(output_path).unlink(missing_ok=True)
        
        return trimmed_data
        
    finally:
        # Cleanup input temp file
        Path(input_path).unlink(missing_ok=True)


def get_video_info(video_data: bytes) -> dict:
    """Get video metadata including duration, resolution, and audio presence.
    
    Args:
        video_data: Raw video bytes
        
    Returns:
        Dictionary with video metadata
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_data)
        temp_path = f.name
    
    try:
        clip = VideoFileClip(temp_path)
        info = {
            "duration": clip.duration,
            "width": clip.w,
            "height": clip.h,
            "fps": clip.fps,
            "has_audio": clip.audio is not None,
        }
        clip.close()
        return info
    finally:
        Path(temp_path).unlink(missing_ok=True)
