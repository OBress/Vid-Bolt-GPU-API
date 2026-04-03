"""Shared video encoding helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

VIDEO_CODEC = "libx264"
PIXEL_FORMAT = "yuv420p"
MOVFLAGS = "+faststart"


def _normalize_frame(frame: Any) -> np.ndarray:
    """Normalize a frame to contiguous uint8 RGB."""
    frame_array = np.asarray(frame)

    if frame_array.ndim != 3 or frame_array.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {frame_array.shape}")

    if frame_array.dtype != np.uint8:
        frame_array = frame_array.astype(np.uint8)

    return np.ascontiguousarray(frame_array)


def encode_mp4_h264(frames: Iterable[Any], fps: float) -> tuple[bytes, dict[str, Any]]:
    """Encode RGB frames as H.264 MP4 with browser-safe defaults."""
    iterator = iter(frames)
    try:
        first_frame = _normalize_frame(next(iterator))
    except StopIteration as exc:
        raise ValueError("No frames to encode") from exc

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for H.264 encoding but was not found on PATH")

    height, width = first_frame.shape[:2]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    process = subprocess.Popen(
        [
            ffmpeg_path,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            VIDEO_CODEC,
            "-pix_fmt",
            PIXEL_FORMAT,
            "-movflags",
            MOVFLAGS,
            tmp_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_count = 0
    stderr_bytes = b""

    try:
        assert process.stdin is not None
        process.stdin.write(first_frame.tobytes())
        frame_count += 1

        for frame in iterator:
            frame_array = _normalize_frame(frame)
            if frame_array.shape[:2] != (height, width):
                raise ValueError(
                    f"All frames must share the same size; expected {(height, width)}, "
                    f"got {frame_array.shape[:2]}"
                )
            process.stdin.write(frame_array.tobytes())
            frame_count += 1
    except BrokenPipeError as exc:
        stderr_bytes = process.stderr.read() if process.stderr is not None else b""
        raise RuntimeError(
            f"ffmpeg terminated early during H.264 encoding: {stderr_bytes.decode('utf-8', errors='ignore')}"
        ) from exc
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

    if process.stderr is not None:
        stderr_bytes = process.stderr.read()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed to encode H.264 MP4: {stderr_bytes.decode('utf-8', errors='ignore')}"
        )

    try:
        mp4_bytes = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return mp4_bytes, {
        "codec": VIDEO_CODEC,
        "pixel_format": PIXEL_FORMAT,
        "movflags": MOVFLAGS,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
