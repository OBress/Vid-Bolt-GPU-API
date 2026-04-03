"""Smoke tests for browser-safe H.264 video encoding."""

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from app.utils.video_encoding import PIXEL_FORMAT, encode_mp4_h264


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)
def test_encode_mp4_h264_outputs_h264_yuv420p_stream():
    frames = [
        np.full((16, 16, 3), fill_value=0, dtype=np.uint8),
        np.full((16, 16, 3), fill_value=127, dtype=np.uint8),
        np.full((16, 16, 3), fill_value=255, dtype=np.uint8),
    ]

    mp4_bytes, encode_info = encode_mp4_h264(frames, fps=24)
    assert mp4_bytes
    assert encode_info["pixel_format"] == PIXEL_FORMAT

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(mp4_bytes)

        probe = subprocess.run(
            [
                shutil.which("ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "json",
                tmp_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["codec_name"] == "h264"
        assert stream["pix_fmt"] == "yuv420p"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
