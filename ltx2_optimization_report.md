# LTX-2 Workflow Optimization & Issue Analysis Report

This report evaluates the current LTX-2 video generation implementation for performance, quality, and compatibility with the **RTX PRO 6000 Blackwell GPU**.

## 🚀 Optimization Findings

### 1. Hardware-Specific Performance (Blackwell GPU)

- **Attention Backend**: LTX-2 currently defaults to `xformers` (if installed) or PyTorch SDPA. While `FlashAttention3` is supported in the codebase, it is incompatible with the **masks** used for keyframe conditioning.
  - **Status**: Using `xformers` is currently the optimal stable choice for Blackwell.
- **Precision**: The current use of `FP8` for the transformer and `bfloat16` for other components is ideal, reducing VRAM usage to ~20GB and significantly speeding up inference on Blackwell architecture.

### 2. Video Processing Bottlenecks (Critical)

The current pipeline suffers from **redundant re-encoding**, which degrades quality and increases processing time:

1.  **Stage 1/2 Encoder**: Generates MP4 with audio using internal `ltx-pipelines`.
2.  **Trim Step**: `moviepy` reads the MP4 and re-encodes it to trim frames.
3.  **Crop Step**: `moviepy` reads the trimmed MP4 and re-encodes it to remove the 64-pixel padding.
    > [!IMPORTANT]
    > This triple re-encoding process is highly inefficient. Each pass with `libx264` introduces compression artifacts and risks audio/video desync.

### 3. Resolution & Padding Constraints

- **Two-Stage Requirement**: LTX-2's two-stage pipeline (low-res + upsample) **strictly requires** resolutions divisible by **64**.
- **Current Approach**: The system pads any resolution (e.g., 1080p) to the next 64-multiple and then crops.
- **Observation**: 1920x1080 is not 64-divisible (1080/64 = 16.875). It is padded to 1920x1088.

### 4. FPS & Duration Accuracy

- **Frame Counting**: The `8k + 1` frame pattern is correctly implemented (`round_up_to_valid_frames`).
- **FPS**: The `frame_rate` is respected, but `moviepy` trimming can sometimes lead to frame jitter if the underlying duration isn't a perfect multiple of the frame time.

---

## 🛠️ Identified Issues & Risks

| Issue Area      | Description                                                   | Risk Level |
| :-------------- | :------------------------------------------------------------ | :--------- |
| **Performance** | Triple re-encoding during post-processing (trim + crop).      | 🔴 High    |
| **Quality**     | Cumulative loss from repeated compression passes.             | 🟠 Medium  |
| **Audio Sync**  | Trimming after audio generation in `moviepy` can cause drift. | 🟠 Medium  |
| **Resolution**  | 1080p (1080) requires padding to 1088, necessitating a crop.  | 🟡 Low     |

---

## 💡 Recommended Next Steps (DO NOT IMPLEMENT YET)

1.  **Single-Pass Processing**: Modify the generator to perform trimming and cropping at the pixel/latent level _before_ the final MP4 encoding.
2.  **FFmpeg Optimization**: If post-process cutting is necessary, use `ffmpeg` with stream copying or high-bitrate encoding to minimize loss.
3.  **1080p Calibration**: Evaluate if native 1088p generation is acceptable to avoid the final crop pass, or use a more efficient padding-aware VAE decoder if available.
4.  **XFormers Verification**: Ensure `xformers` is correctly compiled for the Blackwell environment to maximize the throughput.

---

_Report compiled by Antigravity on 2026-01-11._
