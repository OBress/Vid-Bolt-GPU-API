# Vid-Bolt GPU API

A high-performance FastAPI backend for AI-powered image, video, and music generation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Webhooks](#webhooks)
- [Mode System](#mode-system)
- [Endpoints](#endpoints)
  - [Health & System](#health--system)
  - [Mode Management](#mode-management)
  - [Settings](#settings)
  - [Job Management](#job-management)
  - [Image Generation](#image-generation)
  - [Image Editing](#image-editing)
  - [Video Generation](#video-generation)
  - [LTX-2 Video Generation](#ltx-2-video-generation)
  - [LTX-2 Keyframe Interpolation](#ltx-2-keyframe-interpolation)
  - [Music Generation](#music-generation)
  - [Batch Operations](#batch-operations)
  - [LoRA Management](#lora-management)
  - [GPU Monitoring](#gpu-monitoring)
  - [System Status](#system-status)
  - [Download Status](#download-status)
- [Error Handling](#error-handling)
- [Changelog](#changelog)

---

## Overview

Vid-Bolt GPU API provides AI-powered generation capabilities:

| Capability           | Model                | Description                              |
| -------------------- | -------------------- | ---------------------------------------- |
| **Text-to-Image**    | Z-Image Turbo        | Generate images from text prompts        |
| **Image Editing**    | Qwen-Image-Edit-2511 | Edit images with AI instructions         |
| **Video Generation** | LTX-2 19B            | Generate videos from images (720p/1080p) |
| **Music Generation** | ACE-Step 1.5         | Generate music from text prompts         |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Vid-Bolt GPU API                      │
├─────────────────────────────────────────────────────────┤
│  Queue Manager - Accepts requests (202 Accepted)         │
│  & Schedules based on VRAM Mode (FIFO or Grouped)        │
├─────────────┬───────────────────────────────────────────┤
│   WORKER    │  • Intelligent Job Scheduling             │
│   THREAD    │  • Automatic Mode Switching (Dynamic)     │
│             │  • OOM & Timeout Handling                 │
├─────────────┴───────────────────────────────────────────┤
│   IMAGE MODE       │   VIDEO MODE    │   AUDIO MODE     │
│  ┌──────────────┐  │ ┌─────────────┐ │ ┌──────────────┐ │
│  │ Z-Image Turbo│  │ │ LTX-2 19B   │ │ │ ACE-Step 1.5 │ │
│  │ (text-to-img)│  │ │ (I2V, 720p/ │ │ │ (music gen)  │ │
│  ├──────────────┤  │ │ 1080p)      │ │ └──────────────┘ │
│  │ Qwen-Image-  │  │ └─────────────┘ │                  │
│  │ Edit (editing)│  │                 │                  │
│  └──────────────┘  │                 │                  │
└────────────────────┴─────────────────┴──────────────────┘
```

---

## Quick Start

### 1. Environment Setup

Create a `.env` file:

```env
MOCK_MODE=false
API_KEY=your-secure-api-key
LOG_LEVEL=INFO
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

### 2. Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Check Health

```bash
curl http://localhost:8000/health
```

---

## Authentication

All `/api/v1/*` endpoints require the `X-API-Key` header:

```http
X-API-Key: your-secure-api-key
```

**Exceptions:** The following endpoints do **not** require authentication:

- `GET /health`
- `GET /health/ready`
- `GET /api/v1/download/status`

---

## Webhooks

> **Note:** Webhook URLs are **optional** on all generation endpoints. If provided, results are delivered via webhook and job data is deleted after successful delivery. If omitted, use polling via `/api/v1/jobs/{job_id}`.

### How It Works

1. Submit a generation request with an optional `webhook_url`
2. Poll `/api/v1/jobs/{job_id}` for progress and results
3. If `webhook_url` was provided, receive webhook callback when job completes
4. Job data is automatically deleted after webhook delivery

### Webhook Payload (Success)

```json
{
  "event": "generation.completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "item_id": "scene_001_image",
  "batch_id": null,
  "status": "completed",
  "completed_at": 1715420015.0,
  "generation_type": "image_generation",
  "result": {
    "save_url": "https://storage.example.com/output.png",
    "generation_time": 2.5,
    "metadata": { "seed": 12345, "width": 1920, "height": 1080 }
  }
}
```

### Webhook Payload (Failure)

```json
{
  "event": "generation.failed",
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "item_id": "scene_002_image",
  "batch_id": "batch-abc123",
  "status": "failed",
  "completed_at": 1715420020.0,
  "generation_type": "image_generation",
  "error_message": "GPU out of memory",
  "error_code": "GPU_OUT_OF_MEMORY",
  "retry_count": 1
}
```

### Webhook Headers

| Header                | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `Content-Type`        | `application/json`                                   |
| `X-Webhook-Event`     | `generation.completed` or `generation.failed`        |
| `X-Job-Id`            | The job ID                                           |
| `X-Webhook-Signature` | HMAC-SHA256 signature (if `webhook_secret` provided) |

### Webhook Retry Logic

- **Timeout:** 10 seconds per delivery attempt
- **Retries:** 1 retry after initial failure (30 second delay)
- **Total attempts:** 2 (initial + 1 retry)

### HMAC Signature Verification

If you provide a `webhook_secret`, the payload is signed with HMAC-SHA256:

```python
import hmac
import hashlib

def verify_signature(payload_body: bytes, signature: str, secret: str) -> bool:
    expected = f"sha256={hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()}"
    return hmac.compare_digest(expected, signature)
```

---

## Mode System & Scheduling

The API manages GPU VRAM by loading only the required models for each use case. A **Queue System** manages requests to ensure fairness and efficiency.

### VRAM Loading Modes

Configurable via `POST /api/v1/settings/vram-mode`:

| Mode               | Models Loaded      | VRAM Usage | Best For                   |
| ------------------ | ------------------ | ---------- | -------------------------- |
| `image_generation` | Z-Image Turbo only | ~16GB      | Text-to-image workloads    |
| `image_editing`    | LightX2V only      | ~40GB      | Image editing/inpainting   |
| `video_generation` | LTX-2 only         | ~40GB      | Video generation           |
| `audio_creation`   | ACE-Step           | ~4GB       | Music generation           |
| `all`              | All models         | ~76GB+     | High-VRAM GPUs (A100/H100) |

#### Mode Behavior

1. **image_generation** (Default):
   - Loads **Z-Image Turbo** for text-to-image generation
   - Scheduling: Grouped by job type to minimize switching
   - Switching time: ~15-30s

2. **image_editing**:
   - Loads **LightX2V (Qwen-Image-Edit)** for image editing
   - Scheduling: Grouped by job type to minimize switching
   - Switching time: ~15-30s

3. **video_generation**:
   - Loads **LTX-2 19B** for video generation
   - Scheduling: Grouped by job type to minimize switching
   - Switching time: ~30-60s

4. **audio_creation**:
   - Loads **ACE-Step 1.5** for music generation
   - Scheduling: Grouped by job type to minimize switching
   - Switching time: ~15-30s

5. **all**:
   - Loads **all models simultaneously**
   - Scheduling: Strict FIFO (no switching needed)
   - Switching time: Instant

### Concurrency Limits

The Queue accepts jobs even if the GPU is busy.

| Resource | Limit                 | Behavior                            |
| -------- | --------------------- | ----------------------------------- |
| Queue    | Unbounded (in-memory) | Returns `202 Accepted` immediately. |
| Worker   | 1 Active Job          | Processes one job at a time.        |

### Throughput & Capacity

The API dynamically calculates batch sizes and concurrent processing based on **available VRAM** and **task parameters**. Jobs are automatically grouped by resolution for optimal GPU utilization.

#### Concurrent Capacity by Mode (96GB GPU)

| Mode               | Image Gen (Z-Image)                      | Image Edit (LightX2V)             | Video Gen (LTX-2)                 |
| ------------------ | ---------------------------------------- | --------------------------------- | --------------------------------- |
| `image_generation` | **~29** @ 1920×1080, **~48** @ 1024×1024 | ❌ Not loaded                     | ❌ Not loaded                     |
| `image_editing`    | ❌ Not loaded                            | **6 concurrent** (any resolution) | ❌ Not loaded                     |
| `video_generation` | ❌ Not loaded                            | ❌ Not loaded                     | **4 concurrent** @ 3s, **3** @ 5s |
| `all`              | **~10-15** @ 1024×1024                   | **2-3 concurrent**                | **1-2 concurrent**                |

#### Resolution Scaling (Image Generation)

| Resolution | VRAM per Image | Batch Size |
| ---------- | -------------- | ---------- |
| 512×512    | ~0.8 GB        | 64 (cap)   |
| 1024×1024  | ~1.8 GB        | ~48        |
| 1920×1080  | ~3.0 GB        | ~29        |
| 2048×2048  | ~5.5 GB        | ~16        |

#### Video Duration Scaling (LTX-2)

| Duration   | VRAM per Video | Concurrent Videos |
| ---------- | -------------- | ----------------- |
| 3 seconds  | ~12 GB         | 4                 |
| 5 seconds  | ~16 GB         | 3-4               |
| 10 seconds | ~25 GB         | 2-3               |

#### Optimization Tips

1. **Batch same-resolution jobs**: Jobs with identical dimensions are grouped for vectorized processing
2. **Use dedicated modes**: Single-model modes (`image_generation`, `video_generation`) offer 2-3x higher throughput than `all` mode
3. **Shorter videos first**: Submit shorter-duration videos when possible for higher parallelism
4. **Check queue position**: Poll `/api/v1/jobs/{job_id}` to see `queue_position` for pending jobs

---

## Endpoints

### Health & System

#### `GET /health`

Basic health check. **No authentication required.**

**Response:**

```json
{
  "status": "healthy",
  "version": "0.6.0",
  "mock_mode": false
}
```

---

#### `GET /health/ready`

Readiness check for VM provisioning. **No authentication required.** Returns whether models are loaded and the API is ready for generation requests.

**Response:**

```json
{
  "ready": true,
  "status": "ready",
  "version": "0.6.0",
  "mock_mode": false,
  "current_mode": "image_generation",
  "models_loaded": true
}
```

| Field           | Type         | Description                                                                             |
| --------------- | ------------ | --------------------------------------------------------------------------------------- |
| `ready`         | bool         | Whether the API can accept generation requests                                          |
| `status`        | string       | `ready`, `starting`, `loading_models`, `downloading_models`, `download_failed`, `error` |
| `current_mode`  | string\|null | Current VRAM mode (null if not yet initialized)                                         |
| `models_loaded` | bool         | Whether generation models are loaded in memory                                          |

---

#### `GET /api/v1/status`

Detailed service status. **Requires authentication.**

**Response:**

```json
{
  "status": "healthy",
  "version": "0.6.0",
  "mock_mode": false
}
```

---

### Mode Management

#### `GET /api/v1/mode`

Get the current VRAM mode status including switching progress.

**Response (Not Switching):**

```json
{
  "mode": "video_generation",
  "is_busy": false,
  "active_job_id": null,
  "loaded_models": ["ltx-2-19b"],
  "is_switching": false,
  "switching_target": null,
  "switching_step": null,
  "switching_progress": null
}
```

**Response (Switching in Progress):**

```json
{
  "mode": "image_generation",
  "is_busy": false,
  "active_job_id": null,
  "loaded_models": [],
  "is_switching": true,
  "switching_target": "video_generation",
  "switching_step": "Loading LTX-2 models (this takes 2-3 minutes)...",
  "switching_progress": 0.2
}
```

| Field                | Type         | Description                                                                                     |
| -------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `mode`               | string       | Current mode (`image_generation`, `image_editing`, `video_generation`, `audio_creation`, `all`) |
| `is_busy`            | bool         | Whether a job is currently running                                                              |
| `active_job_id`      | string\|null | ID of the currently running job                                                                 |
| `loaded_models`      | list[string] | Names of currently loaded models                                                                |
| `is_switching`       | bool         | Whether mode switch is in progress                                                              |
| `switching_target`   | string\|null | Target mode when switching                                                                      |
| `switching_step`     | string\|null | Current switching step description                                                              |
| `switching_progress` | float\|null  | Progress 0.0-1.0 when switching                                                                 |

---

#### `POST /api/v1/mode/switch`

Switch between Image Mode and Video Mode. Unloads current models and loads the target mode's models (~30-60 seconds).

**Request:**

```json
{
  "target_mode": "image"
}
```

| Field         | Type   | Required | Description            |
| ------------- | ------ | -------- | ---------------------- |
| `target_mode` | string | ✅       | `"image"` or `"video"` |

> **Note:** For full mode control (including `image_editing`, `audio_creation`, `all`), use the `POST /api/v1/settings/vram-mode` endpoint instead.

**Response:**

```json
{
  "status": "success",
  "previous_mode": "image_generation",
  "current_mode": "video_generation",
  "message": "Successfully switched from image_generation to video_generation mode"
}
```

---

### Settings

#### `GET /api/v1/settings/vram-mode`

Get the current VRAM loading mode.

**Response:**

```json
{
  "mode": "image_generation",
  "description": "Image Generation - Z-Image Turbo only (~16GB VRAM)"
}
```

---

#### `POST /api/v1/settings/vram-mode`

Set the VRAM loading mode. This unloads current models and loads the target mode's models.

**Request:**

```json
{
  "mode": "video_generation"
}
```

| Field  | Type   | Required | Description                                                                              |
| ------ | ------ | -------- | ---------------------------------------------------------------------------------------- |
| `mode` | string | ✅       | One of: `image_generation`, `image_editing`, `video_generation`, `audio_creation`, `all` |

**Response:**

```json
{
  "mode": "video_generation",
  "description": "Video Generation - LTX-2 DistilledPipeline only (~40GB VRAM)"
}
```

---

### Job Management

All generation endpoints are **asynchronous**. They return a `job_id` which you use to poll for status.

#### `GET /api/v1/jobs/{job_id}`

Check the status of a specific job.

**Response (Pending):**

> **Note:** `queue_position` is 1-based and indicates the number of jobs ahead + 1. It is only present when `status` is `pending`.

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "created_at": 1715420000.0,
  "queue_position": 2
}
```

**Response (Processing):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "processing",
  "created_at": 1715420000.0,
  "started_at": 1715420005.0,
  "progress_percent": 45,
  "progress_stage": "generating"
}
```

**Response (Completed):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "completed",
  "created_at": 1715420000.0,
  "completed_at": 1715420015.0,
  "result": {
    "save_url": "https://storage.example.com/result.png",
    "generation_time": 10.0
  }
}
```

---

### Image Generation

#### `POST /api/v1/image/generate`

**Returns HTTP 202 Accepted**. The system will automatically switch to Image Mode if needed.

**Request:**

| Field                 | Type   | Required | Description                                   | Default  |
| --------------------- | ------ | -------- | --------------------------------------------- | -------- |
| `job_id`              | string | ✅       | Unique job identifier                         | -        |
| `prompt`              | string | ✅       | Text description (max 2000 chars)             | -        |
| `aspect_ratio`        | string | ❌       | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`           | `16:9`   |
| `width`               | int    | ❌       | Custom width in pixels (256-2048)             | -        |
| `height`              | int    | ❌       | Custom height in pixels (256-2048)            | -        |
| `num_inference_steps` | int    | ❌       | Number of diffusion steps (1-50)              | `20`     |
| `seed`                | int    | ❌       | Random seed for reproducibility               | -        |
| `lora_name`           | string | ❌       | LoRA style to apply (or `"none"` for no LoRA) | -        |
| `save_url`            | string | ✅       | Presigned PUT URL for output                  | -        |
| `webhook_url`         | string | ❌       | URL to POST when complete                     | -        |
| `item_id`             | string | ❌       | Client identifier (returned in webhook)       | `job_id` |
| `webhook_secret`      | string | ❌       | HMAC signing secret                           | -        |

> **Note:** If `width` and `height` are provided, they override `aspect_ratio`. Both must be specified together.

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "status_url": "/api/v1/jobs/550e8400-e29b...",
  "message": "Job accepted for processing"
}
```

---

### Image Editing

#### `POST /api/v1/image/edit`

**Returns HTTP 202 Accepted**.

**Request:**

| Field             | Type   | Required | Description                               | Default  |
| ----------------- | ------ | -------- | ----------------------------------------- | -------- |
| `job_id`          | string | ✅       | Unique job identifier                     | -        |
| `input_image_url` | string | ✅       | URL of image to edit                      | -        |
| `prompt`          | string | ✅       | Edit instruction (max 2000 chars)         | -        |
| `aspect_ratio`    | string | ❌       | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`       | `16:9`   |
| `mask_image_url`  | string | ❌       | URL of mask image for inpainting          | -        |
| `seed`            | int    | ❌       | Random seed for reproducibility           | -        |
| `save_url`        | string | ✅       | Presigned PUT URL for output              | -        |
| `webhook_url`     | string | ❌       | URL to POST when complete                 | -        |
| `item_id`         | string | ❌       | Client identifier (returned in webhook)   | `job_id` |
| `webhook_secret`  | string | ❌       | HMAC signing secret                       | -        |
| `lora_name`       | string | ❌       | LoRA to apply (see available LoRAs below) | -        |
| `lora_strength`   | float  | ❌       | LoRA strength (0.0-1.0)                   | `0.9`    |

**Available LoRAs:**

| Name              | Description                                         | Prompt Format                            |
| ----------------- | --------------------------------------------------- | ---------------------------------------- |
| `multiple-angles` | 96-position camera control for multi-view synthesis | `<sks> {azimuth} {elevation} {distance}` |

**Multiple Angles LoRA Usage:**

The `multiple-angles` LoRA enables precise camera angle control. Use the `<sks>` trigger token followed by position descriptors:

- **Azimuth:** `front`, `front-left`, `front-right`, `left`, `right`, `back`, `back-left`, `back-right`
- **Elevation:** `below`, `eye-level`, `above`, `overhead`
- **Distance:** `close`, `medium`, `far`

Example prompts:

- `<sks> front-right eye-level medium` - Standard 3/4 view
- `<sks> above front far` - High-angle establishing shot
- `<sks> left eye-level close` - Profile close-up

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "status_url": "/api/v1/jobs/550e8400-e29b...",
  "message": "Job accepted for processing"
}
```

**Example Request with Multiple Angles LoRA:**

```json
{
  "job_id": "angle-test-001",
  "input_image_url": "https://example.com/product.png",
  "prompt": "<sks> front-right eye-level medium",
  "lora_name": "multiple-angles",
  "lora_strength": 0.9,
  "save_url": "https://storage.example.com/outputs/angle-test-001.png",
  "webhook_url": "https://myapp.com/webhook"
}
```

---

### Video Generation

#### `POST /api/v1/video/generate`

**Returns HTTP 202 Accepted**. Generate a video from an input image and prompt.

**Request:**

| Field              | Type   | Required | Description                                      | Default |
| ------------------ | ------ | -------- | ------------------------------------------------ | ------- |
| `job_id`           | string | ✅       | Unique job identifier                            | -       |
| `input_image_url`  | string | ✅       | URL of the first frame image                     | -       |
| `prompt`           | string | ✅       | Description of motion/action (max 2000 chars)    | -       |
| `duration_seconds` | float  | ❌       | Video duration (1.0-8.0 seconds)                 | `4.0`   |
| `fps`              | int    | ❌       | Frames per second (8, 12, 16, 24, or 30)         | `24`    |
| `aspect_ratio`     | string | ❌       | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`              | `16:9`  |
| `width`            | int    | ❌       | Target width (512-1920, overrides aspect_ratio)  | -       |
| `height`           | int    | ❌       | Target height (512-1920, overrides aspect_ratio) | -       |
| `seed`             | int    | ❌       | Random seed for reproducibility                  | -       |
| `end_image_url`    | string | ❌       | Optional URL of end frame for interpolation      | -       |
| `save_url`         | string | ✅       | Presigned PUT URL for output                     | -       |

> **Note:** This endpoint does not currently support `webhook_url`. Use polling via `/api/v1/jobs/{job_id}`.

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "status_url": "/api/v1/jobs/550e8400-e29b...",
  "message": "Job accepted for processing"
}
```

---

### LTX-2 Video Generation

#### `POST /api/v1/ltx2/generate`

**Returns HTTP 202 Accepted**. The system will automatically switch to Video Mode if needed.

**Request:**

| Field              | Type   | Required | Description                                      | Default  |
| ------------------ | ------ | -------- | ------------------------------------------------ | -------- |
| `job_id`           | string | ✅       | Unique job identifier                            | -        |
| `start_frame_url`  | string | ✅       | URL of starting frame image                      | -        |
| `prompt`           | string | ✅       | Motion description (max 2000 chars)              | -        |
| `negative_prompt`  | string | ❌       | What should not appear (max 1000 chars)          | `""`     |
| `duration_seconds` | float  | ❌       | Video length (0.5-10.0)                          | `5.0`    |
| `frame_rate`       | float  | ❌       | Frame rate (8.0-60.0)                            | `24.0`   |
| `aspect_ratio`     | string | ❌       | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`              | `16:9`   |
| `width`            | int    | ❌       | Target width (512-1920, overrides aspect_ratio)  | -        |
| `height`           | int    | ❌       | Target height (512-1920, overrides aspect_ratio) | -        |
| `end_frame_url`    | string | ❌       | Optional URL of end frame for interpolation      | -        |
| `seed`             | int    | ❌       | Random seed for reproducibility                  | -        |
| `enhance_prompt`   | bool   | ❌       | Auto-enhance prompt                              | `false`  |
| `save_url`         | string | ✅       | Presigned PUT URL                                | -        |
| `webhook_url`      | string | ❌       | URL to POST when complete                        | -        |
| `item_id`          | string | ❌       | Client identifier (returned in webhook)          | `job_id` |
| `webhook_secret`   | string | ❌       | HMAC signing secret                              | -        |

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "status_url": "/api/v1/jobs/550e8400-e29b...",
  "message": "Job accepted for processing"
}
```

---

### LTX-2 Keyframe Interpolation

#### `POST /api/v1/ltx2/interpolate`

**Returns HTTP 202 Accepted**. Generate a video by interpolating between multiple keyframes.

**Request:**

| Field              | Type            | Required | Description                                      | Default  |
| ------------------ | --------------- | -------- | ------------------------------------------------ | -------- |
| `job_id`           | string          | ✅       | Unique job identifier                            | -        |
| `prompt`           | string          | ✅       | Video content description (max 2000 chars)       | -        |
| `negative_prompt`  | string          | ❌       | What should not appear (max 1000 chars)          | `""`     |
| `keyframes`        | KeyframeImage[] | ✅       | Keyframe images with frame indices (1-10)        | -        |
| `duration_seconds` | float           | ❌       | Video length (0.5-10.0)                          | `5.0`    |
| `frame_rate`       | float           | ❌       | Frame rate (8.0-60.0)                            | `24.0`   |
| `aspect_ratio`     | string          | ❌       | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`              | `16:9`   |
| `width`            | int             | ❌       | Target width (512-1920, overrides aspect_ratio)  | -        |
| `height`           | int             | ❌       | Target height (512-1920, overrides aspect_ratio) | -        |
| `seed`             | int             | ❌       | Random seed for reproducibility                  | -        |
| `enhance_prompt`   | bool            | ❌       | Auto-enhance prompt                              | `false`  |
| `save_url`         | string          | ✅       | Presigned PUT URL                                | -        |
| `webhook_url`      | string          | ❌       | URL to POST when complete                        | -        |
| `item_id`          | string          | ❌       | Client identifier (returned in webhook)          | `job_id` |
| `webhook_secret`   | string          | ❌       | HMAC signing secret                              | -        |

**KeyframeImage Object:**

| Field         | Type   | Required | Description                     | Default |
| ------------- | ------ | -------- | ------------------------------- | ------- |
| `image_url`   | string | ✅       | URL of the keyframe image       | -       |
| `frame_index` | int    | ✅       | Target frame index (0-indexed)  | -       |
| `strength`    | float  | ❌       | Conditioning strength (0.0-1.0) | `1.0`   |

**Example Request:**

```json
{
  "job_id": "interp-001",
  "prompt": "A person walking from left to right, cinematic lighting",
  "keyframes": [
    {
      "image_url": "https://example.com/start.png",
      "frame_index": 0,
      "strength": 1.0
    },
    {
      "image_url": "https://example.com/end.png",
      "frame_index": 120,
      "strength": 1.0
    }
  ],
  "duration_seconds": 5.0,
  "frame_rate": 24.0,
  "save_url": "https://example.com/upload/video.mp4",
  "webhook_url": "https://myapp.com/api/gpu-callback"
}
```

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "interp-001",
  "status": "pending",
  "status_url": "/api/v1/jobs/interp-001",
  "message": "Job accepted for processing"
}
```

> **Note:** LTX-2 requires frame counts following the pattern `frames = 8k + 1`. The API automatically rounds up to the nearest valid frame count and trims the output to the requested `duration_seconds`.

---

### Music Generation

#### `POST /api/v1/music/generate`

**Returns HTTP 202 Accepted**. Generates music using ACE-Step 1.5 (hybrid LM+DiT architecture).

**Request:**

| Field              | Type   | Required | Description                                         | Default       |
| ------------------ | ------ | -------- | --------------------------------------------------- | ------------- |
| `job_id`           | string | ✅       | Unique job identifier                               | -             |
| `prompt`           | string | ✅       | Music style/genre description                       | -             |
| `lyrics`           | string | ❌       | Lyrics for vocal generation (omit for instrumental) | -             |
| `duration_seconds` | float  | ❌       | Duration (10-600 seconds)                           | `30.0`        |
| `seed`             | int    | ❌       | Random seed for reproducibility                     | -             |
| `bpm`              | int    | ❌       | Tempo in BPM (30-300)                               | Auto-detected |
| `key_scale`        | string | ❌       | Musical key, e.g. `"C Major"`, `"Am"`               | Auto-detected |
| `time_signature`   | string | ❌       | `"2"` (2/4), `"3"` (3/4), `"4"` (4/4), `"6"` (6/8)  | Auto-detected |
| `vocal_language`   | string | ❌       | ISO 639-1 code, e.g. `"en"`, `"zh"`, `"ja"`         | Auto-detected |
| `save_url`         | string | ✅       | Presigned PUT URL for output                        | -             |
| `webhook_url`      | string | ❌       | URL to POST when complete                           | -             |
| `item_id`          | string | ❌       | Client identifier (returned in webhook)             | -             |
| `webhook_secret`   | string | ❌       | HMAC signing secret                                 | -             |

> **Note:** When `bpm`, `key_scale`, `time_signature`, or `vocal_language` are omitted, the ACE-Step 1.5 LM uses Chain-of-Thought reasoning to auto-detect optimal values from the prompt and lyrics.

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "queued",
  "message": "Music generation job queued"
}
```

---

### Batch Operations

Batch endpoints allow submitting multiple items in a single request, reducing API overhead from 300+ calls to just 1.

**Features:**

- Per-item webhook callbacks (each item triggers its own webhook)
- Client-provided `item_id` for tracking each item
- Automatic retry-on-failure (failed items requeued once)
- 5-minute auto-expiry (or immediate deletion on collection)

#### `POST /api/v1/batch/image/generate`

Submit a batch of image generation requests (max 500 items).

**Request:**

```json
{
  "batch_id": "batch-abc123",
  "webhook_url": "https://myapp.com/api/gpu-callback",
  "webhook_secret": "optional-hmac-secret",
  "items": [
    {
      "item_id": "scene_001_image",
      "prompt": "A sunset over mountains",
      "aspect_ratio": "16:9",
      "save_url": "https://storage.example.com/1.png"
    },
    {
      "item_id": "scene_002_image",
      "prompt": "A cat on a windowsill",
      "aspect_ratio": "1:1",
      "save_url": "https://storage.example.com/2.png"
    }
  ]
}
```

**Batch Request Fields:**

| Field            | Type   | Required | Description                                        |
| ---------------- | ------ | -------- | -------------------------------------------------- |
| `batch_id`       | string | ✅       | Unique batch identifier                            |
| `webhook_url`    | string | ✅       | URL to POST when each item completes               |
| `webhook_secret` | string | ❌       | HMAC signing secret                                |
| `items`          | array  | ✅       | List of items (max 500 for images, 100 for videos) |

**Item Fields:**

| Field            | Type   | Required | Description                                   |
| ---------------- | ------ | -------- | --------------------------------------------- |
| `item_id`        | string | ✅       | **REQUIRED:** Client identifier for this item |
| `prompt`         | string | ✅       | Text description                              |
| `save_url`       | string | ✅       | Presigned PUT URL                             |
| _(other fields)_ | -      | ❌       | Same as individual endpoint                   |

**Response (202 Accepted):**

```json
{
  "batch_id": "batch-abc123",
  "status": "pending",
  "total_items": 2,
  "status_url": "/api/v1/batch/batch-abc123",
  "message": "Batch accepted for processing (2 images)"
}
```

---

#### `POST /api/v1/batch/image/edit`

Submit a batch of image editing requests (max 500 items).

**Item Fields:** Same as individual `/api/v1/image/edit`.

---

#### `POST /api/v1/batch/video/generate`

Submit a batch of video generation requests (max 100 items).

**Item Fields:** Same as individual `/api/v1/ltx2/generate` (including `start_frame_url`, optional `end_frame_url`).

---

#### `GET /api/v1/batch/{batch_id}`

Get batch status (non-destructive). Shows progress but not results (results delivered via webhook).

**Response:**

```json
{
  "batch_id": "batch-abc123",
  "status": "processing",
  "batch_type": "image_generation",
  "total_items": 100,
  "completed_items": 50,
  "failed_items": 1,
  "pending_items": 30,
  "processing_items": 19,
  "retrying_items": 0,
  "created_at": 1715420000.0,
  "items": [
    {
      "item_index": 0,
      "item_id": "scene_001_image",
      "job_id": "batch-abc123__item_0",
      "status": "completed",
      "retry_count": 0
    },
    {
      "item_index": 1,
      "item_id": "scene_002_image",
      "job_id": "batch-abc123__item_1",
      "status": "processing",
      "retry_count": 0
    }
  ]
}
```

> **Note:** The `result` field is not included in batch status. Results are delivered via webhook for each item.

---

#### `DELETE /api/v1/batch/{batch_id}`

Collect batch results and immediately delete the batch. Use when done polling.

> **Recommended:** Use DELETE instead of GET for final retrieval to prevent 5-minute auto-expiry issues.

**Response:** Same as GET, but batch is deleted after response.

---

### LoRA Management

Endpoints for managing Z-Image LoRA models. All require authentication.

#### `GET /api/v1/loras/z-image`

List available LoRA models for Z-Image generation.

**Response:**

```json
[
  {
    "name": "multiple-angles",
    "size_bytes": 157286400,
    "modified_time": 1715420000.0
  }
]
```

---

#### `POST /api/v1/loras/z-image/upload`

Upload a new LoRA model (`.safetensors` file only).

**Request:** Multipart file upload with `file` field.

**Response (201 Created):**

```json
{
  "status": "success",
  "message": "LoRA multiple-angles.safetensors uploaded successfully"
}
```

---

#### `PUT /api/v1/loras/z-image/{lora_name}`

Rename an existing LoRA model.

| Query Param | Type   | Required | Description       |
| ----------- | ------ | -------- | ----------------- |
| `new_name`  | string | ✅       | New name for LoRA |

**Response:**

```json
{
  "status": "success",
  "message": "Renamed to new-name"
}
```

---

#### `DELETE /api/v1/loras/z-image/{lora_name}`

Delete a LoRA model.

**Response:**

```json
{
  "status": "success",
  "message": "Deleted multiple-angles"
}
```

---

### GPU Monitoring

#### `GET /api/v1/gpu/status`

Get detailed GPU memory and utilization information.

**Response:**

```json
{
  "available": true,
  "cuda_version": "12.1",
  "driver_version": null,
  "device_count": 1,
  "devices": [
    {
      "device_index": 0,
      "name": "NVIDIA A100-SXM4-80GB",
      "total_gb": 79.15,
      "used_gb": 24.32,
      "free_gb": 54.83,
      "usage_percent": 30.7,
      "temperature_celsius": null,
      "utilization_percent": null
    }
  ],
  "total_memory_gb": 79.15,
  "total_used_gb": 24.32,
  "total_free_gb": 54.83
}
```

---

#### `POST /api/v1/gpu/clear-cache`

Force clear CUDA cache and run garbage collection. Useful after OOM errors.

**Response:**

```json
{
  "status": "success",
  "freed_mb": 256.3,
  "used_before_gb": 24.32,
  "used_after_gb": 24.07
}
```

---

### System Status

#### `GET /api/v1/system/status`

Get comprehensive system and GPU status. **Requires authentication.**

**Response:**

```json
{
  "system": {
    "os": "Linux",
    "os_version": "5.15.0",
    "python_version": "3.11.0",
    "cpu_count": 12,
    "hostname": "gpu-server-01"
  },
  "gpu": {
    "name": "NVIDIA A100-SXM4-80GB",
    "memory_total_gb": 79.15,
    "memory_used_gb": 24.32,
    "memory_free_gb": 54.83,
    "memory_usage_percent": 30.7,
    "temperature_celsius": 42.0,
    "gpu_utilization_percent": 15.0,
    "cuda_version": "12.1",
    "driver_version": "535.129.03"
  },
  "mode": {
    "mode": "image_generation",
    "is_busy": false,
    "active_job_id": null,
    "loaded_models": ["z-image-turbo"]
  },
  "concurrency_limits": {
    "max_concurrent_image_generations": 1,
    "max_concurrent_video_generations": 1
  },
  "mock_mode": false
}
```

---

### Download Status

Endpoints for monitoring model download progress. Useful during initial setup.

#### `GET /api/v1/download/status`

Get current model download status. **No authentication required.**

**Response:**

```json
{
  "status": "downloading",
  "ready": false,
  "total_models": 7,
  "completed_models": 3,
  "current_model": "ltx2-checkpoint",
  "models": {
    "z-image-turbo": {
      "model_name": "z-image-turbo",
      "status": "completed",
      "progress_percent": 100.0,
      "error": null
    }
  },
  "started_at": "2025-01-15T10:00:00",
  "completed_at": null,
  "error": null
}
```

---

#### `POST /api/v1/download/retry`

Retry downloading any failed models. **Requires authentication.**

**Response:** Same as `GET /api/v1/download/status`.

---

## Error Handling

### HTTP Status Codes

| Code  | Meaning                                 |
| ----- | --------------------------------------- |
| `201` | Created (LoRA upload)                   |
| `202` | Accepted - Job queued successfully      |
| `400` | Bad Request - Invalid parameters        |
| `401` | Unauthorized - Missing/invalid API key  |
| `404` | Not Found - Job or resource not found   |
| `409` | Conflict - System busy / already exists |
| `429` | Too Many Requests (Queue Full)          |
| `500` | Internal Server Error                   |
| `503` | Service Unavailable (mode switching)    |

### Job Error Codes (in `GET /jobs/{id}`)

| Code                | Description                           |
| ------------------- | ------------------------------------- |
| `GPU_OUT_OF_MEMORY` | VRAM exhausted. Retry with lower res. |
| `JOB_TIMEOUT`       | Processing took too long.             |
| `GENERATION_FAILED` | Internal model error.                 |

---

## Changelog

### v0.6.0

- **API Documentation**: Comprehensive update to match codebase
- **Video Generation**: Added `/api/v1/video/generate` endpoint (simplified video generation)
- **Keyframe Interpolation**: Added `/api/v1/ltx2/interpolate` endpoint for multi-keyframe video generation
- **LoRA Management**: Added CRUD endpoints for Z-Image LoRA models (`/api/v1/loras/z-image`)
- **GPU Monitoring**: Added `/api/v1/gpu/status` and `/api/v1/gpu/clear-cache` endpoints
- **System Status**: Added `/api/v1/system/status` endpoint
- **Download Status**: Added `/api/v1/download/status` and `/api/v1/download/retry` endpoints
- **Readiness Check**: Added `/health/ready` endpoint for VM provisioning
- **Settings**: Added `GET/POST /api/v1/settings/vram-mode` endpoint documentation
- **Mode Switch**: Added `POST /api/v1/mode/switch` endpoint documentation

### v0.5.0

- **Music Generation**: Added `/api/v1/music/generate` endpoint using ACE-Step 1.5
- **Audio VRAM Mode**: New `audio_creation` mode for dedicated music generation (~4GB)

### v0.4.0

- **Mandatory Webhooks**: All generation endpoints now require `webhook_url`
- **Per-item Callbacks**: Results delivered via webhook for each item (single or batch)
- **Job Cleanup**: Job data deleted after successful webhook delivery
- **HMAC Signing**: Optional `webhook_secret` for payload verification
- **Batch item_id**: Each batch item requires a client-provided `item_id`

### v0.3.0

- **Queue System**: Replaced fail-fast concurrency with a robust Job Queue.
- **Async API**: All generation endpoints now return `202 Accepted` and require polling.
- **Smart Scheduling**: Dynamic Mode prioritizes grouping jobs to minimize switching.
- **VRAM Modes**: Added Static vs Dynamic VRAM settings.

### v0.2.0

- Removed Stream-DiffVSR upscaler
- Native 1080p support in LTX-2

### v0.1.0

- Initial release
