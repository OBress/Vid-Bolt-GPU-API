# Vid-Bolt GPU API

A high-performance FastAPI backend for AI-powered image and video generation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Webhooks](#webhooks)
- [Mode System](#mode-system)
- [Endpoints](#endpoints)
  - [Health & System](#health--system)
  - [Mode Management](#mode-management)
  - [Image Generation](#image-generation)
  - [Image Editing](#image-editing)
  - [Video Generation](#video-generation)
  - [LTX-2 Video Generation](#ltx-2-video-generation)
  - [Batch Operations](#batch-operations)
- [Error Handling](#error-handling)
- [Configuration](#configuration)
- [System Settings](#system-settings)

---

## Overview

Vid-Bolt GPU API provides AI-powered generation capabilities:

| Capability           | Model                | Description                              |
| -------------------- | -------------------- | ---------------------------------------- |
| **Text-to-Image**    | Z-Image Turbo        | Generate images from text prompts        |
| **Image Editing**    | Qwen-Image-Edit-2511 | Edit images with AI instructions         |
| **Video Generation** | LTX-2 19B            | Generate videos from images (720p/1080p) |

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
│     IMAGE MODE       │          VIDEO MODE              │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │ Z-Image Turbo  │  │  │ LTX-2 19B                  │  │
│  │ (text-to-img)  │  │  │ (image-to-video, 720p/1080p│  │
│  ├────────────────┤  │  │ with native 2x upsampling) │  │
│  │ Qwen-Image-Edit│  │  └────────────────────────────┘  │
│  │ (image editing)│  │                                  │
│  └────────────────┘  │                                  │
└──────────────────────┴──────────────────────────────────┘
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

**Exception:** The `/health` endpoint does not require authentication.

---

## Webhooks

> **IMPORTANT:** All generation endpoints require a `webhook_url`. Results are delivered via webhook only - job data is deleted after successful webhook delivery.

### How It Works

1. Submit a generation request with a `webhook_url`
2. Poll `/api/v1/jobs/{job_id}` for **progress only** (no result in response)
3. Receive webhook callback when job completes (success or failure)
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

Configurable via `/api/v1/settings/vram-mode`:

| Mode               | Models Loaded      | VRAM Usage | Best For                   |
| ------------------ | ------------------ | ---------- | -------------------------- |
| `image_generation` | Z-Image Turbo only | ~8GB       | Text-to-image workloads    |
| `image_editing`    | LightX2V only      | ~12GB      | Image editing/inpainting   |
| `video_generation` | LTX-2 only         | ~20GB      | Video generation           |
| `all`              | All models         | ~40GB+     | High-VRAM GPUs (A100/H100) |

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

4. **all**:
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

## Endpoints

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

| Field                | Type         | Description                                                                   |
| -------------------- | ------------ | ----------------------------------------------------------------------------- |
| `mode`               | string       | Current mode (`image_generation`, `image_editing`, `video_generation`, `all`) |
| `is_busy`            | bool         | Whether a job is currently running                                            |
| `is_switching`       | bool         | Whether mode switch is in progress                                            |
| `switching_target`   | string\|null | Target mode when switching                                                    |
| `switching_step`     | string\|null | Current switching step description                                            |
| `switching_progress` | float\|null  | Progress 0.0-1.0 when switching                                               |

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
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `prompt` | string | ✅ | Text description (max 2000 chars) | - |
| `aspect_ratio` | string | ❌ | `16:9`, `9:16`, `1:1`, `4:3`, `3:4` | `16:9` |
| `save_url` | string | ✅ | Presigned PUT URL for output | - |
| `webhook_url` | string | ✅ | **REQUIRED:** URL to POST when complete | - |
| `item_id` | string | ❌ | Client identifier (returned in webhook) | `job_id` |
| `webhook_secret` | string | ❌ | HMAC signing secret | - |

**Response (Immediate - 202 Accepted):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "status_url": "/api/v1/jobs/550e8400-e29b...",
  "message": "Job accepted for processing"
}
```

> **Note:** Results are delivered via webhook only. The status endpoint shows progress but not the final result.

---

### Image Editing

#### `POST /api/v1/image/edit`

**Returns HTTP 202 Accepted**.

**Request:**
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `input_image_url` | string | ✅ | URL of image to edit | - |
| `prompt` | string | ✅ | Edit instruction | - |
| `save_url` | string | ✅ | Presigned PUT URL for output | - |
| `webhook_url` | string | ✅ | **REQUIRED:** URL to POST when complete | - |
| `item_id` | string | ❌ | Client identifier (returned in webhook) | `job_id` |
| `webhook_secret` | string | ❌ | HMAC signing secret | - |

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

### Video Generation (LTX-2)

#### `POST /api/v1/ltx2/generate`

**Returns HTTP 202 Accepted**. The system will automatically switch to Video Mode if needed.

**Request:**
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `start_frame_url` | string | ✅ | URL of starting frame image | - |
| `prompt` | string | ✅ | Motion description | - |
| `end_frame_url` | string | ❌ | Optional URL of end frame for interpolation | - |
| `duration_seconds` | float | ❌ | Video length (0.5-10.0) | `5.0` |
| `frame_rate` | float | ❌ | Frame rate (8.0-60.0) | `24.0` |
| `aspect_ratio` | string | ❌ | `16:9`, `9:16`, `1:1`, `4:3`, `3:4` | `16:9` |
| `width` | int | ❌ | Target width (512-1920, overrides aspect_ratio) | - |
| `height` | int | ❌ | Target height (512-1920, overrides aspect_ratio) | - |
| `negative_prompt` | string | ❌ | What should not appear in the video | `""` |
| `seed` | int | ❌ | Random seed for reproducibility | - |
| `enhance_prompt` | bool | ❌ | Auto-enhance prompt | `false` |
| `save_url` | string | ✅ | Presigned PUT URL | - |
| `webhook_url` | string | ✅ | **REQUIRED:** URL to POST when complete | - |
| `item_id` | string | ❌ | Client identifier (returned in webhook) | `job_id` |
| `webhook_secret` | string | ❌ | HMAC signing secret | - |

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

## Error Handling

### HTTP Status Codes

| Code  | Meaning                                |
| ----- | -------------------------------------- |
| `202` | Accepted - Job queued successfully     |
| `400` | Bad Request - Invalid parameters       |
| `401` | Unauthorized - Missing/invalid API key |
| `429` | Too Many Requests (Queue Full)         |
| `500` | Internal Server Error                  |

### Job Error Codes (in `GET /jobs/{id}`)

| Code                | Description                           |
| ------------------- | ------------------------------------- |
| `GPU_OUT_OF_MEMORY` | VRAM exhausted. Retry with lower res. |
| `JOB_TIMEOUT`       | Processing took too long.             |
| `GENERATION_FAILED` | Internal model error.                 |

---

## Changelog

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
