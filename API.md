# Vid-Bolt GPU API

A high-performance FastAPI backend for AI-powered image and video generation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Mode System](#mode-system)
- [Endpoints](#endpoints)
  - [Health & System](#health--system)
  - [Mode Management](#mode-management)
  - [Image Generation](#image-generation)
  - [Image Editing](#image-editing)
  - [Video Generation](#video-generation)
  - [LTX-2 Video Generation](#ltx-2-video-generation)
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

## Mode System & Scheduling

The API manages GPU VRAM by possibly loading only one set of models at a time. A **Queue System** manages requests to ensure fairness and efficiency.

### VRAM Loading Modes

Configurable via `/api/v1/settings/vram-mode`:

1. **Dynamic Mode (Default)**:

   - **Behavior**: Loads only Image OR Video models to save VRAM.
   - **Scheduling**: **Grouped**. The queue worker prioritizes jobs matching the _current_ mode to minimize expensive switching.
   - **Switching**: Takes ~30-60s. Occurs automatically when the queue for the current mode is empty.
   - **Best For**: GPUs with < 40GB VRAM.

2. **Static Mode**:
   - **Behavior**: Loads ALL models (Image + Video) simultaneously.
   - **Scheduling**: **Strict FIFO**. Jobs are processed exactly in order of arrival.
   - **Switching**: Instant.
   - **Best For**: High-VRAM GPUs (A100/H100, >40GB).

### Concurrency Limits

The Queue accepts jobs even if the GPU is busy.

| Resource | Limit                 | Behavior                            |
| -------- | --------------------- | ----------------------------------- |
| Queue    | Unbounded (in-memory) | Returns `202 Accepted` immediately. |
| Worker   | 1 Active Job          | Processes one job at a time.        |

---

## Endpoints

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

**Response (Immediate):**

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
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `input_image_url` | string | ✅ | URL of image to edit | - |
| `prompt` | string | ✅ | Edit instruction | - |
| `save_url` | string | ✅ | Presigned PUT URL for output | - |

**Response (Immediate):**

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
| `input_image_url` | string | ✅ | URL of starting frame | - |
| `prompt` | string | ✅ | Motion description | - |
| `duration_seconds` | float | ❌ | Video length (1.0-8.0) | `4.0` |
| `save_url` | string | ✅ | Presigned PUT URL | - |

**Response (Immediate):**

```json
{
  "job_id": "550e8400-e29b...",
  "status": "pending",
  "status_url": "/api/v1/jobs/550e8400-e29b...",
  "message": "Job accepted for processing"
}
```

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
