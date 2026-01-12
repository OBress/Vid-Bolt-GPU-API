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
│  Mode Manager - Controls which models are loaded         │
│  (Supports Static All-Load or Dynamic Loading)           │
├──────────────────────┬──────────────────────────────────┤
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

## Mode System

The API operates in either **Image Mode** or **Video Mode** to efficiently manage GPU VRAM.

### Current Modes

| Mode    | Loaded Models                  | Endpoints Available                 |
| ------- | ------------------------------ | ----------------------------------- |
| `image` | Z-Image Turbo, Qwen-Image-Edit | `/api/v1/image/*`                   |
| `video` | LTX-2 19B                      | `/api/v1/video/*`, `/api/v1/ltx2/*` |

### Mode Behavior

The API supports two VRAM loading strategies, configurable via `/api/v1/settings/vram-mode`:

1. **Static Mode (Default)**:

   - All models (Image + Video) are loaded at startup.
   - **Switching is INSTANT**.
   - Requires high VRAM (approx 24GB+).

2. **Dynamic Mode**:
   - Only models for the current mode are loaded.
   - **Switching takes ~30-60 seconds** (unloading/reloading).
   - Saves VRAM, allowing operation on smaller GPUs.

### Concurrency Limits

| Mode  | Max Concurrent Generations         |
| ----- | ---------------------------------- |
| Image | 2 (combined across Z-Image + Qwen) |
| Video | 1 (LTX-2 + upscaling pipeline)     |

---

## Endpoints

### Health & System

#### `GET /health`

Basic health check (no authentication required).

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "mock_mode": false
}
```

---

#### `GET /api/v1/system/status`

Detailed system and GPU status.

**Response:**

```json
{
  "system": {
    "os": "Linux",
    "os_version": "5.15.0",
    "python_version": "3.11.0",
    "cpu_count": 16,
    "hostname": "gpu-server"
  },
  "gpu": {
    "name": "NVIDIA RTX PRO 6000",
    "memory_total_gb": 96.0,
    "memory_used_gb": 24.5,
    "memory_free_gb": 71.5,
    "memory_usage_percent": 25.5,
    "temperature_celsius": 42,
    "gpu_utilization_percent": 0,
    "cuda_version": "12.4",
    "driver_version": "550.54.14"
  },
  "mode": {
    "mode": "image",
    "is_busy": false,
    "active_job_id": null,
    "loaded_models": ["z-image-turbo", "qwen-image-edit-2511"]
  },
  "concurrency_limits": {
    "max_concurrent_image_generations": 2,
    "max_concurrent_video_generations": 1
  },
  "mock_mode": false
}
```

---

### Mode Management

#### `GET /api/v1/mode`

Get current mode status.

**Response:**

```json
{
  "mode": "image",
  "is_busy": false,
  "active_job_id": null,
  "loaded_models": ["z-image-turbo", "qwen-image-edit-2511"]
}
```

---

#### `POST /api/v1/mode/switch`

Switch between Image Mode and Video Mode.

**Request:**

```json
{
  "target_mode": "video"
}
```

**Response:**

```json
{
  "status": "success",
  "previous_mode": "image",
  "current_mode": "video",
  "message": "Successfully switched from image to video mode"
}
```

**Errors:**

- `503` - Mode switch already in progress or system busy

---

### System Settings

#### `GET /api/v1/settings/vram-mode`

Get current VRAM loading strategy.

**Response:**

```json
{
  "mode": "static",
  "description": "Static loading - instant switching, higher VRAM usage"
}
```

#### `POST /api/v1/settings/vram-mode`

Set VRAM loading strategy.

**Request:**

```json
{
  "mode": "dynamic"
}
```

**Response:**

```json
{
  "mode": "dynamic",
  "description": "Dynamic loading - saves VRAM"
}
```

---

### Image Generation

#### `POST /api/v1/image/generate`

Generate an image from a text prompt.

**Requires:** Image Mode

**Request:**
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `prompt` | string | ✅ | Text description (max 2000 chars) | - |
| `aspect_ratio` | string | ❌ | `16:9`, `9:16`, `1:1`, `4:3`, `3:4` | `16:9` |
| `width` | integer | ❌ | Custom width (256-2048) | `null` |
| `height` | integer | ❌ | Custom height (256-2048) | `null` |
| `seed` | integer | ❌ | Random seed for reproducibility | random |
| `num_inference_steps` | integer | ❌ | Diffusion steps (1-50) | `20` |
| `save_url` | string | ✅ | Presigned PUT URL for output | - |

**Example:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "A futuristic city skyline at sunset, cyberpunk style",
  "aspect_ratio": "16:9",
  "save_url": "https://storage.example.com/image.png?sig=..."
}
```

**Response:**

```json
{
  "status": "completed",
  "generation_time": 2.5,
  "save_url": "https://storage.example.com/image.png"
}
```

---

### Image Editing

#### `POST /api/v1/image/edit`

Edit an existing image with AI-powered transformations.

**Requires:** Image Mode

**Request:**
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `input_image_url` | string | ✅ | URL of image to edit | - |
| `prompt` | string | ✅ | Edit instruction (max 2000 chars) | - |
| `aspect_ratio` | string | ❌ | Output aspect ratio | `16:9` |
| `mask_image_url` | string | ❌ | Mask for inpainting | `null` |
| `seed` | integer | ❌ | Random seed | random |
| `save_url` | string | ✅ | Presigned PUT URL for output | - |

**Example:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "input_image_url": "https://example.com/original.png",
  "prompt": "Make it look like a Van Gogh painting",
  "save_url": "https://storage.example.com/edited.png?sig=..."
}
```

**Response:**

```json
{
  "status": "completed",
  "generation_time": 3.2,
  "save_url": "https://storage.example.com/edited.png"
}
```

---

### Video Generation

#### `POST /api/v1/video/generate`

Generate a video from an input image.

**Requires:** Video Mode

**Request:**
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `input_image_url` | string | ✅ | URL of starting frame | - |
| `prompt` | string | ✅ | Motion/action description | - |
| `duration_seconds` | float | ❌ | Video length (1.0-8.0) | `4.0` |
| `fps` | integer | ❌ | 8, 12, 16, 24, or 30 | `24` |
| `aspect_ratio` | string | ❌ | Output aspect ratio | `16:9` |
| `width` | integer | ❌ | Target width (512-1920), e.g. 1920 for 1080p | `null` |
| `height` | integer | ❌ | Target height (512-1920), e.g. 1080 for 1080p | `null` |
| `end_image_url` | string | ❌ | Optional final frame | `null` |
| `seed` | integer | ❌ | Random seed | random |
| `save_url` | string | ✅ | Presigned PUT URL for output | - |

> **Note:** For 1080p video, set `width: 1920` and `height: 1080`. Without explicit dimensions, 16:9 defaults to 720p.

**Example:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "input_image_url": "https://example.com/start.png",
  "prompt": "Camera slowly pans right, clouds moving in background",
  "duration_seconds": 4.0,
  "fps": 24,
  "save_url": "https://storage.example.com/video.mp4?sig=..."
}
```

**Response:**

```json
{
  "status": "completed",
  "generation_time": 45.2,
  "save_url": "https://storage.example.com/video.mp4"
}
```

---

### LTX-2 Video Generation

Advanced video generation with LTX-2 19B model.

#### `POST /api/v1/ltx2/generate`

Image-to-video generation with optional end frame.

**Requires:** Video Mode

**Request:**
| Field | Type | Required | Description | Default |
|-------|------|----------|-------------|---------|
| `job_id` | string | ✅ | Unique job identifier | - |
| `input_image_url` | string | ✅ | URL of starting frame | - |
| `prompt` | string | ✅ | Motion description | - |
| `negative_prompt` | string | ❌ | What to avoid | `null` |
| `duration_seconds` | float | ❌ | Video length (1.0-8.0) | `4.0` |
| `frame_rate` | float | ❌ | Frames per second | `24.0` |
| `aspect_ratio` | string | ❌ | Output aspect ratio | `16:9` |
| `width` | integer | ❌ | Target width (512-1920), e.g. 1920 for 1080p | `null` |
| `height` | integer | ❌ | Target height (512-1920), e.g. 1080 for 1080p | `null` |
| `end_image_url` | string | ❌ | Final frame for interpolation | `null` |
| `seed` | integer | ❌ | Random seed | random |
| `enhance_prompt` | boolean | ❌ | AI prompt enhancement | `false` |
| `save_url` | string | ✅ | Presigned PUT URL | - |

> **Note:** For 1080p video, set `width: 1920` and `height: 1080`. Without explicit dimensions, 16:9 defaults to 720p.

**Response:**

```json
{
  "status": "completed",
  "generation_time": 52.3,
  "save_url": "https://storage.example.com/video.mp4",
  "duration_seconds": 4.0,
  "has_audio": false
}
```

---

#### `POST /api/v1/ltx2/interpolate`

Keyframe interpolation between multiple images.

**Requires:** Video Mode

**Request:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440003",
  "prompt": "Smooth transition between scenes",
  "keyframes": [
    {
      "image_url": "https://example.com/frame1.png",
      "frame_index": 0,
      "strength": 1.0
    },
    {
      "image_url": "https://example.com/frame2.png",
      "frame_index": 48,
      "strength": 0.8
    },
    {
      "image_url": "https://example.com/frame3.png",
      "frame_index": 96,
      "strength": 1.0
    }
  ],
  "duration_seconds": 4.0,
  "frame_rate": 24.0,
  "width": 1920,
  "height": 1080,
  "save_url": "https://storage.example.com/interpolated.mp4?sig=..."
}
```

> **Note:** For 1080p, include `width` and `height`. Keyframe images should match target resolution.

---

## Error Handling

All errors return a consistent JSON structure:

```json
{
  "status": "failed",
  "error_code": "ERROR_CODE",
  "error_message": "Human-readable description"
}
```

### HTTP Status Codes

| Code  | Meaning                                  |
| ----- | ---------------------------------------- |
| `400` | Bad Request - Invalid parameters         |
| `401` | Unauthorized - Missing/invalid API key   |
| `503` | Service Unavailable - Wrong mode or busy |
| `500` | Internal Server Error                    |

### Error Codes

| Code                   | Description                     |
| ---------------------- | ------------------------------- |
| `VALIDATION_ERROR`     | Invalid request parameters      |
| `AUTHENTICATION_ERROR` | Missing or invalid API key      |
| `MODE_ERROR`           | Request requires different mode |
| `BUSY_ERROR`           | System busy with another job    |
| `INTERNAL_ERROR`       | Unexpected server error         |

---

## Configuration

### Environment Variables

| Variable               | Required | Default          | Description                |
| ---------------------- | -------- | ---------------- | -------------------------- |
| `MOCK_MODE`            | ❌       | `true`           | Enable mock mode (no GPU)  |
| `API_KEY`              | ✅       | -                | API authentication key     |
| `LOG_LEVEL`            | ❌       | `INFO`           | Logging verbosity          |
| `CORS_ALLOWED_ORIGINS` | ❌       | `localhost:3000` | Comma-separated origins    |
| `DEFAULT_MODEL_MODE`   | ❌       | `image`          | Startup mode (image/video) |

### Model Configuration

Model paths and inference parameters are hardcoded in `app/config.py` for security and simplicity:

```python
# app/config.py
class InferenceConfig:
    MAX_CONCURRENT_IMAGE_GENERATIONS = 2
    MAX_CONCURRENT_VIDEO_GENERATIONS = 1
    # ... other settings
```

---

## Rate Limits

The API enforces concurrency limits, not rate limits:

- **Image Mode:** Max 2 concurrent generations
- **Video Mode:** Max 1 concurrent generation

Exceeded requests receive `503 Service Unavailable`.

---

## Changelog

### v0.2.0

- Removed Stream-DiffVSR upscaler (using LTX-2 native 2x upsampling)
- Added `width` and `height` parameters for explicit 1080p support
- Fixed video resolution handling

### v0.1.0

- Initial release
- Image generation (Z-Image Turbo)
- Image editing (Qwen-Image-Edit)
- Video generation (LTX-2)
- Dynamic mode switching
- GPU monitoring endpoint
