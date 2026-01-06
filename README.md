# Vid-Bolt GPU API

A FastAPI-based GPU image and video generation API. This is the **local development version** with mock implementations - no actual ComfyUI or GPU code yet.

## Overview

This API provides endpoints for:

- **Image Generation**: Generate images from text prompts
- **Image Editing**: Edit existing images (inpaint, outpaint, style transfer, etc.)
- **Video Generation**: Create videos from images with motion

All generated outputs are uploaded via presigned URLs provided by the calling client.

## Prerequisites

- Python 3.11+
- pip

## Quick Start

### 1. Clone and Setup

```bash
cd Vid-Bolt-GPU-API

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev
# Or: pip install -r requirements-dev.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values (API key, etc.)
```

### 3. Run the Server

```bash
make run
# Or: uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health Check

```bash
# No authentication required
curl http://localhost:8000/health

# With authentication
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/status
```

### Image Generation

```bash
curl -X POST http://localhost:8000/api/v1/image/generate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "prompt": "A beautiful sunset over mountains",
    "width": 1280,
    "height": 720
  }'
```

### Image Editing

```bash
curl -X POST http://localhost:8000/api/v1/image/edit \
  -H "X-API-Key: your-api-key" \
  -F "job_id=550e8400-e29b-41d4-a716-446655440001" \
  -F "input_image=@your_image.png" \
  -F "prompt=Make it look vintage" \
  -F "edit_type=style_transfer"
```

### Video Generation

```bash
curl -X POST http://localhost:8000/api/v1/video/generate \
  -H "X-API-Key: your-api-key" \
  -F "job_id=550e8400-e29b-41d4-a716-446655440002" \
  -F "input_image=@your_image.png" \
  -F "prompt=Slow zoom in" \
  -F "duration_seconds=4.0" \
  -F "fps=24"
```

## Development

### Running Tests

```bash
make test
# Or: pytest tests/ -v --cov=app
```

### Linting

```bash
make lint    # Check for issues
make format  # Auto-format code
```

## Environment Variables

| Variable                     | Description                 | Default     |
| ---------------------------- | --------------------------- | ----------- |
| `MOCK_MODE`                  | Enable mock implementations | `true`      |
| `API_KEY`                    | API authentication key      | Required    |
| `COMFY_HOST`                 | ComfyUI host                | `127.0.0.1` |
| `COMFY_PORT`                 | ComfyUI port                | `8188`      |
| `MAX_IMAGE_SIZE_MB`          | Max upload size             | `10`        |
| `MAX_VIDEO_DURATION_SECONDS` | Max video length            | `8`         |
| `LOG_LEVEL`                  | Logging level               | `INFO`      |

## Architecture

```
vid-bolt-gpu-api/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration management
│   ├── dependencies.py   # Dependency injection
│   ├── exceptions.py     # Custom exceptions
│   ├── routers/          # API endpoints
│   ├── models/           # Pydantic schemas
│   ├── services/         # Business logic
│   └── utils/            # Utilities
└── tests/                # Test suite
```

## Mock Mode

When `MOCK_MODE=true` (default), the API:

- Simulates processing delays (2-10 seconds depending on operation)
- Generates placeholder images with gradient backgrounds and text overlays
- Creates simple MP4 videos with static backgrounds
- Uses presigned URLs from the client for output storage

## License

MIT
