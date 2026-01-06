# Vid-Bolt GPU API Documentation

This document provides detailed information about the Vid-Bolt GPU API endpoints, including authentication requirements, request parameters, and response structures.

## Authentication

All API endpoints under `/api/v1/*` require authentication using an API key.
Include the `X-API-Key` header in your requests.

```http
X-API-Key: YOUR_API_KEY
```

## Base URL

Where not specified, endpoints are relative to the API base URL.

## Endpoints

### 1. Image Generation

Generate an image from a text prompt.

**Endpoint:** `POST /api/v1/image/generate`

**Description:**
Generates an image based on a text prompt using the configured AI model. The generated image is uploaded to a storage provider (Cloudflare R2), and a public URL is returned.

**Request Body (`application/json`):**

| Field                 | Type    | Required | Description                         | Constraints                               | Default         |
| :-------------------- | :------ | :------- | :---------------------------------- | :---------------------------------------- | :-------------- |
| `job_id`              | string  | Yes      | Unique UUID for the job             | Min length 1                              | -               |
| `prompt`              | string  | Yes      | Text description of the image       | Max 2000 chars                            | -               |
| `aspect_ratio`        | string  | No       | Aspect ratio of the generated image | Enum: `16:9`, `9:16`, `1:1`, `4:3`, `3:4` | `16:9`          |
| `seed`                | integer | No       | Random seed for reproducibility     | -                                         | `null` (random) |
| `num_inference_steps` | integer | No       | Number of diffusion steps           | 1 - 50                                    | `20`            |
| `save_url`            | string  | Yes      | Presigned PUT URL for upload        | -                                         | -               |

**Example Request:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "A futuristic city skyline at sunset, cyberpunk style",
  "aspect_ratio": "16:9",
  "num_inference_steps": 25,
  "save_url": "https://storage.example.com/bucket/image.png?signature=..."
}
```

**Response Body (`application/json`):**

| Field             | Type   | Description                                           |
| :---------------- | :----- | :---------------------------------------------------- |
| `status`          | string | Status of the request (always "completed" on success) |
| `generation_time` | float  | Time taken to generate in seconds                     |
| `save_url`        | string | The public URL where the image was saved              |

**Example Response:**

```json
{
  "status": "completed",
  "generation_time": 2.5,
  "save_url": "https://storage.example.com/bucket/image.png"
}
```

---

### 2. Image Editing

Edit an existing image using AI-powered transformations.

**Endpoint:** `POST /api/v1/image/edit`

**Description:**
Performs edits such as inpainting, style transfer, or background removal on an existing image provided via URL.

**Request Body (`application/json`):**

| Field             | Type    | Required    | Description                      | Constraints                               | Default |
| :---------------- | :------ | :---------- | :------------------------------- | :---------------------------------------- | :------ |
| `job_id`          | string  | Yes         | Unique UUID for the job          | -                                         | -       |
| `input_image_url` | string  | Yes         | URL of the image to edit         | Valid URL                                 | -       |
| `prompt`          | string  | Yes         | Description of the desired edit  | Max 2000 chars                            | -       |
| `aspect_ratio`    | string  | No          | Aspect ratio of the edited image | Enum: `16:9`, `9:16`, `1:1`, `4:3`, `3:4` | `16:9`  |
| `mask_image_url`  | string  | Conditional | URL of the mask image            | Required for inpainting operations        | `null`  |
| `seed`            | integer | No          | Random seed                      | -                                         | `null`  |
| `save_url`        | string  | Yes         | Presigned PUT URL for upload     | -                                         | -       |

**Example Request:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "input_image_url": "https://example.com/original.png",
  "prompt": "Make it look like a Van Gogh painting",
  "aspect_ratio": "16:9",
  "save_url": "https://storage.example.com/bucket/edited.png?signature=..."
}
```

**Response Body (`application/json`):**

| Field             | Type   | Description                 |
| :---------------- | :----- | :-------------------------- |
| `status`          | string | Status (always "completed") |
| `generation_time` | float  | Processing time in seconds  |
| `save_url`        | string | URL of the edited image     |

---

### 3. Video Generation

Generate a video from an input image with AI-powered motion.

**Endpoint:** `POST /api/v1/video/generate`

**Description:**
Creates a short video clip animating the provided input image based on a text prompt.

**Request Body (`application/json`):**

| Field              | Type    | Required | Description                         | Constraints                               | Default |
| :----------------- | :------ | :------- | :---------------------------------- | :---------------------------------------- | :------ |
| `job_id`           | string  | Yes      | Unique UUID for the job             | -                                         | -       |
| `input_image_url`  | string  | Yes      | URL of the starting frame           | Valid URL                                 | -       |
| `prompt`           | string  | Yes      | Description of the motion/action    | Max 2000 chars                            | -       |
| `duration_seconds` | float   | No       | Length of video                     | 1.0 - 8.0                                 | `4.0`   |
| `fps`              | integer | No       | Frames per second                   | 8, 12, 16, 24, or 30                      | `24`    |
| `aspect_ratio`     | string  | No       | Aspect ratio of the generated video | Enum: `16:9`, `9:16`, `1:1`, `4:3`, `3:4` | `16:9`  |
| `seed`             | integer | No       | Random seed                         | -                                         | `null`  |
| `end_image_url`    | string  | No       | Optional URL for the final frame    | Valid URL                                 | `null`  |
| `save_url`         | string  | Yes      | Presigned PUT URL for upload        | -                                         | -       |

**Example Request:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "input_image_url": "https://example.com/start_frame.png",
  "prompt": "Camera pans slowly to the right, clouds moving",
  "duration_seconds": 4.0,
  "fps": 24,
  "aspect_ratio": "16:9",
  "save_url": "https://storage.example.com/bucket/video.mp4?signature=..."
}
```

**Response Body (`application/json`):**

| Field             | Type   | Description                 |
| :---------------- | :----- | :-------------------------- |
| `status`          | string | Status (always "completed") |
| `generation_time` | float  | Processing time in seconds  |
| `save_url`        | string | URL of the generated video  |

---

## Error Responses

Errors are returned with an appropriate HTTP status code (400, 401, 500) and a JSON body.

**Error Body:**

```json
{
  "status": "failed",
  "error_code": "ERROR_CODE_STRING",
  "error_message": "Human readable error description"
}
```

**Common Error Codes:**

- `VALIDATION_ERROR`: Invalid request parameters (e.g., missing fields, bad types).
- `AUTHENTICATION_ERROR`: Missing or invalid API key.
- `INTERNAL_ERROR`: Unexpected server-side error.
