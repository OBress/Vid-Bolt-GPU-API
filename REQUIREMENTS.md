# System Requirements & Dependencies

> **Target Platform**: NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7) on Ubuntu 22.04 LTS (jammy) x86_64

---

## Hardware Requirements

| Requirement          | Specification                                                |
| -------------------- | ------------------------------------------------------------ |
| **GPU**              | NVIDIA RTX PRO 6000 Blackwell (24,064 CUDA Cores, 96GB VRAM) |
| **GPU Architecture** | Blackwell (Compute Capability 10.0)                          |
| **System RAM**       | Minimum 64GB, Recommended 128GB                              |
| **Storage**          | 500GB+ SSD (for models)                                      |
| **OS**               | Ubuntu 22.04 LTS x86_64                                      |

---

## System-Level Dependencies

### NVIDIA Driver & CUDA Toolkit

| Component         | Required Version | Notes                                  |
| ----------------- | ---------------- | -------------------------------------- |
| **NVIDIA Driver** | R570+            | Blackwell minimum driver               |
| **CUDA Toolkit**  | 12.8+            | Required for Blackwell (sm_100/sm_120) |
| **cuDNN**         | 9.0+             | Deep learning primitives               |
| **NCCL**          | 2.21+            | Multi-GPU communication                |
| **TensorRT**      | 10.0+            | Optional, for inference optimization   |

```bash
# Install CUDA 12.8+ on Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8 nvidia-driver-570
```

---

## Python Environment

| Requirement | Version                                |
| ----------- | -------------------------------------- |
| **Python**  | 3.12+ (3.13 recommended for Blackwell) |
| **pip**     | 24.0+                                  |
| **uv**      | 0.5+ (optional, for faster installs)   |

```bash
# Recommended: Create dedicated environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## Core Dependencies

### PyTorch Ecosystem (Blackwell-Compatible)

| Package       | Version    | Index URL                                |
| ------------- | ---------- | ---------------------------------------- |
| `torch`       | `==2.8.0`  | `https://download.pytorch.org/whl/cu128` |
| `torchvision` | `==0.23.0` | `https://download.pytorch.org/whl/cu128` |
| `torchaudio`  | `==2.8.0`  | `https://download.pytorch.org/whl/cu128` |
| `triton`      | `>=3.3.0`  | PyPI                                     |

```bash
# PyTorch 2.8.0 with CUDA 12.8 for Blackwell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install triton>=3.3.0
```

### Memory Optimization

| Package        | Version          | Purpose                        |
| -------------- | ---------------- | ------------------------------ |
| `xformers`     | `>=0.0.32.post2` | Memory-efficient attention     |
| `accelerate`   | `>=0.34.0`       | Model parallelism & offloading |
| `bitsandbytes` | `>=0.44.0`       | 8-bit optimizers               |

```bash
pip install xformers>=0.0.32.post2 accelerate>=0.34.0 bitsandbytes>=0.44.0
```

---

## AI/ML Framework Dependencies

### Diffusers & HuggingFace Ecosystem

| Package           | Version    | Purpose                                   |
| ----------------- | ---------- | ----------------------------------------- |
| `diffusers`       | `>=0.33.0` | ZImagePipeline, ControlNet, StreamDiffVSR |
| `transformers`    | `>=4.51.0` | Text encoders (T5, Gemma, CLIP)           |
| `safetensors`     | `>=0.4.5`  | Safe model serialization                  |
| `huggingface-hub` | `>=0.26.0` | Model downloads                           |
| `peft`            | `>=0.12.0` | LoRA/adapter support                      |
| `tokenizers`      | `>=0.20.0` | Fast tokenization                         |

```bash
# Diffusers 0.33.0+ required for ZImagePipeline
pip install git+https://github.com/huggingface/diffusers
pip install transformers>=4.51.0 safetensors>=0.4.5 huggingface-hub>=0.26.0 peft>=0.12.0
```

---

## Model-Specific Dependencies

### Z-Image Turbo (Text-to-Image)

| Package        | Version    | Notes                                          |
| -------------- | ---------- | ---------------------------------------------- |
| `diffusers`    | `>=0.33.0` | Includes `ZImagePipeline` (PRs #12703, #12715) |
| `torch`        | `>=2.5.0`  | Native Diffusers inference                     |
| `transformers` | `>=4.51.0` | Text encoder                                   |
| `Pillow`       | `>=11.0.0` | Image processing                               |

---

### LightX2V (Image Editing - Qwen-Image-Edit-2511)

| Package          | Version    | Notes                                                        |
| ---------------- | ---------- | ------------------------------------------------------------ |
| `lightx2v`       | Latest     | `pip install -v git+https://github.com/ModelTC/LightX2V.git` |
| `torch`          | `<=2.8.0`  | Framework constraint                                         |
| `torchvision`    | `<=0.23.0` | Vision operations                                            |
| `torchaudio`     | `<=2.8.0`  | Audio handling                                               |
| `sgl-kernel`     | Latest     | SGLang optimized kernels                                     |
| `einops`         | `>=0.8.0`  | Tensor operations                                            |
| `loguru`         | `>=0.7.0`  | Logging                                                      |
| `imageio`        | `>=2.36.0` | Image I/O                                                    |
| `imageio-ffmpeg` | `>=0.6.0`  | FFmpeg bindings                                              |
| `decord`         | Latest     | Video decoding                                               |

```bash
pip install -v git+https://github.com/ModelTC/LightX2V.git
```

---

### LTX-2 (Video Generation)

| Package         | Version    | Notes                     |
| --------------- | ---------- | ------------------------- |
| `ltx-core`      | `1.0.0`    | Core model implementation |
| `ltx-pipelines` | `1.0.0`    | High-level pipelines      |
| `torch`         | `~=2.7`    | Framework                 |
| `scipy`         | `>=1.14`   | Scientific computing      |
| `einops`        | Latest     | Tensor reshaping          |
| `av`            | `>=14.1.0` | Video encoding            |
| `tqdm`          | Latest     | Progress bars             |
| `Pillow`        | Latest     | Image handling            |

```bash
# Install from LTX-2 repository (vendored in repos/)
cd repos/LTX-2
uv sync --frozen
# Or with pip:
pip install -e repos/LTX-2/packages/ltx-core
pip install -e repos/LTX-2/packages/ltx-pipelines
```

---

### Stream-DiffVSR (Video Upscaling)

| Package           | Version       | Notes                      |
| ----------------- | ------------- | -------------------------- |
| `torch`           | `==2.8.0`     | CUDA 12.8 build            |
| `torchvision`     | `==0.23.0`    | RAFT optical flow          |
| `diffusers`       | `>=0.31.0`    | Pipeline base              |
| `xformers`        | Latest        | Memory optimization        |
| `opencv-python`   | `==4.10.0.84` | Frame processing           |
| `onnxruntime-gpu` | `==1.20.1`    | ONNX inference             |
| `basicsr`         | `==1.4.2`     | Super-resolution utilities |
| `einops`          | `==0.8.0`     | Tensor ops                 |
| `mmengine`        | `==0.10.5`    | OpenMMLab engine           |
| `mmcv`            | `==2.2.0`     | OpenMMLab computer vision  |

---

## CUDA Libraries (Runtime)

```bash
pip install cupy-cuda12x \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvtx-cu12
```

---

## FastAPI Application Dependencies

| Package             | Version     | Purpose               |
| ------------------- | ----------- | --------------------- |
| `fastapi`           | `>=0.115.0` | Web framework         |
| `uvicorn[standard]` | `>=0.32.0`  | ASGI server           |
| `pydantic`          | `>=2.10.0`  | Data validation       |
| `pydantic-settings` | `>=2.6.0`   | Settings management   |
| `python-multipart`  | `>=0.0.18`  | File uploads          |
| `aiofiles`          | `>=24.1.0`  | Async file I/O        |
| `python-dotenv`     | `>=1.0.0`   | Environment variables |
| `boto3`             | `>=1.35.0`  | AWS/R2 storage        |

---

## Media Processing

| Package          | Version    | Purpose                  |
| ---------------- | ---------- | ------------------------ |
| `Pillow`         | `>=11.0.0` | Image processing         |
| `opencv-python`  | `>=4.10.0` | Video frame manipulation |
| `moviepy`        | `>=1.0.3`  | Video editing            |
| `av`             | `>=14.1.0` | Audio/video encoding     |
| `imageio`        | `>=2.36.1` | Image I/O                |
| `imageio-ffmpeg` | `>=0.6.0`  | FFmpeg integration       |

---

## Complete Installation Script

```bash
#!/bin/bash
# Full installation script for RTX PRO 6000 Blackwell on Ubuntu 22.04

set -e

# 1. Create Python environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 2. Install PyTorch 2.8.0 with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install core AI dependencies
pip install triton>=3.3.0
pip install xformers>=0.0.32.post2
pip install accelerate>=0.34.0 bitsandbytes>=0.44.0

# 4. Install Diffusers from source (for ZImagePipeline)
pip install git+https://github.com/huggingface/diffusers
pip install transformers>=4.51.0 safetensors>=0.4.5 peft>=0.12.0

# 5. Install LightX2V
pip install -v git+https://github.com/ModelTC/LightX2V.git

# 6. Install LTX-2 packages (vendored in repos/)
pip install -e repos/LTX-2/packages/ltx-core
pip install -e repos/LTX-2/packages/ltx-pipelines

# 7. Install FastAPI and utilities
pip install fastapi>=0.115.0 uvicorn[standard]>=0.32.0
pip install pydantic>=2.10.0 pydantic-settings>=2.6.0
pip install python-multipart>=0.0.18 aiofiles>=24.1.0
pip install boto3>=1.35.0 python-dotenv>=1.0.0

# 8. Install media processing
pip install Pillow>=11.0.0 opencv-python>=4.10.0
pip install moviepy>=1.0.3 av>=14.1.0

# 9. Install CUDA runtime libraries
pip install cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

# 10. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

---

## Version Compatibility Matrix

| Component        | Z-Image | LightX2V | LTX-2 | Stream-DiffVSR |
| ---------------- | ------- | -------- | ----- | -------------- |
| **Python**       | ≥3.10   | ≥3.10    | ≥3.10 | ≥3.10          |
| **PyTorch**      | ≥2.5.0  | ≤2.8.0   | ~2.7  | 2.8.0          |
| **CUDA**         | 12.1+   | 12.4+    | 12.9  | 12.8           |
| **Diffusers**    | ≥0.33.0 | Any      | N/A   | ≥0.31.0        |
| **Transformers** | ≥4.51.0 | ≥4.45.0  | Any   | ≥4.45.0        |

---

## Blackwell-Specific Optimizations

```python
# Enable FP8 for maximum performance on Blackwell
# In app/config.py:
LTX2_FP8_ENABLED = True

# Use Flash Attention 3 for optimal attention performance
ZIMAGE_ATTENTION_BACKEND = "_flash_3"
LIGHTX2V_ATTN_MODE = "flash_attn3"

# Compile models for faster inference
ZIMAGE_COMPILE = True  # torch.compile on transformer
```

---

## Model Download Commands

```bash
# Z-Image Turbo (~12GB)
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir models/z-image-turbo

# Qwen-Image-Edit-2511 (~14GB)
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/qwen-image-edit-2511
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/loras/qwen-image-edit-2511

# LTX-2 (~40GB total)
huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir models/ltx-2
huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-lora-384.safetensors --local-dir models/ltx-2
huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir models/ltx-2
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir models/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized

# Stream-DiffVSR (auto-downloaded at runtime)
# Model ID: Jamichsu/Stream-DiffVSR
```

---

## Troubleshooting

### Blackwell GPU Not Detected

```bash
# Ensure driver version is R570+
nvidia-smi
# Should show: RTX PRO 6000, Driver 570.xx+, CUDA 12.8
```

### CUDA Version Mismatch

```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
# Should output: 12.8
```

### xFormers Compatibility

```bash
# For Blackwell, install from PyTorch index
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

---

## Summary of Pinned Versions

```txt
# Core Framework (Blackwell)
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
triton>=3.3.0

# AI/ML
diffusers>=0.33.0
transformers>=4.51.0
accelerate>=0.34.0
safetensors>=0.4.5
peft>=0.12.0

# Memory Optimization
xformers>=0.0.32.post2
bitsandbytes>=0.44.0

# Media
Pillow>=11.0.0
opencv-python>=4.10.0
moviepy>=1.0.3
av>=14.1.0

# FastAPI
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.10.0
```
