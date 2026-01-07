# Model Downloads for Vid-Bolt GPU API

This directory contains AI model weights used by the GPU API for generation tasks.

## Directory Structure

```
models/
├── README.md                     # This file
├── z-image-turbo/                # Z-Image Turbo (text-to-image)
│   ├── transformer/              # DiT transformer weights
│   ├── vae/                      # VAE encoder/decoder
│   ├── text_encoder/             # Qwen2 text encoder
│   ├── tokenizer/                # Tokenizer files
│   └── scheduler/                # Scheduler config
├── qwen-image-edit-2511/         # Qwen-Image-Edit-2511 (image editing)
│   ├── transformer/              # DiT transformer weights
│   ├── vae/                      # VAE encoder/decoder
│   ├── text_encoder/             # Qwen2.5-VL text encoder
│   └── ...                       # Other model files
└── loras/
    ├── z-image-turbo/            # LoRAs for Z-Image
    └── qwen-image-edit-2511/     # LoRAs for Qwen-Image-Edit
        └── Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors
```

---

## Z-Image Turbo Download

**Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  
**Size**: ~12GB  
**Requirements**: 16GB VRAM (bfloat16), CUDA GPU

### Option 1: HuggingFace CLI (Recommended)

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download to models/z-image-turbo/
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir models/z-image-turbo --local-dir-use-symlinks False
```

### Option 2: Git LFS

```bash
# Make sure Git LFS is installed
git lfs install

# Clone directly into models directory
cd models
git clone https://huggingface.co/Tongyi-MAI/Z-Image-Turbo z-image-turbo
```

### Option 3: Python Script

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    local_dir="models/z-image-turbo",
    local_dir_use_symlinks=False,
)
```

---

## Qwen-Image-Edit-2511 Download (LightX2V)

**Model**: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)  
**8-Step LORA**: [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)  
**Size**: Base ~20GB, LORA ~500MB  
**Requirements**: 16GB VRAM (with offloading), 24GB recommended

> **Note**: This model uses the LightX2V framework for accelerated inference. The 8-step distilled LORA provides **~42x speedup** compared to the base 40-step model.

### Step 1: Download Base Model

```bash
huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
    --local-dir models/qwen-image-edit-2511 \
    --local-dir-use-symlinks False
```

### Step 2: Download 8-Step Distilled LORA

```bash
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
    --local-dir models/loras/qwen-image-edit-2511 \
    --local-dir-use-symlinks False
```

### Verification

```bash
# Check base model
ls models/qwen-image-edit-2511/
# Expected: transformer, vae, text_encoder, etc.

# Check LORA weights
ls models/loras/qwen-image-edit-2511/
# Expected: Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors
```

---

## LoRA Weights

Place LoRA weights in the `loras/` directory:

```
models/loras/
├── my-style-lora.safetensors
├── anime-lora.safetensors
└── ...
```

### Popular LoRA Sources

LoRA files compatible with Z-Image can be found at:

- [Civitai](https://civitai.com/) - Community LoRAs
- [Hugging Face](https://huggingface.co/models?other=lora) - Official LoRAs

> **Note**: Z-Image-Turbo uses a different architecture than Stable Diffusion. Ensure LoRAs are specifically trained for Z-Image or compatible architectures.

---

## Verification

After downloading, verify the model structure:

```bash
# Should show these subdirectories:
ls models/z-image-turbo/
# Expected: scheduler  text_encoder  tokenizer  transformer  vae
```

Check essential files exist:

```bash
# Transformer weights (sharded)
ls models/z-image-turbo/transformer/*.safetensors

# VAE weights
ls models/z-image-turbo/vae/*.safetensors

# Text encoder
ls models/z-image-turbo/text_encoder/*.safetensors
```

---

## GPU Requirements

### Z-Image Turbo

| Mode               | VRAM Required  |
| ------------------ | -------------- |
| bfloat16 (default) | ~16GB          |
| float16            | ~16GB          |
| CPU offload        | ~8GB GPU + RAM |

### Qwen-Image-Edit-2511 (LightX2V)

| Configuration             | VRAM Required  |
| ------------------------- | -------------- |
| Full precision            | ~24GB          |
| With text encoder offload | ~16GB          |
| With full CPU offload     | ~8GB GPU + RAM |

> **Tip**: For best performance with LightX2V, use the 8-step distilled LORA which eliminates the need for CFG (classifier-free guidance), effectively halving the memory requirements.
