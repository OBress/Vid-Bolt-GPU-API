# Model Downloads for Vid-Bolt GPU API

This directory contains AI model weights used by the GPU API for generation tasks.

## Directory Structure

```
models/
├── README.md               # This file
├── z-image-turbo/          # Z-Image Turbo model (6B parameters)
│   ├── transformer/        # DiT transformer weights
│   ├── vae/                # VAE encoder/decoder
│   ├── text_encoder/       # Qwen2 text encoder
│   ├── tokenizer/          # Tokenizer files
│   └── scheduler/          # Scheduler config
└── loras/                  # LoRA fine-tuning weights
    └── (your-lora.safetensors)
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

| Mode               | VRAM Required  |
| ------------------ | -------------- |
| bfloat16 (default) | ~16GB          |
| float16            | ~16GB          |
| CPU offload        | ~8GB GPU + RAM |

For lower VRAM GPUs, see [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) for 4GB support.
