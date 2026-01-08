# Z-Image Turbo Model Weights

⚡ **Z-Image Turbo** - An efficient 6B parameter image generation model achieving sub-second inference with only 8 NFEs (Number of Function Evaluations).

## Requirements

- **Disk Space**: ~13GB
- **VRAM**: 16GB minimum (with CPU offloading), 24GB recommended
- **GPU**: NVIDIA GPU with CUDA support

## Download Methods

Choose **one** of the following methods:

---

### Method 1: HuggingFace CLI (Recommended)

```bash
# Install/update HuggingFace CLI
pip install -U huggingface_hub

# Download to this directory
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir ./

# OR with high-performance xet storage
HF_XET_HIGH_PERFORMANCE=1 huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir ./
```

---

### Method 2: Python Script

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    local_dir="./",
    local_dir_use_symlinks=False,
)
```

---

### Method 3: Git LFS

```bash
# Install Git LFS if not already installed
git lfs install

# Clone the model repository
git clone https://huggingface.co/Tongyi-MAI/Z-Image-Turbo ./
```

---

## Verify Installation

After downloading, this directory should contain:

```
z-image-turbo/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   ├── config.json
│   └── *.safetensors
├── tokenizer/
│   ├── tokenizer_config.json
│   └── ...
├── transformer/
│   ├── config.json
│   └── *.safetensors (multiple shards)
└── vae/
    ├── config.json
    └── *.safetensors
```

## Model Information

| Property        | Value                      |
| --------------- | -------------------------- |
| Model ID        | `Tongyi-MAI/Z-Image-Turbo` |
| Parameters      | 6B                         |
| Inference Steps | 8-9 (optimal)              |
| Guidance Scale  | 0.0 (CFG-free distilled)   |
| VRAM Usage      | ~14GB (bfloat16)           |
| License         | Apache 2.0                 |

## Links

- 🤗 [HuggingFace Model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- 📄 [Technical Report](https://arxiv.org/abs/2511.22699)
- 🌐 [Official GitHub](https://github.com/Tongyi-MAI/Z-Image)
