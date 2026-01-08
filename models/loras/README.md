# LoRA Adapters

This directory stores LoRA (Low-Rank Adaptation) weights for fine-tuned model variations.

## Directory Structure

```
loras/
└── z-image/           # Z-Image compatible LoRAs
    └── <lora_name>.safetensors
```

## Usage

1. Place LoRA `.safetensors` files in the appropriate subdirectory
2. Reference by filename (without extension) in API requests:

```json
{
  "prompt": "A beautiful sunset",
  "lora_name": "your_lora_name",
  "lora_weight": 1.0
}
```

## Finding LoRAs

Z-Image Turbo compatible LoRAs can be found on:

- [HuggingFace - Z-Image Adapters](https://huggingface.co/models?other=base_model:adapter:Tongyi-MAI/Z-Image-Turbo)
- [Civitai](https://civitai.com/) (filter by Z-Image/ZImage)

## Training LoRAs

For training custom LoRAs, refer to:

- [DiffSynth-Studio Z-Image LoRA Training](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md)
