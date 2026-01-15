#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - Model Download Script
# =============================================================================
# Downloads all required models from HuggingFace to ./models directory
# 
# Total size: ~80GB
#   - Z-Image Turbo: ~12GB
#   - Qwen-Image-Edit-2511: ~14GB + LoRA
#   - LTX-2: ~40GB (checkpoint + LoRA + upsampler + Gemma)
#   - Stream-DiffVSR: Auto-downloaded at runtime
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Vid-Bolt - Model Download Script                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create directory structure
mkdir -p "$MODELS_DIR/z-image-turbo"
mkdir -p "$MODELS_DIR/qwen-image-edit-2511"
mkdir -p "$MODELS_DIR/loras/z-image"
mkdir -p "$MODELS_DIR/loras/qwen-image-edit-2511"
mkdir -p "$MODELS_DIR/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized"

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface-cli...${NC}"
    python3 -m pip install -q huggingface-hub || pip install -q huggingface-hub
fi

# =============================================================================
# Z-Image Turbo (~12GB)
# =============================================================================
echo -e "${YELLOW}[1/5] Downloading Z-Image Turbo...${NC}"
huggingface-cli download Tongyi-MAI/Z-Image-Turbo \
    --local-dir "$MODELS_DIR/z-image-turbo" \
    --local-dir-use-symlinks False
echo -e "${GREEN}  ✓ Z-Image Turbo downloaded${NC}"

# =============================================================================
# Qwen-Image-Edit-2511 + Lightning LoRA (~14GB + ~500MB)
# =============================================================================
echo -e "${YELLOW}[2/5] Downloading Qwen-Image-Edit-2511...${NC}"
huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
    --local-dir "$MODELS_DIR/qwen-image-edit-2511" \
    --local-dir-use-symlinks False
echo -e "${GREEN}  ✓ Qwen-Image-Edit-2511 downloaded${NC}"

echo -e "${YELLOW}[3/5] Downloading LightX2V LoRA (8-step distilled)...${NC}"
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
    Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors \
    --local-dir "$MODELS_DIR/loras/qwen-image-edit-2511" \
    --local-dir-use-symlinks False
echo -e "${GREEN}  ✓ LightX2V LoRA downloaded${NC}"

# =============================================================================
# Convert Qwen-Image-Edit to FP8 (reduces VRAM from ~38GB to ~19GB)
# =============================================================================
FP8_DIR="$MODELS_DIR/qwen-image-edit-2511-fp8"
if [ ! -d "$FP8_DIR" ] || [ -z "$(ls -A $FP8_DIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}[3.5/5] Converting Qwen-Image-Edit to FP8...${NC}"
    echo -e "  This enables 2x concurrent instances with same VRAM"
    
    mkdir -p "$FP8_DIR"
    
    # Run the converter from LightX2V tools
    python "$PROJECT_DIR/LightX2V/tools/convert/converter.py" \
        --source "$MODELS_DIR/qwen-image-edit-2511" \
        --output "$FP8_DIR" \
        --output_ext .safetensors \
        --output_name qwen_image_dit_fp8 \
        --linear_type fp8 \
        --non_linear_dtype torch.bfloat16 \
        --model_type qwen_image_dit \
        --quantized \
        --save_by_block \
        --copy_no_weight_files
    
    echo -e "${GREEN}  ✓ FP8 conversion complete${NC}"
else
    echo -e "${GREEN}  ✓ FP8 model already exists, skipping conversion${NC}"
fi

# =============================================================================
# LTX-2 Components (~40GB total)
# =============================================================================
echo -e "${YELLOW}[4/5] Downloading LTX-2 components...${NC}"

# Main checkpoint (Distilled FP8 for 8-step inference)
echo -e "  Downloading ltx-2-19b-distilled-fp8.safetensors..."
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled-fp8.safetensors \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

# Spatial upsampler
echo -e "  Downloading ltx-2-spatial-upscaler-x2-1.0.safetensors..."
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

# Distilled LoRA (required for KeyframeInterpolationPipeline)
echo -e "  Downloading ltx-2-19b-distilled-lora-384.safetensors..."
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled-lora-384.safetensors \
    --local-dir "$MODELS_DIR/ltx-2" \
    --local-dir-use-symlinks False

echo -e "${GREEN}  ✓ LTX-2 components downloaded${NC}"

# =============================================================================
# Gemma Text Encoder (~12GB)
# =============================================================================
echo -e "${YELLOW}[5/5] Downloading Gemma-3-12B text encoder...${NC}"
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$MODELS_DIR/ltx-2/gemma-3-12b-it-qat-q4_0-unquantized" \
    --local-dir-use-symlinks False
echo -e "${GREEN}  ✓ Gemma text encoder downloaded${NC}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  All Models Downloaded!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Models directory: ${BLUE}$MODELS_DIR${NC}"
echo ""
echo -e "Directory structure:"
echo -e "  models/"
echo -e "  ├── z-image-turbo/"
echo -e "  ├── qwen-image-edit-2511/         (BF16 base model)"
echo -e "  ├── qwen-image-edit-2511-fp8/     (FP8 quantized - 50% less VRAM)"
echo -e "  ├── loras/"
echo -e "  │   ├── z-image/"
echo -e "  │   └── qwen-image-edit-2511/"
echo -e "  └── ltx-2/"
echo -e "      ├── ltx-2-19b-distilled-fp8.safetensors"
echo -e "      ├── ltx-2-spatial-upscaler-x2-1.0.safetensors"
echo -e "      ├── ltx-2-19b-distilled-lora-384.safetensors"
echo -e "      └── gemma-3-12b-it-qat-q4_0-unquantized/"
echo ""
du -sh "$MODELS_DIR" 2>/dev/null || echo "Total size: ~85GB"
