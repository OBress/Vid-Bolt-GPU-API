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
echo -e "${YELLOW}[1/4] Downloading Z-Image Turbo...${NC}"
huggingface-cli download Tongyi-MAI/Z-Image-Turbo \
    --local-dir "$MODELS_DIR/z-image-turbo" \
    --local-dir-use-symlinks False
echo -e "${GREEN}  ✓ Z-Image Turbo downloaded${NC}"

# Note: We skip Qwen-Image-Edit-2511 BF16 (~14GB) and separate LoRA (~500MB)
# because the FP8 model below has Lightning LoRA baked in and uses less VRAM

# =============================================================================
# Download Pre-converted FP8 Model (reduces VRAM from ~38GB to ~19GB)
# This model has the Lightning 8-step LoRA already baked in!
# =============================================================================
FP8_DIR="$MODELS_DIR/qwen-image-edit-2511-fp8"
if [ ! -d "$FP8_DIR" ] || [ -z "$(ls -A $FP8_DIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}[2/4] Downloading pre-converted FP8 model...${NC}"
    echo -e "  This enables 2x concurrent instances with same VRAM"
    
    # Download the pre-converted FP8 model with 8-step Lightning LoRA baked in
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
        --include "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_split/*" \
        --local-dir "$MODELS_DIR/temp-fp8-download" \
        --local-dir-use-symlinks False
    
    # Move to correct location
    mv "$MODELS_DIR/temp-fp8-download/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_split" "$FP8_DIR"
    rm -rf "$MODELS_DIR/temp-fp8-download"
    
    echo -e "${GREEN}  ✓ FP8 model downloaded (~20.5GB)${NC}"
else
    echo -e "${GREEN}  ✓ FP8 model already exists, skipping download${NC}"
fi

# =============================================================================
# LTX-2 Components (~40GB total)
# =============================================================================
echo -e "${YELLOW}[3/4] Downloading LTX-2 components...${NC}"

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
echo -e "${YELLOW}[4/4] Downloading Gemma-3-12B text encoder...${NC}"
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
echo -e "  ├── z-image-turbo/                 (~12GB)"
echo -e "  ├── qwen-image-edit-2511-fp8/      (~20GB, has Lightning LoRA baked in)"
echo -e "  └── ltx-2/"
echo -e "      ├── ltx-2-19b-distilled-fp8.safetensors"
echo -e "      ├── ltx-2-spatial-upscaler-x2-1.0.safetensors"
echo -e "      ├── ltx-2-19b-distilled-lora-384.safetensors"
echo -e "      └── gemma-3-12b-it-qat-q4_0-unquantized/"
echo ""
du -sh "$MODELS_DIR" 2>/dev/null || echo "Total size: ~70GB"
