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
    pip install -q huggingface-hub
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
echo -e "  ├── qwen-image-edit-2511/"
echo -e "  ├── loras/"
echo -e "  │   ├── z-image/"
echo -e "  │   └── qwen-image-edit-2511/"
echo -e "  └── ltx-2/"
echo -e "      ├── ltx-2-19b-distilled-fp8.safetensors"
echo -e "      ├── ltx-2-spatial-upscaler-x2-1.0.safetensors"
echo -e "      └── gemma-3-12b-it-qat-q4_0-unquantized/"
echo ""
du -sh "$MODELS_DIR" 2>/dev/null || echo "Total size: ~80GB"
