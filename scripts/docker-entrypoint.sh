#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - Docker Entrypoint
# =============================================================================
# This script runs on container startup and:
# 1. Downloads FP8 model if missing (auto-download)
# 2. Starts the uvicorn server
# =============================================================================

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# Auto-download FP8 Model if Missing
# =============================================================================
FP8_DIR="/app/models/qwen-image-edit-2511-fp8"
ENCODER_CHECK="$FP8_DIR/text_encoder/model-00001-of-00004.safetensors"

# Check if FP8 model AND text_encoder exist
if [ -f "$ENCODER_CHECK" ]; then
    echo -e "${GREEN}[Startup] FP8 model found - using optimized inference (~19GB VRAM)${NC}"
else
    echo -e "${YELLOW}[Startup] FP8 model not found - downloading from HuggingFace...${NC}"
    echo -e "  This is a one-time download (~27GB, ~10-15 minutes)"
    
    # Install huggingface-cli if not available
    if ! command -v huggingface-cli &> /dev/null; then
        pip install -q huggingface-hub
    fi
    
    # Download pre-converted FP8 model with Lightning LoRA baked in
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
        --include "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_split/*" \
        --local-dir "/app/models/temp-fp8-download" \
        --local-dir-use-symlinks False
    
    # Move contents to correct location (flatten directory structure)
    mkdir -p "$FP8_DIR"
    mv /app/models/temp-fp8-download/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_split/* "$FP8_DIR/"
    rm -rf "/app/models/temp-fp8-download"
    
    # Download text_encoder, vae, scheduler, tokenizer from original Qwen model
    echo -e "${YELLOW}[Startup] Downloading text_encoder, vae, and configs (~7GB)...${NC}"
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
        --include "text_encoder/*" "vae/*" "scheduler/*" "tokenizer/*" "*.json" "*.txt" \
        --exclude "transformer/*" \
        --local-dir "/app/models/temp-components" \
        --local-dir-use-symlinks False
    
    # Copy components to FP8 directory
    cp -r /app/models/temp-components/* "$FP8_DIR/"
    rm -rf "/app/models/temp-components"
    
    echo -e "${GREEN}[Startup] FP8 model + all components downloaded successfully!${NC}"
fi

# =============================================================================
# Start the application
# =============================================================================
echo -e "${GREEN}[Startup] Starting Vid-Bolt GPU API...${NC}"
exec "$@"
