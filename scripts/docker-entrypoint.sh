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
CONFIG_CHECK="$FP8_DIR/scheduler/scheduler_config.json"

# Check if FP8 model AND config files exist
if [ -f "$CONFIG_CHECK" ]; then
    echo -e "${GREEN}[Startup] FP8 model found - using optimized inference (~19GB VRAM)${NC}"
else
    echo -e "${YELLOW}[Startup] FP8 model not found - downloading from HuggingFace...${NC}"
    echo -e "  This is a one-time download (~20.5GB, ~5-10 minutes)"
    
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
    
    # Download config files from original Qwen model (scheduler, tokenizer, etc.)
    echo -e "${YELLOW}[Startup] Downloading config files from original Qwen model...${NC}"
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
        --include "scheduler/*" "tokenizer/*" "*.json" "*.txt" \
        --exclude "*.safetensors" "*.bin" "*.ckpt" "*.pth" "*.pt" \
        --local-dir "/app/models/temp-config-download" \
        --local-dir-use-symlinks False
    
    # Copy config files to FP8 directory
    cp -rn /app/models/temp-config-download/* "$FP8_DIR/" 2>/dev/null || true
    rm -rf "/app/models/temp-config-download"
    
    echo -e "${GREEN}[Startup] FP8 model + config files downloaded successfully!${NC}"
fi

# =============================================================================
# Start the application
# =============================================================================
echo -e "${GREEN}[Startup] Starting Vid-Bolt GPU API...${NC}"
exec "$@"
