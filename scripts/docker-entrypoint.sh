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
FP8_CKPT="$FP8_DIR/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"
ENCODER_CHECK="$FP8_DIR/text_encoder/model-00001-of-00004.safetensors"

# Check if FP8 checkpoint AND text_encoder exist
if [ -f "$FP8_CKPT" ] && [ -f "$ENCODER_CHECK" ]; then
    echo -e "${GREEN}[Startup] FP8 model found - using optimized inference (~19GB VRAM)${NC}"
else
    echo -e "${YELLOW}[Startup] FP8 model not found - downloading from HuggingFace...${NC}"
    echo -e "  This is a one-time download (~27GB, ~10-15 minutes)"
    
    mkdir -p "$FP8_DIR"
    
    # Use inline Python with huggingface_hub (same as model_downloader.py)
    python3 << 'EOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

fp8_dir = "/app/models/qwen-image-edit-2511-fp8"
temp_dir = "/app/models/temp-components"

# Download the single-file 8-step FP8 checkpoint (~20.5GB)
print("[Startup] Downloading FP8 checkpoint...")
hf_hub_download(
    repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
    filename="qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors",
    local_dir=fp8_dir,
)

# Download text_encoder, vae, scheduler, tokenizer from original Qwen model (~7GB)
print("[Startup] Downloading text_encoder, vae, and configs...")
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2511",
    allow_patterns=["text_encoder/*", "vae/*", "scheduler/*", "tokenizer/*", "*.json", "*.txt"],
    ignore_patterns=["transformer/*"],
    local_dir=temp_dir,
)

# Copy components to FP8 directory
import shutil
for item in Path(temp_dir).iterdir():
    dest = Path(fp8_dir) / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

# Cleanup temp directory
shutil.rmtree(temp_dir)
print("[Startup] FP8 model download complete!")
EOF
    
    echo -e "${GREEN}[Startup] FP8 model + all components downloaded successfully!${NC}"
fi

# =============================================================================
# Auto-download ACE-Step 1.5 Model if Missing
# =============================================================================
# ACE-Step 1.5 auto-downloads models on first initialize_service() call.
# The handler checks checkpoints/ dir and downloads from HuggingFace as needed.
# We pre-download here to avoid delays on first API request.
ACESTEP_CKPT_DIR="/app/repos/ACE-Step-1.5/checkpoints"
ACESTEP_CHECK="$ACESTEP_CKPT_DIR/acestep-v15-turbo/config.json"

if [ -f "$ACESTEP_CHECK" ]; then
    echo -e "${GREEN}[Startup] ACE-Step 1.5 model found${NC}"
else
    echo -e "${YELLOW}[Startup] ACE-Step 1.5 model not found - downloading from HuggingFace...${NC}"
    echo -e "  This is a one-time download (~3GB)"
    
    mkdir -p "$ACESTEP_CKPT_DIR"
    
    python3 << 'EOF'
from huggingface_hub import snapshot_download

print("[Startup] Downloading ACE-Step 1.5 model (VAE + text encoder + turbo DiT + LM)...")
snapshot_download(
    repo_id="ACE-Step/Ace-Step1.5",
    local_dir="/app/repos/ACE-Step-1.5/checkpoints",
)
print("[Startup] ACE-Step 1.5 model download complete!")
EOF
    
    echo -e "${GREEN}[Startup] ACE-Step 1.5 models downloaded successfully!${NC}"
fi

# =============================================================================
# Start the application
# =============================================================================
echo -e "${GREEN}[Startup] Starting Vid-Bolt GPU API...${NC}"
exec "$@"
