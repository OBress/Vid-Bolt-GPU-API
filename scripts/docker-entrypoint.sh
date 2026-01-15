#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - Docker Entrypoint
# =============================================================================
# This script runs on container startup and:
# 1. Checks if FP8 quantized model exists, converts if missing
# 2. Starts the uvicorn server
# =============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# FP8 Model Check and Conversion
# =============================================================================
FP8_DIR="/app/models/qwen-image-edit-2511-fp8"
SOURCE_DIR="/app/models/qwen-image-edit-2511"

if [ -d "$SOURCE_DIR" ]; then
    if [ ! -d "$FP8_DIR" ] || [ -z "$(ls -A $FP8_DIR 2>/dev/null)" ]; then
        echo -e "${YELLOW}[Startup] FP8 model not found, converting...${NC}"
        echo -e "  This is a one-time operation (~5-10 minutes)"
        
        mkdir -p "$FP8_DIR"
        
        python /app/LightX2V/tools/convert/converter.py \
            --source "$SOURCE_DIR" \
            --output "$FP8_DIR" \
            --output_ext .safetensors \
            --output_name qwen_image_dit_fp8 \
            --linear_type fp8 \
            --non_linear_dtype torch.bfloat16 \
            --model_type qwen_image_dit \
            --quantized \
            --save_by_block \
            --copy_no_weight_files
        
        echo -e "${GREEN}[Startup] FP8 conversion complete!${NC}"
    else
        echo -e "${GREEN}[Startup] FP8 model found, skipping conversion${NC}"
    fi
else
    echo -e "${YELLOW}[Startup] Source model not found at $SOURCE_DIR, skipping FP8 conversion${NC}"
fi

# =============================================================================
# Start the application
# =============================================================================
echo -e "${GREEN}[Startup] Starting Vid-Bolt GPU API...${NC}"
exec "$@"
