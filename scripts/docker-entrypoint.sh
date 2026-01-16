#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - Docker Entrypoint
# =============================================================================
# This script runs on container startup and:
# 1. Attempts FP8 conversion if model doesn't exist (optional, skips on failure)
# 2. Starts the uvicorn server
# =============================================================================

# Don't exit on error - we want to continue even if FP8 conversion fails
set +e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# =============================================================================
# FP8 Model Check and Conversion (Optional)
# =============================================================================
FP8_DIR="/app/models/qwen-image-edit-2511-fp8"
SOURCE_DIR="/app/models/qwen-image-edit-2511"

if [ -d "$SOURCE_DIR" ]; then
    if [ ! -d "$FP8_DIR" ] || [ -z "$(ls -A $FP8_DIR 2>/dev/null)" ]; then
        echo -e "${YELLOW}[Startup] FP8 model not found. Attempting conversion...${NC}"
        echo -e "  Note: FP8 is optional - BF16 will be used if conversion fails"
        
        mkdir -p "$FP8_DIR"
        
        # Try to run conversion, but don't fail if it doesn't work
        if python /app/LightX2V/tools/convert/converter.py \
            --source "$SOURCE_DIR" \
            --output "$FP8_DIR" \
            --output_ext .safetensors \
            --output_name qwen_image_dit_fp8 \
            --linear_type fp8 \
            --non_linear_dtype torch.bfloat16 \
            --model_type qwen_image_dit \
            --quantized \
            --save_by_block \
            --copy_no_weight_files 2>/dev/null; then
            echo -e "${GREEN}[Startup] FP8 conversion complete!${NC}"
        else
            echo -e "${RED}[Startup] FP8 conversion failed (missing dependencies). Using BF16 model instead.${NC}"
            echo -e "  This is fine - BF16 uses more VRAM but works correctly."
            # Clean up empty FP8 directory
            rm -rf "$FP8_DIR"
        fi
    else
        echo -e "${GREEN}[Startup] FP8 model found, skipping conversion${NC}"
    fi
else
    echo -e "${YELLOW}[Startup] Source model not found at $SOURCE_DIR, skipping FP8 check${NC}"
fi

# =============================================================================
# Start the application
# =============================================================================
echo -e "${GREEN}[Startup] Starting Vid-Bolt GPU API...${NC}"
exec "$@"
