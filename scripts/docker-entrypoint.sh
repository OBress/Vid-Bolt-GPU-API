#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - Docker Entrypoint
# =============================================================================
# This script runs on container startup and starts the uvicorn server.
# FP8 models are downloaded by download-models.sh (no conversion needed).
# =============================================================================

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# Check FP8 Model Status
# =============================================================================
FP8_DIR="/app/models/qwen-image-edit-2511-fp8"

if [ -d "$FP8_DIR" ] && [ -n "$(ls -A $FP8_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}[Startup] FP8 model found - using optimized inference (~19GB VRAM)${NC}"
else
    echo -e "${YELLOW}[Startup] FP8 model not found - using BF16 (~38GB VRAM)${NC}"
    echo -e "  Run ./scripts/download-models.sh to download FP8 model for 2x concurrency"
fi

# =============================================================================
# Start the application
# =============================================================================
echo -e "${GREEN}[Startup] Starting Vid-Bolt GPU API...${NC}"
exec "$@"
