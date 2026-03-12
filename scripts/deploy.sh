#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - One-Step Deployment Script
# =============================================================================
# Usage:
#   ./scripts/deploy.sh              # Full deploy (build + models + run)
#   ./scripts/deploy.sh --no-models  # Skip model download
#   ./scripts/deploy.sh --build-only # Only build image
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Vid-Bolt GPU API - Deployment Script               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Parse Arguments
# =============================================================================
SKIP_MODELS=false
BUILD_ONLY=false

for arg in "$@"; do
    case $arg in
        --no-models)
            SKIP_MODELS=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-models    Skip model download"
            echo "  --build-only   Only build the Docker image"
            echo "  --help         Show this help message"
            exit 0
            ;;
    esac
done

# =============================================================================
# Prerequisites Check
# =============================================================================
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi
echo -e "  ✓ Docker installed"

# Check NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}  ⚠ NVIDIA Container Toolkit may not be installed${NC}"
    echo -e "    Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  ✓ GPU detected: ${GREEN}$GPU_NAME${NC}"
else
    echo -e "${YELLOW}  ⚠ nvidia-smi not found (GPU may still work in container)${NC}"
fi

# =============================================================================
# Environment Setup
# =============================================================================
echo -e "${YELLOW}[2/5] Setting up environment...${NC}"

# Create .env from example if not exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "  ✓ Created .env from .env.example"
        echo -e "${YELLOW}    ⚠ Please edit .env with your API keys${NC}"
    fi
fi

# Create models directory
mkdir -p models
echo -e "  ✓ Models directory ready"

# =============================================================================
# Build Docker Image
# =============================================================================
echo -e "${YELLOW}[3/5] Building Docker image...${NC}"
echo -e "  This may take 10-15 minutes on first build..."

docker compose build

echo -e "  ${GREEN}✓ Docker image built successfully${NC}"

if [ "$BUILD_ONLY" = true ]; then
    echo -e "${GREEN}Build complete! Run 'docker compose up -d' to start.${NC}"
    exit 0
fi

# =============================================================================
# Install Auto-Deploy Service (runs on every VM boot)
# =============================================================================
echo -e "${YELLOW}[3.5/5] Installing auto-deploy service...${NC}"

if [ -f "$SCRIPT_DIR/auto-deploy.sh" ] && [ -f "$SCRIPT_DIR/vidbolt-deploy.service" ]; then
    sudo cp "$SCRIPT_DIR/auto-deploy.sh" /usr/local/bin/vidbolt-auto-deploy
    sudo chmod +x /usr/local/bin/vidbolt-auto-deploy
    sudo cp "$SCRIPT_DIR/vidbolt-deploy.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable vidbolt-deploy.service 2>/dev/null
    echo -e "  ${GREEN}✓ Auto-deploy enabled (checks for updates on every VM boot)${NC}"
else
    echo -e "  ${YELLOW}⚠ Auto-deploy scripts not found, skipping${NC}"
fi

# =============================================================================
# Download Models
# =============================================================================
if [ "$SKIP_MODELS" = false ]; then
    echo -e "${YELLOW}[4/5] Downloading models from HuggingFace...${NC}"
    echo -e "  This may take 30-60 minutes depending on connection speed..."
    
    # Run model download script
    if [ -f "$SCRIPT_DIR/download-models.sh" ]; then
        bash "$SCRIPT_DIR/download-models.sh"
    else
        echo -e "${YELLOW}  ⚠ download-models.sh not found, skipping model download${NC}"
    fi
else
    echo -e "${YELLOW}[4/5] Skipping model download (--no-models)${NC}"
fi

# =============================================================================
# Start Services
# =============================================================================
echo -e "${YELLOW}[5/5] Starting services...${NC}"

docker compose up -d

# Wait for health check
echo -e "  Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Final status
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Deployment Complete!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  API URL:     ${BLUE}http://localhost:8000${NC}"
echo -e "  Health:      ${BLUE}http://localhost:8000/health${NC}"
echo -e "  Docs:        ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "  Logs:        docker compose logs -f"
echo -e "  Stop:        docker compose down"
echo ""
