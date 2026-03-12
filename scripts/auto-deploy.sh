#!/bin/bash
# =============================================================================
# Vid-Bolt GPU API - Boot Deploy Script
# =============================================================================
# Runs automatically on VM startup via systemd. Checks for code updates,
# rebuilds if needed, and starts the application.
#
# Flow: VM boots → this script runs → checks git → rebuilds if needed → starts app
# =============================================================================

set -euo pipefail

REPO_DIR="/home/ubuntu/Vid-Bolt-GPU-API"
LOG_FILE="/var/log/vidbolt-deploy.log"
COMPOSE_CMD="docker compose"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "VM boot — starting deploy check"
log "=========================================="

cd "$REPO_DIR"

# ---- Check for updates ----
log "Fetching latest from GitHub..."
git fetch origin main --quiet 2>&1 | tee -a "$LOG_FILE" || {
    log "WARNING: git fetch failed (network issue?). Starting with current version."
    $COMPOSE_CMD up -d 2>&1 | tee -a "$LOG_FILE"
    exit 0
}

LOCAL_HASH=$(git rev-parse HEAD)
REMOTE_HASH=$(git rev-parse origin/main)

if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
    log "Already up to date ($LOCAL_HASH). Starting application..."
    $COMPOSE_CMD up -d 2>&1 | tee -a "$LOG_FILE"
    log "Application started."
    exit 0
fi

# ---- New commits found — update and rebuild ----
log "UPDATE FOUND: $LOCAL_HASH -> $REMOTE_HASH"

CHANGED_SUMMARY=$(git log --oneline "$LOCAL_HASH..$REMOTE_HASH" 2>/dev/null || echo "(unable to get log)")
log "New commits:"
echo "$CHANGED_SUMMARY" | while IFS= read -r line; do log "  $line"; done

log "Pulling latest code..."
git pull origin main --ff-only 2>&1 | tee -a "$LOG_FILE"

# Check what changed to decide rebuild strategy
CHANGED_FILES=$(git diff --name-only "$LOCAL_HASH" "$REMOTE_HASH" 2>/dev/null || echo "")

if echo "$CHANGED_FILES" | grep -qE '^(Dockerfile|docker-compose\.yml|repos/|requirements\.txt|scripts/docker-entrypoint\.sh)'; then
    log "Infrastructure changed — full rebuild required."

    log "Stopping containers..."
    $COMPOSE_CMD down 2>&1 | tee -a "$LOG_FILE" || true

    log "Pruning old Docker images..."
    docker system prune -af 2>&1 | tee -a "$LOG_FILE"

    log "Building new image (--no-cache)..."
    $COMPOSE_CMD build --no-cache 2>&1 | tee -a "$LOG_FILE"

    log "Starting new container..."
    $COMPOSE_CMD up -d 2>&1 | tee -a "$LOG_FILE"
else
    log "Only app/ code changed — quick restart."
    $COMPOSE_CMD restart 2>&1 | tee -a "$LOG_FILE"
fi

log "Deploy complete! Running version: $(git rev-parse --short HEAD)"
log "=========================================="
