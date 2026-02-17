#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"

echo "=== Downloading Models ==="
echo "Target directory: $MODELS_DIR"

# Check huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub[cli]
fi

# LTX-2 (LTX-Video 2.0 distilled FP8)
echo ""
echo "--- Downloading LTX-Video 2.0 (FP8 distilled, ~27GB) ---"
huggingface-cli download Lightricks/LTX-Video-2.0-distilled \
    --local-dir "$MODELS_DIR/ltx2" \
    --include "*.safetensors" "*.json" "*.txt" "*.model"

# SVD-XT
echo ""
echo "--- Downloading SVD-XT (~10GB) ---"
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt \
    --local-dir "$MODELS_DIR/svd" \
    --include "*.safetensors" "*.json" "*.txt"

echo ""
echo "=== Download complete ==="
echo "LTX-2: $MODELS_DIR/ltx2"
echo "SVD-XT: $MODELS_DIR/svd"
du -sh "$MODELS_DIR/ltx2" "$MODELS_DIR/svd" 2>/dev/null || true
