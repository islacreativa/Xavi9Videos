#!/usr/bin/env bash
set -euo pipefail

echo "=== Xavi9Videos Setup ==="

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers."
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found."
    exit 1
fi

# Check NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit may not be installed."
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check .env
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env from .env.example - edit it with your NGC_API_KEY"
    else
        echo "ERROR: .env.example not found."
        exit 1
    fi
fi

# Verify NGC_API_KEY
source .env
if [ -z "${NGC_API_KEY:-}" ] || [ "$NGC_API_KEY" = "your_ngc_api_key_here" ]; then
    echo "WARNING: Set your NGC_API_KEY in .env"
fi

# NGC Docker login
echo ""
echo "Logging into NGC container registry..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo ""
echo "=== Setup complete ==="
echo "Next: ./scripts/download_models.sh (optional, models download on first run)"
echo "Then: docker compose up --build"
