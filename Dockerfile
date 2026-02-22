FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA 13.0 support (required for GB10 sm_121)
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Install diffusers from main (LTX2Pipeline not yet in stable release)
RUN pip install --no-cache-dir --break-system-packages \
    git+https://github.com/huggingface/diffusers.git

# Install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/outputs /app/models

# Force system cuBLAS over PyTorch's bundled version (required for GB10 sm_121)
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13:/usr/local/cuda/lib64/libcublasLt.so.13

EXPOSE 7860

CMD ["python", "-m", "app.main"]
