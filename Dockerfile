FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /app/outputs /app/models

EXPOSE 7860

CMD ["python", "-m", "app.main"]
