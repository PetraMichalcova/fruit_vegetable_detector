FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

# OpenCV and PyTorch need these shared libraries on the slim base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY config.py .
COPY videos ./videos

# The Hugging Face model is downloaded by main.py on first start and cached in
# the named volume configured in compose.yaml.
RUN mkdir -p /app/output /root/.cache/huggingface

ENTRYPOINT ["python", "-u", "main.py"]
