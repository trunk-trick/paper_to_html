# CUDA 12.8 runtime: matches the cu128 torch wheel an RTX 50 card needs.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip3 install -r requirements.txt

COPY . .

# The base model is not baked in: mount the hub cache at /models.
#   docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/models pdf2reading
EXPOSE 8000
ENTRYPOINT ["python3", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
