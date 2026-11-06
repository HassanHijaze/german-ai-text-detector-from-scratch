FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the model into the image at build time, so the container runs offline
ARG MODEL_ID=HassanHijazi/german-ai-text-mmbert-base
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_ID}', local_dir='/app/model', ignore_patterns=['training_args.bin', '*.md', '.gitattributes'])"

COPY app.py .

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

ENV MODEL_DIR=/app/model \
    HF_HUB_OFFLINE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

EXPOSE 7860
CMD ["python", "app.py"]
