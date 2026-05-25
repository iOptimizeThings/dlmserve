# Targets SM 8.0–8.9 (A100, H100, RTX 3090/4090). Blackwell (SM 12.0) requires
# PyTorch nightly — not supported in this image. See docs/docker.md.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY dlmserve/ dlmserve/

RUN uv pip install --system .

VOLUME /root/.cache/huggingface

EXPOSE 8000

ENV DLMSERVE_MODEL=gsai-ml/LLaDA-8B-Instruct
ENV DLMSERVE_DTYPE=int4
ENV DLMSERVE_LOG_LEVEL=info

CMD ["dlmserve"]
