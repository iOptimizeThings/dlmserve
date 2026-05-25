# Docker deployment

## Prerequisites

**NVIDIA drivers** (already required to use your GPU) and **nvidia-container-toolkit** (one-time install):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Run

```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  dlmserve
```

The `-v` flag reuses your local HuggingFace model cache so the model is not re-downloaded.

## Configuration

| Environment variable   | Default                        | Description                     |
|------------------------|--------------------------------|---------------------------------|
| `DLMSERVE_MODEL`       | `gsai-ml/LLaDA-8B-Instruct`   | HuggingFace model ID            |
| `DLMSERVE_DTYPE`       | `int4`                         | `int4` or `bf16`       |
| `DLMSERVE_LOG_LEVEL`   | `info`                         | `debug`, `info`, `warning`      |
| `DLMSERVE_DEVICE`      | `cuda`                         | `cuda` or `cpu`                 |

Example — run in bf16 on a different model:

```bash
docker run --gpus all -p 8000:8000 \
  -e DLMSERVE_MODEL=gsai-ml/LLaDA-8B-Instruct \
  -e DLMSERVE_DTYPE=bf16 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  dlmserve
```

## GPU compatibility

The image bundles **PyTorch 2.5.1 + CUDA 12.4** and targets:

| Architecture | GPUs                          |
|--------------|-------------------------------|
| SM 8.0       | A100                          |
| SM 9.0       | H100                          |
| SM 8.6       | RTX 3090, RTX 3080, A6000     |
| SM 8.9       | RTX 4090, RTX 4080, L40       |

**RTX 5000-series (Blackwell, SM 12.0) is not supported by this image.** PyTorch 2.5.1 does not include SM 12.0 kernels. Blackwell support will ship when a stable PyTorch release includes it. Until then, use the local venv install on Blackwell hardware.

## Build from source

```bash
docker build -t dlmserve .
```
