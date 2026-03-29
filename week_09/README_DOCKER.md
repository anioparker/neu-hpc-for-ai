GPU Docker setup for this project

Steps to build and run a CUDA 12.4 container that mounts this repository:

1) Build image

```bash
docker build -t deepseek_moe_tk:cuda12.4 .
```

2) Run a one-off container that verifies PyTorch and GPU access

```bash
docker run --gpus all --rm -it -v "$PWD":/workspace deepseek_moe_tk:cuda12.4 bash -c "python3 scripts/verify_pytorch.py"
```

3) Alternatively use docker-compose:

```bash
docker compose up --build
```

Notes
- This uses the official NVIDIA CUDA base image for CUDA 12.4 + cuDNN on Ubuntu 22.04.
- Ensure your host has the NVIDIA Container Toolkit installed (`nvidia-docker2` / `--gpus` support).
- If PyTorch wheels for `cu124` are unavailable for a given torch version, pip will fall back to the default index; you may need to choose a compatible torch wheel or build from source.
