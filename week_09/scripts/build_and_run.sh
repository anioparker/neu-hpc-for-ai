#!/usr/bin/env bash
set -euo pipefail

# Build image
docker build -t deepseek_moe_tk:cuda12.4 .

# Run verification using GPU
docker run --gpus all --rm -it -v "$PWD":/workspace deepseek_moe_tk:cuda12.4 bash -c "python3 scripts/verify_pytorch.py"
