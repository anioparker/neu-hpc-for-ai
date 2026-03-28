#!/usr/bin/env bash
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="8.0"
# If you want PTX fallback too, use:
# export TORCH_CUDA_ARCH_LIST="8.0+PTX"

echo "=== compiler ==="
which gcc || true
which g++ || true
gcc --version || true
g++ --version || true

echo "=== cuda ==="
which nvcc || true
nvcc --version || true

echo "=== torch ==="
python - <<'PY'
import os
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("TORCH_CUDA_ARCH_LIST =", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
