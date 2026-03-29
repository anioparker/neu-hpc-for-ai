#!/usr/bin/env bash
set -euo pipefail

export THUNDERKITTENS_ROOT="${THUNDERKITTENS_ROOT:-/root/thunderkittens}"
export CPLUS_INCLUDE_PATH="${THUNDERKITTENS_ROOT}:${CPLUS_INCLUDE_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

python -m pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .