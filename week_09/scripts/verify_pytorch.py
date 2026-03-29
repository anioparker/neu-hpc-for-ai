#!/usr/bin/env python3
import subprocess
import torch

print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())

print('\n--- nvidia-smi ---')
try:
    subprocess.run(["nvidia-smi"], check=True)
except Exception as e:
    print('nvidia-smi failed:', e)
