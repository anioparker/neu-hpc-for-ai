# DeepSeek MoE + ThunderKittens on Modal B200

This project builds a custom CUDA extension for a DeepSeek-style MoE FFN:
- Router is computed in PyTorch
- Top-k dispatch is passed to a CUDA kernel
- CUDA kernel uses ThunderKittens warp-level MMA
- Blackwell-targeted build flags for B200
- Modal deployment with `gpu="B200"`

## Run on Modal

```bash
modal run modal_app.py