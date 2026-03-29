import os
import modal

APP_NAME = "deepseek-moe-thunderkittens-b200"
CUDA_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "gcc",
        "g++",
        "ninja-build",
        "wget",
        "curl",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install numpy",
        "pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        "git clone --depth=1 https://github.com/HazyResearch/ThunderKittens.git /root/thunderkittens",
    )
    .env(
        {
            "THUNDERKITTENS_ROOT": "/root/thunderkittens",
            "TORCH_CUDA_ARCH_LIST": "10.0+PTX",
            "MAX_JOBS": "8",
            "CC": "gcc",
            "CXX": "g++",
        }
    )
    .add_local_dir(".", remote_path="/root/project", copy=True)
    .run_commands(
        "cd /root/project && "
        "CPLUS_INCLUDE_PATH=/root/thunderkittens:${CPLUS_INCLUDE_PATH} "
        "pip install -e . --no-build-isolation"
    )
)

volume = modal.Volume.from_name("deepseek-moe-tk-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
    volumes={"/cache": volume},
)
def run_demo():
    import torch
    from deepseek_moe_tk import DeepseekMoETwitterKittens

    torch.manual_seed(0)
    assert torch.cuda.is_available()

    device = "cuda"
    dtype = torch.bfloat16

    model = DeepseekMoETwitterKittens(
        hidden_size=4096,
        intermediate_size=11008,
        num_experts=16,
        top_k=2,
        dtype=dtype,
        device=device,
    ).eval()

    x = torch.randn(8, 256, 4096, device=device, dtype=dtype)

    # warmup
    for _ in range(5):
        y = model(x)
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    for _ in range(20):
        y = model(x)
    torch.cuda.synchronize()
    t1 = time.time()

    toks = x.shape[0] * x.shape[1] * 20
    print("output:", y.shape, y.dtype)
    print(f"elapsed={t1-t0:.4f}s")
    print(f"tokens/sec={toks/(t1-t0):.2f}")

    return {
        "shape": tuple(y.shape),
        "dtype": str(y.dtype),
        "elapsed_s": t1 - t0,
        "tokens_per_sec": toks / (t1 - t0),
    }


@app.local_entrypoint()
def main():
    out = run_demo.remote()
    print(out)
