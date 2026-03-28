import modal

APP_NAME = "deepseek-moe-multigpu-modal"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "gcc",
        "g++",
        "ninja-build",
    )
    .env({
        "CC": "gcc",
        "CXX": "g++",
        "CUDA_HOME": "/usr/local/cuda",
    })
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "numpy>=1.24",
        "modal>=1.0.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir("deepseek_moe", remote_path="/root/deepseek_moe", copy=True)
    .add_local_dir("csrc", remote_path="/root/csrc", copy=True)
    .add_local_dir("tests", remote_path="/root/tests", copy=True)
    .add_local_dir("scripts", remote_path="/root/scripts", copy=True)
    .add_local_file("setup.py", remote_path="/root/setup.py", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt", copy=True)
    .add_local_file("modal_app.py", remote_path="/root/modal_app.py", copy=True)
    .run_commands("cd /root && bash /root/scripts/build_extension.sh")
)


@app.function(image=image, gpu="A100:4", timeout=60 * 60)
def run_operator_test():
    import os
    import subprocess

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "/root/tests/test_deepseek_moe.py",
        ],
        check=True,
        env=env,
    )


@app.function(image=image, gpu="A100:4", timeout=60 * 60)
def run_transformer_test():
    import subprocess
    subprocess.run(["python", "/root/tests/test_transformer.py"], check=True)


@app.function(image=image, gpu="A100:4", timeout=60 * 60)
def run_benchmark():
    import os
    import subprocess

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "deepseek_moe.benchmark",
        ],
        check=True,
        env=env,
    )


@app.function(image=image, gpu="A100:4", timeout=60 * 60)
def run_train():
    import os
    import subprocess

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "deepseek_moe.train",
        ],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main(mode: str = "benchmark"):
    if mode == "operator_test":
        run_operator_test.remote()
    elif mode == "transformer_test":
        run_transformer_test.remote()
    elif mode == "train":
        run_train.remote()
    else:
        run_benchmark.remote()