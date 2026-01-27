import subprocess
import pathlib
import modal

LOCAL_DIR = pathlib.Path(__file__).parent.resolve()
REMOTE_DIR = "/root/work"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    # NEW (Modal 1.x): include local source files into the image
    .add_local_dir(local_path=str(LOCAL_DIR), remote_path=REMOTE_DIR)
)

app = modal.App("kernel-10-warptiling-h100", image=image)

@app.function(gpu="H100", timeout=60 * 20)
def run(m: int = 4096, n: int = 4096, k: int = 4096, reps: int = 50):
    cu = f"{REMOTE_DIR}/kernel_10.cu"
    out = "/tmp/warptiling"

    # Compile for H100 (SM90)
    # "-DBM=128", "-DBN=128", "-DBK=16", "-DTM=8", "-DTN=8",
    subprocess.run(
        [
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-lineinfo",
            cu, "-o", out
        ],
        check=True,
    )

    # Most of these kernels accept: M N K reps
    subprocess.run([out, str(m), str(n), str(k), str(reps)], check=True)

@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, reps: int = 50):
    run.remote(m, n, k, reps)