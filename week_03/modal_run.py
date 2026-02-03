import pathlib
import subprocess
import modal

LOCAL_DIR = pathlib.Path(__file__).parent.resolve()
REMOTE_DIR = "/root/work"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "cmake", "ninja-build", "git")
    .add_local_dir(local_path=str(LOCAL_DIR), remote_path=REMOTE_DIR)
)

app = modal.App("cuda-mmm-h100", image=image)

@app.function(gpu="H100", timeout=60 * 20)
def run():
    build_dir = f"{REMOTE_DIR}/build"

    # Always work from repo root so relative paths are stable
    subprocess.run(["bash", "-lc", f"cd {REMOTE_DIR} && rm -rf build && mkdir -p build"], check=True)

    # Configure + build
    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_DIR} && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    subprocess.run(["bash", "-lc", f"cd {REMOTE_DIR} && cmake --build build -j"], check=True)

    # Calculations (blog uses 4092)
    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_DIR} && python scripts/calc_bounds.py --m 4092 --n 4092 --k 4092"],
        check=True,
    )

    # Sweeps
    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_DIR} && python scripts/sweep.py --bin ./build/sgemm --m 4092 --n 4092 --k 4092 --reps 50"],
        check=True,
    )
    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_DIR} && python scripts/sweep.py --bin ./build/sgemm --m 4096 --n 4096 --k 4096 --reps 50"],
        check=True,
    )

    # Show outputs
    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_DIR} && ls -lah results/raw && echo '--- head h100_4092.csv ---' && head -n 50 results/raw/h100_4092.csv"],
        check=True,
    )

if __name__ == "__main__":
    run.remote()
