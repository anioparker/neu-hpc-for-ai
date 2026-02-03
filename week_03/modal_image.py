# Optional: keep image definition separate. Not used by default.
import pathlib
import modal

LOCAL_DIR = pathlib.Path(__file__).parent.resolve()
REMOTE_DIR = "/root/work"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "ninja-build", "git")
    .pip_install("pandas")
    .add_local_dir(local_path=str(LOCAL_DIR), remote_path=REMOTE_DIR)
)
