import modal, subprocess

app = modal.App("cuda-smi")

@app.function(gpu="any")
def check():
    out = subprocess.check_output(["nvidia-smi"], text=True)
    print(out)
    return "Driver Version:" in out and "CUDA Version:" in out

@app.local_entrypoint()
def main():
    print(check.remote())
