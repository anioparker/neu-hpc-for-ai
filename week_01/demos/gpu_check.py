import modal

image = modal.Image.debian_slim().pip_install("torch")
app = modal.App("gpu-check", image=image)

@app.function(gpu="T4")
def run():
    import torch
    assert torch.cuda.is_available()
    return torch.cuda.get_device_name(0)

@app.local_entrypoint()
def main():
    print(run.remote())

