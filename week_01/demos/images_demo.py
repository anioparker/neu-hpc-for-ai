import modal

app = modal.App("images-demo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("pandas==2.2.0", "numpy")
)

@app.function(image=image)
def show_env():
    import pandas as pd
    import numpy as np
    return (pd.__version__, np.__version__)

@app.local_entrypoint()
def main():
    print(show_env.remote())
