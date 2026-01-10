import modal

app = modal.App("resources-demo")

@app.function(cpu=2.0, memory=4096)
def heavy():
    return "reserved >= 2 cores and >= 4096 MB"

@app.local_entrypoint()
def main():
    print(heavy.remote())
