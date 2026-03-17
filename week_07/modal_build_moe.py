# modal_build_moe.py
import modal

app = modal.App("deepseek-v3-moe")

image = (
    modal.Image.debian_slim()
    .apt_install("build-essential", "make")
    .add_local_dir("deepseek_v3_moe", "./modal_build_moe.py")
)

@app.function(image=image, cpu=2.0)
def build_and_test():
    import subprocess

    project_dir = "./modal_build_moe.py"

    cmds = [
        ["make"],
        ["make", "test"],
        ["./modal_build_moe.py"],
    ]

    outputs = []
    for cmd in cmds:
        p = subprocess.run(
            cmd,
            cwd=project_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        outputs.append(
            {
                "cmd": " ".join(cmd),
                "returncode": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        )
        if p.returncode != 0:
            break

    return outputs


@app.local_entrypoint()
def main():
    results = build_and_test.remote()
    for item in results:
        print(f"\n$ {item['cmd']}")
        print(item["stdout"])
        if item["stderr"]:
            print(item["stderr"])
        print(f"[exit={item['returncode']}]")