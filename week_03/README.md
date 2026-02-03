# cuda-matmul-optimization-h100

Replicate the CUDA-MMM SGEMM worklog (siboehm.com/articles/22/CUDA-MMM) on an NVIDIA H100 using Modal.
This repo builds a small SGEMM runner (`sgemm`), runs multiple kernel variants, and saves results as CSV.

## Repo layout
- `src/` + `include/`: CUDA kernels + runner
- `scripts/calc_bounds.py`: reproduces FLOPs / bytes / arithmetic intensity calculations
- `scripts/sweep.py`: runs kernels and writes CSV into `results/raw/`
- `modal_run.py`: builds + runs the full sweep on an H100

## Build locally (optional)
> If you have a GPU that is not H100, set `CMAKE_CUDA_ARCHITECTURES` accordingly.

```bash
# Example: local GPU
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=XX
cmake --build build -j

# Run one kernel
./build/sgemm --kernel 0 --m 4092 --n 4092 --k 4092 --reps 50
./build/sgemm --kernel 5 --m 4092 --n 4092 --k 4092 --reps 50
