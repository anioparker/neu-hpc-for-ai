from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.abspath(os.path.dirname(__file__))
tk_root = os.environ.get("THUNDERKITTENS_ROOT", "/root/thunderkittens")

extra_compile_args = {
    "cxx": ["-O3", "-std=c++20"],
    "nvcc": [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        # Blackwell native cubin + PTX fallback
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_100,code=compute_100",
    ],
}

setup(
    name="deepseek_moe_tk",
    version="0.1.0",
    description="DeepSeek-style MoE with ThunderKittens on Modal B200",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="deepseek_moe_tk_ext",
            sources=[
                os.path.join("csrc", "bindings.cpp"),
                os.path.join("csrc", "deepseek_moe_tk_kernel.cu"),
            ],
            include_dirs=[
                os.path.join(this_dir, "csrc"),
                tk_root,
                os.path.join(tk_root, "include"),
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
