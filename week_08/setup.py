from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="deepseek_moe_cuda",
    version="0.1.0",
    packages=["deepseek_moe"],
    ext_modules=[
        CUDAExtension(
            name="deepseek_moe_cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/deepseek_moe_kernel.cu",
            ],
            include_dirs=["csrc/include"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)