cmake_minimum_required(VERSION 3.22)
project(cuda_mmm_h100 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CMake 3.22 plays nicer with numeric archs:
set(CMAKE_CUDA_ARCHITECTURES 90)

# Still emit sm_90a if nvcc supports it (H100). If this errors, delete this line.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90a")

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB KERNEL_SRCS ${CMAKE_SOURCE_DIR}/src/kernels/*.cu)

add_executable(sgemm
    src/main.cu
    src/cublas_fp32.cu
    src/verify.cu
    ${KERNEL_SRCS}
)

target_link_libraries(sgemm PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
