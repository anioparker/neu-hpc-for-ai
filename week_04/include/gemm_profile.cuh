#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"


void cpu_gemm(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int x = 0; x < M; x++) {
        for (int y = 0; y < N; y++) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = alpha * sum + beta * C[x * N + y];
        }
    }
}

int run_benchmark(int M, int N, int K) {
    // Allocate host memory that uses fp32
    float *h_A = (float*) malloc(M * K * sizeof(float));
    float *h_B = (float*) malloc(K * N * sizeof(float));

    float *h_C = (float*) malloc(M * N * sizeof(float));
    float *h_C_ref = (float*) malloc(M * N * sizeof(float)); // the results from the cpu gemm
    
    std::cout << "Allocated host memory" << std::endl;


    // C = alpha * A * B + beta * C
    float alpha = 1.0;
    float beta = 1.0;

    // Initialize random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize fp32 host memory so that it is filled with random numbers
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    for (int i = 0; i < M * N; ++i) h_C[i] = dis(gen);

    // set to random floats as well if beta is not set to 0.
    for (int i = 0; i < M * N; ++i) h_C_ref[i] = h_C[i];

    std::cout << "Initialized matrices" << std::endl;

    if(true) cpu_gemm(M, N, K, alpha, h_A, h_B, beta, h_C_ref);

    // Allocate device memory for float  elements (2 bytes)
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, M*K*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, K*N*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, M*N*sizeof(float)));

    std::cout << "Allocated device memory" << std::endl;


    // we use CUDA stream to generate events that are helpful for timing kernel execution
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));


    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice, stream));
    std::cout << "Copied matrices to device" << std::endl;

    
    // warming the GPU by just executing the kernel 10 times sequentially
    // and then discarding the result and not recording any timing.
    for(int i = 0; i < 10; i++) { // warmup
        checkCudaErrors(cudaMemcpyAsync(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice, stream));
        matmul(stream, M, N, K, alpha, d_A, d_B, beta, d_C);
        checkCudaErrors(cudaPeekAtLastError());   // catch launch/config errors early
    }
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());            // read and clear any surfaced errors

    // can assume the GPU is ready for profiling


    cudaEvent_t start, stop;
    // creating events is allocating memory for them
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // recording a start event on the stream to keep track of when our profiling run started
    checkCudaErrors(cudaEventRecord(start, stream));

    // collecting 50 different samples for the matmul's runtime
    constexpr int ITERS = 50;
    for(int i = 0; i < ITERS; i++) {
        // on each run, we need to reset the device matrix C if beta > 0
        // the time it takes to do this memcpy is being tracked in the start-stop event duration
        //  so its making our kernel measure slower than what it actually is
        checkCudaErrors(cudaMemcpyAsync(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice, stream));
        matmul(stream, M, N, K, alpha, d_A, d_B, beta, d_C);
        checkCudaErrors(cudaPeekAtLastError());   // catch launch/config errors early
    }
    checkCudaErrors(cudaEventRecord(stop, stream));

    // why is it sufficient to synchronize on the stop event 
    //  rather than needing to a cudaDeviceSynchronize?
    checkCudaErrors(cudaEventSynchronize(stop));   // one sync to surface runtime errors
    checkCudaErrors(cudaGetLastError());           // now catch them

    // ms is the total time to execute all 50 iterations of kernel launches consecutively
    float ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
    float us_per = (ms * 1e3f) / ITERS;
    // us_per is the number of microseconds elapsed for a single kernel execution of matmul


    // per matmul, alpha * A * B
    double flops = double(M) * double(K) * double(N) + double(M) * double(N);   
    double tflops = (flops / us_per) * 1e-6;                    // TFLOP/s

    printf("Avg kernel time: %.2f us\n", us_per);
    printf("Achieved: %.2f TFLOP/s\n", tflops);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost, stream));
    std::cout << "Copied result back to host" << std::endl;

    // compare the output of the CPU impl with the output of the kernel
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if( error > 0.1 ) { // large because of float vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 20) std::cout << "Too many errors to show them all.\n";            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;
    std::cout << "Error percent: " << error_count*100.0/(M*N) << "%" << std::endl;

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    int M, N, K;
    M = 8;
    K = 8;
    N = 8;
    run_benchmark(M, N, K);
    return 0;
}
