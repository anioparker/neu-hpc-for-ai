#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define CHECK_CUDA(cmd) do { cudaError_t e = cmd; if (e) { fprintf(stderr, "CUDA error %d\n", e); exit(1); } } while(0)
#define CHECK_NCCL(cmd) do { ncclResult_t r = cmd; if (r) { fprintf(stderr, "NCCL error %s\n", ncclGetErrorString(r)); exit(1); } } while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    CHECK_CUDA(cudaSetDevice(rank));

    // Create NCCL communicator
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // count = elements each rank sends to each other rank
    size_t count = 4;
    size_t total = count * nranks;

    // Allocate device buffers (must be distinct — no in-place)
    float *d_send, *d_recv;
    CHECK_CUDA(cudaMalloc(&d_send, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_recv, total * sizeof(float)));

    // Fill sendbuf on host: chunk j = data destined for rank j
    // Convention: value = rank * 100 + destination * 10 + element index
    float* h_send = (float*)malloc(total * sizeof(float));
    for (int j = 0; j < nranks; j++)
        for (int k = 0; k < (int)count; k++)
            h_send[j * count + k] = rank * 100.0f + j * 10.0f + k;

    CHECK_CUDA(cudaMemcpy(d_send, h_send, total * sizeof(float), cudaMemcpyHostToDevice));

    // Single-call all-to-all
    CHECK_NCCL(ncclAlltoAll(d_send, d_recv, count, ncclFloat, comm, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Verify: recvbuf chunk i should contain data sent by rank i to us
    float* h_recv = (float*)malloc(total * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_recv, d_recv, total * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Rank %d received:", rank);
    for (int i = 0; i < (int)total; i++) printf(" %.0f", h_recv[i]);
    printf("\n");

    // Cleanup
    free(h_send);
    free(h_recv);
    cudaFree(d_send);
    cudaFree(d_recv);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}