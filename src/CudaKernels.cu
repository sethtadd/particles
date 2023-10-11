#include <stdio.h>
#include "CudaKernels.hpp"

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i % 10000 == 0)
    {
        printf("i: %d\n", i);
    }

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__host__ void runVectorAdd(const float *h_A, const float *h_B, float *h_C, int numElements)
{
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(float);

    // Allocate device vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    // Check for allocation errors
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate device vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    // Check for allocation errors
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate device vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    // Check for allocation errors
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy host vectors A and B to device vectors A and B
    printf("Copy input data from the host memory to CUDA device memory\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the vectorAdd CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Check for errors during kernel execution
    err = cudaDeviceSynchronize(); // Blocks execution until kernel is finished
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy device vector C to host vector C
    printf("Copy output data from CUDA device memory to host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
