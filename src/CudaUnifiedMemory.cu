#include "CudaUnifiedMemory.hpp"
#include <cuda_runtime.h>

CudaUnifiedMemory::CudaUnifiedMemory(std::size_t size) : size_(size)
{
    cudaMallocManaged(&pointer_, size_);
}

CudaUnifiedMemory::~CudaUnifiedMemory()
{
    cudaFree(pointer_);
}

void *CudaUnifiedMemory::getPointer()
{
    return pointer_;
}
