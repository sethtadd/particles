#ifndef CUDA_UNIFIED_MEMORY_HPP
#define CUDA_UNIFIED_MEMORY_HPP

#include <cstddef>

class CudaUnifiedMemory
{
public:
    CudaUnifiedMemory(std::size_t size);
    ~CudaUnifiedMemory();

    void *getPointer();

private:
    void *pointer_;
    std::size_t size_;
};

#endif // CUDA_UNIFIED_MEMORY_HPP
