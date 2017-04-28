#ifndef CUDASTM
#define CUDASTM
#include "helper/helper.cuh"
#include "structures/CUDAStructures.cuh"

__host__ int hey(void);
__host__ int hey2(void);
__host__ int testGlt(void);
__global__ void testGltKernel(GlobalLockTable g_lock, int* cudaPtr, int* val);
__global__ void changeArray(CUDAArray<WriteEntry> arr, int* ptr, int val);

#endif
