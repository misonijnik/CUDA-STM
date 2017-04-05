#ifndef CUDASTM
#define CUDASTM
#include "helper/helper.cuh"
#include "structures/CUDAStructures.cuh"

__host__ int hey(void);
__global__ void changeArray(CUDAArray<WriteEntry> arr, int* ptr);

#endif
