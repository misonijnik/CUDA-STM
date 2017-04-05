#include "CUDAStructures.cuh"

__device__ unsigned int GetVersion(GLTEntry* entry )
{
	return (unsigned int)(entry->version);
}
