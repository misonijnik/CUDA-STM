#ifndef CUDASTRUCTURES
#define CUDASTRUCTURES

#include "../helper/helper.cuh"

typedef	struct
{
	unsigned version : 11;
	unsigned owner : 19;
	unsigned locked : 1;
	unsigned pre_locked : 1;
} GLTEntry;

__device__ unsigned int GetVersion(GLTEntry* entry );

typedef struct ReadEntry
{
	void* cudaPtr;
	int value;
	unsigned version : 11;
} ReadEntry;

typedef struct WriteEntry
{
	void* cudaPtr;
	int value;
} WriteEntry;

typedef CUDAArray<GLTEntry> GlobalLockTable;

#endif
