#include <stdio.h>
#include <stdint.h>
#include "STM.cuh"
#include "helper/helper.cuh"
#include "structures/CUDAStructures.cuh"

__host__ int hey(void)
{
	cudaError_t error = cudaDeviceSynchronize();
	cudaCheckError();
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaCheckError();
	int device;
	for (device = 0; device < deviceCount; ++device)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cudaCheckError();
		printf("Device %d has compute capability %d.%d.%lu.\n", device, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem);
	}
	return 0;
}

__host__ int hey2(void)
{
	CUDAArray<WriteEntry> hmm = CUDAArray<WriteEntry>(10);
	cudaDeviceSynchronize();
	cudaCheckError();
	int* ptr;
	cudaMalloc((void**)&ptr, sizeof(int)*2);
	int* ptr2 = ptr + 1;
	cudaDeviceSynchronize();
	cudaCheckError();
	printf("%u.\n", (uintptr_t)ptr);
	printf("%u.\n", (uintptr_t)(ptr2));
	printf("%u.\n", ((uintptr_t)(ptr2) - (uintptr_t)(ptr)));
	printf("%p.\n", (ptr2));
	printf("%p.\n", (void*)(19662336));
	changeArray<<<1,1>>>(hmm, ptr, 7);
	cudaDeviceSynchronize();
	cudaCheckError();

	WriteEntry* entryPtr;
	entryPtr = hmm.GetData();

	cudaCheckError();

	int* tmp = (int*)malloc(sizeof(int));
	*tmp= entryPtr[0].value.iVal;
	int intTmp = *tmp;
	printf("Здравствуй, %d мир!\n", intTmp);
	cudaCheckError();

	free(tmp);
	cudaFree(ptr);

	return 0;
}

__host__ int testGlt(void)
{
	int* ptr;
	cudaMalloc((void**)&ptr, sizeof(int)*4);
	cudaDeviceSynchronize();
	cudaCheckError();
	int* value;
	cudaMalloc((void**)&value, sizeof(int));
	GlobalLockTable g_lock = GlobalLockTable(ptr, sizeof(int)*4, sizeof(int), 1);
	testGltKernel<<<1,1>>>(g_lock, ptr, value);
	int* val = (int*)malloc(sizeof(int));
	cudaMemcpy(val, value, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d.\n", (*val));
	//printf("%lu.\n", g_lock.hash(ptr+1));
	return 0;
}

__global__ void testGltKernel(GlobalLockTable g_lock, int* cudaPtr, int* val)
{
	GLTEntry tmp;
	tmp.locked = 1;

	g_lock.setEntryAt(cudaPtr, tmp);
	tmp.locked = 1;
	g_lock.setEntryAt(cudaPtr+1, tmp);
	*val = g_lock.getEntryAt(cudaPtr).locked;
}

__global__ void changeArray(CUDAArray<WriteEntry> arr, int* ptr, int val)
{
	WriteEntry tmp;
	tmp.cudaPtr = ptr;
	tmp.value.iVal = val;

	arr.SetAt(0, tmp);
}