#include <stdio.h>
#include "STM.cuh"
#include "helper/helper.cuh"
#include "structures/CUDAStructures.cuh"

__host__ int hey(void)
{
	GlobalLockTable hah = GlobalLockTable(1, 100);

	CUDAArray<WriteEntry> hmm = CUDAArray<WriteEntry>(1, 10);
	int* ptr;
	cudaMalloc((void**)&ptr, sizeof(int));
	cudaError_t error = cudaDeviceSynchronize();
	changeArray<<<1,1>>>(hmm, ptr);
	error = cudaDeviceSynchronize();
	WriteEntry* entryPtr = hmm.GetData();
	int* tmp = entryPtr[0].value;
	int intTmp = *tmp;
	printf("Здравствуй, %d мир!\n", intTmp);
	error = cudaGetLastError();
	return 0;
}

__global__ void changeArray(CUDAArray<WriteEntry> arr, int* ptr)
{
	WriteEntry tmp;
	tmp.cudaPtr = ptr;
	tmp.value = 6;

	arr.SetAt(0, 0, tmp);
}
