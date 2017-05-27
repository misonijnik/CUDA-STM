#include <stdio.h>
#include <stdint.h>
#include "STM.cuh"
#include "helper.cuh"
#include "CUDAStructures.cuh"

__host__ int hey()
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

__host__ int hey3()
{
	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, size * 2);
	printf("%u.\n", size);
	double* ptr;
	cudaMalloc((void**)&ptr, sizeof(double)*100);
	cudaDeviceSynchronize();
	cudaCheckError();
	double* val = (double*)malloc(sizeof(double)*100);
	cudaCheckError();
	val[0] = 0;
	val[1] = 0;
	//cudaMemcpy(ptr, val, sizeof(int)*100, cudaMemcpyHostToDevice);
	cudaMemset(ptr, 0, sizeof(double) * 100);
	cudaDeviceSynchronize();
	cudaCheckError();
	GlobalLockTable g_lock = GlobalLockTable(ptr, sizeof(double), 100, 1);
	cudaCheckError();
	testCorrectSTM<<<100,1024>>>(g_lock, ptr);//todo fix error with more block
	cudaDeviceSynchronize();
	cudaCheckError();
	g_lock.Dispose();
	cudaMemcpy(val, ptr, sizeof(double)*100, cudaMemcpyDeviceToHost);
	printf("%f.\n", (val[0]));
	printf("%f.\n", (val[1]));
	free(val);
	cudaFree(val);

	return 0;
}

__host__ int hey2()
{
	CUDAArray<WriteEntry<int> > hmm = CUDAArray<WriteEntry<int> >(10);
	cudaDeviceSynchronize();
	cudaCheckError();
	int* ptr;
	cudaMalloc((void**)&ptr, sizeof(int)*2);
	int* ptr2 = ptr + 1;
	cudaDeviceSynchronize();
	cudaCheckError();
	printf("%lu.\n", (uintptr_t)ptr);
	printf("%lu.\n", (uintptr_t)(ptr2));
	printf("%lu.\n", ((uintptr_t)(ptr2) - (uintptr_t)(ptr)));
	printf("%p.\n", (ptr2));
	printf("%p.\n", (void*)(19662336));
	changeArray<<<1,1>>>(hmm, ptr, 7);
	cudaDeviceSynchronize();
	cudaCheckError();

	WriteEntry<int>* entryPtr;
	entryPtr = hmm.GetData();

	cudaCheckError();

	int* tmp = (int*)malloc(sizeof(int));
	*tmp= entryPtr[0].value;
	int intTmp = *tmp;
	printf("Здравствуй, %d мир!\n", intTmp);
	cudaCheckError();

	free(tmp);
	cudaFree(ptr);

	return 0;
}

__host__ int testGlt()
{
	int* ptr;
	cudaMalloc((void**)&ptr, sizeof(int)*4);
	cudaDeviceSynchronize();
	cudaCheckError();
	int* value;
	cudaMalloc((void**)&value, sizeof(int));
	GlobalLockTable g_lock = GlobalLockTable(ptr, sizeof(int), 4, 1);
	testGltKernel<<<1,1>>>(g_lock, ptr, value);
	int* val = (int*)malloc(sizeof(int));
	cudaMemcpy(val, value, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d.\n", (*val));
	//printf("%lu.\n", g_lock.hash(ptr+1));
	return 0;
}

__global__ void testGltKernel(GlobalLockTable g_lock, int* cudaPtr, int* val)
{
	GlobalLockEntry tmp;
	tmp.entry.locked = 1;

	g_lock.setEntryAt(cudaPtr, tmp);
	tmp.entry.locked = 1;
	g_lock.setEntryAt(cudaPtr+1, tmp);
	*val = g_lock.getEntryAt(cudaPtr).entry.locked;
}

__global__ void testCorrectSTM(GlobalLockTable g_lock, double* cudaPtr)
{
	LocalMetadata<double> local_data = LocalMetadata<double>(&g_lock);
	size_t length = g_lock.getLength();
	size_t count = blockDim.x*gridDim.x;
	unsigned int tmpOne = 0;
	unsigned int tmpTwo = 0;
	unsigned int tmp = uniqueIndex();
	tmp = tmp % 100;
	/*if (uniqueIndex() + 1 > count/2)
	{
		tmpOne++;
	}
	else 
	{
		tmpTwo++;
	}*/
	double val = 0;
	do
	{
		local_data.txStart();
		val = local_data.txRead(cudaPtr + tmp);
		if(local_data.isAborted())
		{
			local_data.releaseLocks();
			continue;
		}
		val++;
		local_data.txWrite(cudaPtr + tmp, val);
		if(local_data.isAborted())
		{
			local_data.releaseLocks();
			continue;
		}

		if(local_data.txValidate())
		{
			local_data.txCommit();
			local_data.releaseLocks();
			break;
		}
	} while (true);
	//printf("thread %u, val %d\n", uniqueIndex(), val);
}

__global__ void changeArray(CUDAArray<WriteEntry<int> > arr, int* ptr, int val)
{
	WriteEntry<int> tmp;
	tmp.cudaPtr = ptr;
	tmp.value = val;

	arr.SetAt(0, tmp);
}
