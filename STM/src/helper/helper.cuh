#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef CUDAHELPER
#define CUDAHELPER

template<typename T> class CUDASet
{
private:
	size_t realCount;
	void __host__ __device__ upsize()
	{
		T* tmpPtr;
		cudaError_t error = cudaMalloc((void**)&tmpPtr, 2*realCount*sizeof(T));
		error = cudaDeviceSynchronize();
		cudaMemcpy(tmpPtr, cudaPtr, sizeof(T)*(realCount), cudaMemcpyDeviceToDevice);
		error = cudaDeviceSynchronize();
		realCount *= 2;
		cudaFree(cudaPtr);
		error = cudaDeviceSynchronize();
		cudaPtr = tmpPtr;
		error = cudaGetLastError();
	}

public:
	T* cudaPtr;
	size_t Count;

	__host__ __device__ CUDASet()
	{
		Count = 0;
		realCount = 4;
		cudaError_t error = cudaMalloc((void**)&cudaPtr, realCount*sizeof(T));
		error = cudaDeviceSynchronize();
		error = cudaGetLastError();
	}

	__host__ __device__ CUDASet(const CUDASet& set)
	{
		cudaPtr = set.cudaPtr;
		Count = set.Count;
		realCount = set.realCount;
	}

	CUDASet(T* cpuPtr, unsigned int count)
	{
		Count = count;
		cudaError_t error = cudaMalloc((void**)&cudaPtr, realCount*sizeof(T));
		error = cudaDeviceSynchronize();
		cudaMemcpy(cudaPtr, cpuPtr, sizeof(T)*(Count), cudaMemcpyHostToDevice);
		error = cudaDeviceSynchronize();
		error = cudaGetLastError();
	}


};

template<typename T> class CUDAArray
{
private:
public:
	T* cudaPtr;
	size_t Length;

	CUDAArray()
	{

	}

	__host__ __device__ CUDAArray(const CUDAArray& arr)
	{
		cudaPtr = arr.cudaPtr;
		Length = arr.Length;
	}

	CUDAArray(T* cpuPtr, int length)
	{
		Length = length;
		cudaError_t error = cudaMalloc((void**)&cudaPtr, length*sizeof(T));
		error = cudaDeviceSynchronize();
		error = cudaMemcpy(cudaPtr, cpuPtr, length*sizeof(T), cudaMemcpyHostToDevice);
		error = cudaDeviceSynchronize();
	}

	CUDAArray(int length)
	{
		Length = length;
		cudaError_t error = cudaMalloc((void**)&cudaPtr, length*sizeof(T));
		error = cudaDeviceSynchronize();
	}

	T* GetData()
	{
		T* arr = (T*)malloc(sizeof(T)*Length);
		GetData(arr);
		return arr;
	}

	void GetData(T* arr)
	{
		cudaError_t error = cudaMemcpy(arr, cudaPtr, Length*sizeof(T), cudaMemcpyDeviceToHost);
		error = cudaDeviceSynchronize();

	}

	__device__ T At(int index)
	{
		return cudaPtr[index];
	}

	__device__ T* AtPtr(int index)
	{
		return &(cudaPtr[index]);
	}

	__device__ void SetAt(int index, T value)
	{
		cudaPtr[index] = value;
	}

	void Dispose()
	{
		cudaFree(cudaPtr);
	}

	__host__ __device__  ~CUDAArray()
	{

	}
};

template class CUDAArray<int>;

template class CUDAArray<float>;

#endif

#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess) {\
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(0); }}

