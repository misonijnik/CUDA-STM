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
	size_t deviceStride;
public:
	T* cudaPtr;
	size_t Height;
	size_t Width;
	size_t Stride;

	CUDAArray()
	{

	}

	__host__ __device__ CUDAArray(const CUDAArray& arr)
	{
		cudaPtr = arr.cudaPtr;
		Height = arr.Height;
		Width = arr.Width;
		Stride = arr.Stride;
		deviceStride = arr.deviceStride;
	}

	CUDAArray(T* cpuPtr, int width, int height)
	{
		Width = width;
		Height = height;
		cudaError_t error = cudaMallocPitch((void**)&cudaPtr, &Stride, Width*sizeof(T), Height);
		error = cudaDeviceSynchronize();
		deviceStride = Stride / sizeof(T);
		error = cudaMemcpy2D(cudaPtr, Stride, cpuPtr, Width*sizeof(T),
			Width*sizeof(T), Height, cudaMemcpyHostToDevice);
		error = cudaDeviceSynchronize();
		error = cudaGetLastError();
	}

	CUDAArray(int width, int height)
	{
		Width = width;
		Height = height;
		cudaError_t error = cudaMallocPitch((void**)&cudaPtr, &Stride, Width*sizeof(T), Height);
		error = cudaDeviceSynchronize();
		deviceStride = Stride / sizeof(T);
	}

	T* GetData()
	{
		T* arr = (T*)malloc(sizeof(T)*Width*Height);
		GetData(arr);
		return arr;
	}

	void GetData(T* arr)
	{
		cudaError_t error = cudaMemcpy2D(arr, Width*sizeof(T), cudaPtr, Stride, Width*sizeof(T), Height, cudaMemcpyDeviceToHost);
		error = cudaDeviceSynchronize();
	}

	__device__ T At(int row, int column)
	{
		return cudaPtr[row*deviceStride + column];
	}

	__device__ T* AtPtr(int row, int column)
	{
		return &(cudaPtr[row*deviceStride + column]);
	}

	__device__ void SetAt(int row, int column, T value)
	{
		cudaPtr[row*deviceStride + column] = value;
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
		exit(0);\
