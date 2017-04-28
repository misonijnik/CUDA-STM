#ifndef CUDASTRUCTURES
#define CUDASTRUCTURES

#include "../helper/helper.cuh"
#include <stdint.h>

typedef union
{
	int iVal;
	double dVal;
	char chVal;
}Value;

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
	Value value;
	unsigned version : 11;
} ReadEntry;

typedef struct WriteEntry
{
	void* cudaPtr;
	Value value;
} WriteEntry;



class GlobalLockTable
{
private:
	CUDAArray<GLTEntry> _glt;
	void* _sharedMemPtr;
	size_t _memSize;
	size_t _wordSize;
	size_t _numberWordLock;

	__host__ __device__ unsigned long hash(void* cudaPtr)
	{
		//must control range
		unsigned long tmp = (uintptr_t(cudaPtr) - (uintptr_t(_sharedMemPtr)))/(_wordSize*_numberWordLock);
		return tmp;
	}

public:

	__host__ GlobalLockTable(void* sharedMemPtr, size_t memSize, size_t wordSize, size_t numberWordLock)
	{
		_sharedMemPtr = sharedMemPtr;
		_memSize = memSize;
		_wordSize = wordSize;
		_numberWordLock = numberWordLock;
		_glt = CUDAArray<GLTEntry>(_memSize/(_wordSize*_numberWordLock));
	}

	__device__ GLTEntry getEntryAt(void* cudaPtr)
	{
		return _glt.At(hash(cudaPtr));
	}

	__device__ void setEntryAt(void* cudaPtr, GLTEntry entry)
	{
		_glt.SetAt(hash(cudaPtr), entry);
	}


};


#endif