#ifndef CUDASTM
#define CUDASTM
#include "helper.cuh"
#include "CUDAStructures.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_VERSION 2047

/*typedef	struct __align__(4)
{
	unsigned version : 11;
	unsigned owner : 19;
	unsigned pre_locked : 1;
	unsigned locked : 1;
}gle;*/





/*typedef union
{
	gle entry;
	unsigned int i;
}GlobalLockEntry;*/



typedef	struct __align__(4)
{
	unsigned version : 11;
	unsigned index : 19;
	unsigned pre_locked : 1;
	unsigned locked : 1;
}LocalLockEntry;

typedef struct 
{
	void* cudaPtr;
	unsigned int size;
	char value[16];
	unsigned version : 11;
}ReadEntry;

typedef struct
{
	void* cudaPtr;
	unsigned int size;
	char value[16];
}WriteEntry;

typedef	struct
{
	unsigned int memSize;
	unsigned int wordSize;
}GlobalLockTableInfo;

class GlobalLockTable
{
private:
	CUDAArray<unsigned int> _glt;
	void* _sharedMemPtr;
	CUDAArray<uint2> _info; // x - memSize, y - wordSize
	unsigned int _lengthInfo;
	size_t _length;

	__device__ void checkIndex(unsigned long tmp)
	{
		if (tmp > _length - 1)
		{
			printf("error %lu\n", tmp);
		}
	}

public:
	__host__ GlobalLockTable(void* sharedMemPtr, uint2* info, unsigned int lengthInfo)
	{
		_sharedMemPtr = sharedMemPtr;
		_info = CUDAArray<uint2>(info, lengthInfo);

		_lengthInfo = lengthInfo;
		//printf("%u, %u \n", info[0].x, info[0].y);
		_length = 0;
		for (size_t i = 0; i < lengthInfo; i++)
		{
			_length += info[i].x / info[i].y;
		}
		_glt = CUDAArray<unsigned int>(_length);
	}

	__device__ size_t getLength()
	{
		return _length;
	}

	__device__ unsigned int getEntryAt(unsigned long index)
	{
		checkIndex(index);
		return _glt.At(index);
	}

	__device__ unsigned int* getEntryAtPtr(unsigned long index)
	{
		checkIndex(index);
		return _glt.AtPtr(index);
	}

	__device__ unsigned int getEntryAt(void* cudaPtr)
	{
		return _glt.At(hash(cudaPtr));
	}

	__device__ unsigned int* getEntryAtPtr(void* cudaPtr)
	{
		return _glt.AtPtr(hash(cudaPtr));
	}

	__device__ void setEntryAt(void* cudaPtr, unsigned int entry)
	{
		_glt.SetAt(hash(cudaPtr), entry);
	}

	__device__ unsigned long hash(void* cudaPtr)
	{
		//TODO control range
		uintptr_t tmpIndex = (uintptr_t(cudaPtr) - (uintptr_t(_sharedMemPtr)));
		size_t tableIndex = 0;
		for (size_t i = 0; i < _lengthInfo; i++)
		{
			if (tmpIndex >= _info.At(i).x)
			{
				tableIndex += _info.At(i).x / _info.At(i).y;
				tmpIndex -= _info.At(i).x;
			}
			else
			{
				tableIndex += tmpIndex / _info.At(i).y;
				//break;
				return tableIndex;
			}
		}
		printf("ERROR");
		/*if (uniqueIndex() == 105)
		{
			printf("%lu one thread %lu\n", tableIndex, uniqueIndex());
			printf("%lu two thread %lu\n", (uintptr_t(cudaPtr) - (uintptr_t(_sharedMemPtr))) / _info.At(0).y, uniqueIndex());
		}*/
		
		//printf("%u, %u \n", _info.At(0).x, _info.At(0).y);
		//return (uintptr_t(cudaPtr) - (uintptr_t(_sharedMemPtr))) / _info.At(0).y;
		return 0;
	}

	__host__ void Dispose()
	{
		//_glt.Dispose();
		//_info.Dispose();
	}

	__host__ ~GlobalLockTable()
	{
		Dispose();
	}

};


class LocalMetadata
{
private:
	GlobalLockTable* g_lock;
	CUDASet<ReadEntry> readSet;
	CUDASet<WriteEntry> writeSet;
	CUDASet<LocalLockEntry> lockSet;
	bool abort;

	/*_device__ GlobalLockEntry calcPreLockedVal(unsigned int version, unsigned int index)
	{
		GlobalLockEntry tmp;
		tmp.entry.version = version;
		tmp.entry.owner = index;
		tmp.entry.pre_locked = 1;
		tmp.entry.locked = 0;
		return tmp;
	}

	__device__ GlobalLockEntry calcLockedVal(unsigned int version)
	{
		GlobalLockEntry tmp;
		tmp.entry.version = version;
		tmp.entry.owner = 0;
		tmp.entry.pre_locked = 0;
		tmp.entry.locked = 1;
		return tmp;
	}

	__device__ GlobalLockEntry calcUnlockVal(unsigned int version)
	{
		GlobalLockEntry tmp;
		tmp.entry.version = version;
		tmp.entry.owner = 0;
		tmp.entry.pre_locked = 0;
		tmp.entry.locked = 0;
		return tmp;
	}*/

	__device__ __host__ unsigned int getVersion(unsigned int val)
	{
		return (val >> 21);
	}

	__device__ __host__ unsigned int getOwner(unsigned int val)
	{
		return ((val << 11) >> 13);
	}

	__device__ __host__ unsigned int getPreLockBit(unsigned int val)
	{
		return ((val >> 1) & 1);
	}

	__device__ __host__ unsigned int getLockBit(unsigned int val)
	{
		return (val & 1);
	}

	__device__ __host__ void setVersion(unsigned int* val, unsigned int version)
	{
		unsigned int tmp;
		tmp = *val;
		*val = ((tmp << 11) >> 11) + (version << 21);
	}

	__device__ __host__ unsigned int calcPreLockedVal(unsigned int version, unsigned int index)
	{
		return (version << 21) + (index << 2) + 2;
	}

	__device__ __host__ unsigned int calcLockedVal(unsigned int version)
	{
		return (version << 21) + 1;
	}

	__device__ __host__ unsigned int calcUnlockVal(unsigned int version)
	{
		return (version << 21);
	}

	__device__ bool tryPreLock()
	{
		unsigned int tmpLock;
		unsigned int preLockVal;
		unsigned int length = lockSet.getCount();
		for (size_t i = 0; i < length; ++i)
		{
			do
			{
				tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
				
				if (getVersion(tmpLock) != lockSet.getByIndex(i)->version || \
					getLockBit(tmpLock) == 1 || \
					(getPreLockBit(tmpLock) == 1 && getOwner(tmpLock) < uniqueIndex()))
				{
					releaseLocks();
					return false;
				}
				preLockVal = calcPreLockedVal(lockSet.getByIndex(i)->version, uniqueIndex());
			} while (atomicCAS((g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index)), \
				tmpLock, preLockVal) != tmpLock);
			lockSet.getByIndex(i)->pre_locked = 1;
		}
		/*for (size_t i = 0; i < lockSet.getCount(); i++)
		{
			if (lockSet.getByIndex(i)->index == 0)
			{
				GlobalLockEntry tmp = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
				printf("thread %u: locked %u  owner %u version %u\n", uniqueIndex(), tmp.entry.pre_locked, tmp.entry.owner, tmp.entry.version);
			}
		}*/
		return true;
	}


	__device__ bool tryLock()
	{
		unsigned int preLockVal;
		unsigned int finalLockVal;
		unsigned int length = lockSet.getCount();
		for (size_t i = 0; i < length; ++i)
		{
			preLockVal = calcPreLockedVal(lockSet.getByIndex(i)->version, uniqueIndex());
			finalLockVal = calcLockedVal(lockSet.getByIndex(i)->version);
			if (atomicCAS((g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index)), \
				preLockVal, finalLockVal) != preLockVal)
			{
				releaseLocks();
				return false;
			}
			lockSet.getByIndex(i)->pre_locked = 0;
			lockSet.getByIndex(i)->locked = 1;
		}
		/*for (size_t i = 0; i < lockSet.getCount(); i++)
		{
			if (lockSet.getByIndex(i)->index == 0)
			{
				unsigned int tmp = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
				printf("thread %u: locked %u  owner %u version %u\n", uniqueIndex(), getLockBit(tmp), getOwner(tmp), getVersion(tmp));
			}
		}*/
		return true;
	}

public:
	LocalMetadata()
	{
		abort = false;
		g_lock = NULL;
	}

	__device__ LocalMetadata(GlobalLockTable* glt)
	{
		abort = false;
		g_lock = glt;
	}

	__device__ void txStart()
	{
		readSet = CUDASet<ReadEntry>();
		writeSet = CUDASet<WriteEntry>();
		lockSet = CUDASet<LocalLockEntry>();
		abort = false;
	}

	template<typename T>
	__device__ T txRead(T* ptr)
	{
		T val;
		if(getLockBit(g_lock->getEntryAt(ptr)) == 0)
		{
			bool isFound = false;
			unsigned int length = writeSet.getCount();
			for (size_t i = 0; i < length; i++)
			{
				if (writeSet.getByIndex(i)->cudaPtr == ptr)
				{
					isFound = true;
					memcpy(&val, writeSet.getByIndex(i)->value, sizeof(T));
					//val = writeSet.getByIndex(i)->value;
					break;
				}
			}
			if(!isFound)
			{
				ReadEntry tmpReadEntry;
				tmpReadEntry.cudaPtr = ptr;

				val = *ptr;
				//memcpy(&val, ptr, sizeof(T));
				//val = atomicAdd(ptr, 0);

				//tmpReadEntry.value = val;
				memcpy(tmpReadEntry.value, &val, sizeof(T));


				tmpReadEntry.size = sizeof(T);
				//unsigned int tmpLock = atomicAdd(g_lock->getEntryAtPtr(ptr), 0);
				tmpReadEntry.version = getVersion(g_lock->getEntryAt(ptr));

				/*if(uniqueIndex() == 0)
				{
					printf("read val %d\n", *(int*)tmpReadEntry.value);
				}*/

				readSet.Add(tmpReadEntry);
			}
		}
		else
		{
			val = 0;
			abort = true;
		}
		return val;
	}

	template<typename T>
	__device__ void txWrite(T* ptr, T value)
	{
		T val = value;
		if(getLockBit(g_lock->getEntryAt(ptr)) == 0)
		{
			bool isFound = false;
			unsigned int length = writeSet.getCount();
			for (size_t i = 0; i < length; i++)
			{
				if (writeSet.getByIndex(i)->cudaPtr == ptr)
				{
					isFound = true;
					//writeSet.getByIndex(i)->value = &val;
					memcpy(writeSet.getByIndex(i)->value, &val, sizeof(T));
					break;
				}
			}
			if(!isFound)
			{
				WriteEntry tmpWriteEntry;


				//tmpWriteEntry.value = &val;
				memcpy(tmpWriteEntry.value, &val, sizeof(T));


				tmpWriteEntry.size = sizeof(T);
				tmpWriteEntry.cudaPtr = ptr;
				writeSet.Add(tmpWriteEntry);

				LocalLockEntry tmpLocalLockEntry;
				tmpLocalLockEntry.index = g_lock->hash(ptr);
				tmpLocalLockEntry.version = getVersion(g_lock->getEntryAt(ptr));
				lockSet.Add(tmpLocalLockEntry);
			}
		}
		else
		{
			abort = true;
		}
	}

	__device__ bool txValidate()
	{
		if(tryPreLock() == true)
		{
			unsigned int length = readSet.getCount();
			for (size_t i = 0; i < length; i++)
			{				
				if (getVersion(g_lock->getEntryAt(readSet.getByIndex(i)->cudaPtr)) != readSet.getByIndex(i)->version)
				{
					return false;
				}				
			}			
			return tryLock();
		}
		else
		{
			return false;
		}
	}

	__device__ void txCommit()
	{
		unsigned int length = writeSet.getCount();
		/*for (size_t i = 0; i < lockSet.getCount(); i++)
		{
			if (lockSet.getByIndex(i)->index == 99)
			{
				printf("thread %u: %d, %d, %d\n", uniqueIndex(), *(int*)(readSet.getByIndex(i)->value), *(int*)(writeSet.getByIndex(i)->cudaPtr), *(int*)(writeSet.getByIndex(i)->value));
			}
		}*/
		for (size_t i = 0; i < length; i++)
		{
			
			//*(writeSet.getByIndex(i)->cudaPtr) = writeSet.getByIndex(i)->value;

			memcpy(writeSet.getByIndex(i)->cudaPtr, writeSet.getByIndex(i)->value, writeSet.getByIndex(i)->size);

			/*WriteEntry<T> tmp;
			writeSet.getByIndex(i, &tmp);
			*(writeSet.getByIndex(i)->cudaPtr) = tmp.value;*/
			
		}
		__threadfence();
		length = lockSet.getCount();
		for (size_t i = 0; i < length; i++)
		{
			/*if (lockSet.getByIndex(i)->index == 0)
			{
				printf("thread %u: version %u %u\n", uniqueIndex(), (*(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index))).entry.version, lockSet.getByIndex(i)->version + 1);
			}*/
			if (lockSet.getByIndex(i)->version < MAX_VERSION)
			{
				setVersion(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index), lockSet.getByIndex(i)->version + 1);
				//(*(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index))).entry.version = lockSet.getByIndex(i)->version + 1;
			}
			else
			{
				setVersion(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index), 0);
			}
		}
	}


	__device__ void releaseLocks()
	{
		unsigned int length = lockSet.getCount();
		unsigned int preLockVal;
		unsigned int unLockVal;
		unsigned int tmpLock;

		for (size_t i = 0; i < length; i++)
		{
			tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
			unLockVal = calcUnlockVal(getVersion(tmpLock));
			if (lockSet.getByIndex(i)->pre_locked == 1)
			{
				preLockVal = calcPreLockedVal(getVersion(tmpLock), uniqueIndex());
				if(atomicCAS((g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index)), \
					preLockVal, preLockVal - 1) == preLockVal)
				{
					lockSet.getByIndex(i)->locked = 1;
				}
			}
		}

		for (size_t i = 0; i < length; i++)
		{
			tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
			unLockVal = calcUnlockVal(getVersion(tmpLock));
			if (lockSet.getByIndex(i)->locked == 1)
			{
				*(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index)) = unLockVal;
			}
		}
	}

	__device__ bool isAborted()
	{
		return abort;
	}

	__device__ CUDASet<ReadEntry>* getReadSet()
	{
		return &readSet;
	}

	__device__ CUDASet<WriteEntry>* getWriteSet()
	{
		return &writeSet;
	}

	__device__ CUDASet<LocalLockEntry>* getLockSet()
	{
		return &lockSet;
	}
};




__host__ int hey();
//__host__ int hey2();
__host__ int hey3();
__host__ int testGlt();
__global__ void changeArray(CUDAArray<WriteEntry> arr, int* ptr, int val);
__global__ void testCorrectSTM(GlobalLockTable g_lock, int* cudaPtr);

#endif
