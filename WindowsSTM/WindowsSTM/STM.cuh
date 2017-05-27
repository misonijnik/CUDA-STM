#ifndef CUDASTM
#define CUDASTM
#include "helper.cuh"
#include "CUDAStructures.cuh"

#define MAX_VERSION 2047

typedef	struct
{
	unsigned version : 11;
	unsigned owner : 19;
	unsigned pre_locked : 1;
	unsigned locked : 1;
}gle;

typedef union
{
	gle entry;
	int i;
}GlobalLockEntry;



typedef	struct
{
	unsigned version : 11;
	unsigned index : 20;
	unsigned locked : 1;
}LocalLockEntry;

template<typename T>
struct ReadEntry
{
	T* cudaPtr;
	T value;
	unsigned version : 11;
};

template<typename T>
struct WriteEntry
{
	T* cudaPtr;
	T value;
};


class GlobalLockTable
{
private:
	CUDAArray<GlobalLockEntry> _glt;
	void* _sharedMemPtr;
	size_t _memSize;
	size_t _wordSize;
	size_t _numberWordLock;
	size_t _length;

	__device__ void checkIndex(unsigned long tmp)
	{
		if (tmp > _length - 1)
		{
			printf("error %lu\n", tmp);
		}
	}
public:

	/*__host__ __device__ GlobalLockTable(const GlobalLockTable& table)
	{
		_sharedMemPtr = table._sharedMemPtr;
		_wordSize = table._wordSize;
		_memSize = table._memSize;
		_numberWordLock = table._numberWordLock;
		_length = table._length;
		_glt = table._glt;
	}*/

	__host__ GlobalLockTable(void* sharedMemPtr, size_t wordSize, size_t memSize, size_t numberWordLock)
	{
		_sharedMemPtr = sharedMemPtr;
		_wordSize = wordSize;
		_memSize = memSize*_wordSize;
		_numberWordLock = numberWordLock;
		_length = memSize /_numberWordLock;
		_glt = CUDAArray<GlobalLockEntry>(_length);
	}

	__device__ size_t getLength()
	{
		return _length;
	}

	__device__ GlobalLockEntry getEntryAt(unsigned long index)
	{
		checkIndex(index);
		return _glt.At(index);
	}

	__device__ GlobalLockEntry* getEntryAtPtr(unsigned long index)
	{
		checkIndex(index);
		return _glt.AtPtr(index);
	}

	template<typename T>
	__device__ GlobalLockEntry getEntryAt(T* cudaPtr)
	{
		return _glt.At(hash(cudaPtr));
	}

	template<typename T>
	__device__ GlobalLockEntry* getEntryAtPtr(T* cudaPtr)
	{
		return _glt.AtPtr(hash(cudaPtr));
	}

	template<typename T>
	__device__ void setEntryAt(T* cudaPtr, GlobalLockEntry entry)
	{
		_glt.SetAt(hash(cudaPtr), entry);
	}

	__host__ __device__ unsigned long hash(void* cudaPtr)
	{
		//TODO control range
		unsigned long tmp = (uintptr_t(cudaPtr) - (uintptr_t(_sharedMemPtr)))/(_wordSize*_numberWordLock);
		
		return tmp;
	}

	__host__ void Dispose()
	{
		_glt.Dispose();
	}

	__host__ ~GlobalLockTable()
	{
		Dispose();
	}

};

template<typename T>
class LocalMetadata
{
private:
	GlobalLockTable* g_lock;
	const unsigned int sizeBuf = 1000;
	CUDASet<ReadEntry<T> > readSet;
	CUDASet<WriteEntry<T> > writeSet;
	CUDASet<LocalLockEntry > lockSet;
	bool abort;

	__device__ GlobalLockEntry calcPreLockedVal(unsigned int version, unsigned int index)
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
	}

	__device__ bool tryPreLock()
	{
		GlobalLockEntry tmpLock;
		GlobalLockEntry preLockVal;
		unsigned int length = lockSet.getCount();
		for (size_t i = 0; i < length; ++i)
		{
			do
			{
				tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
				//printf("thread %u: %u %u, %u\n", uniqueIndex(), tmpLock.entry.version, tmpNode->value.version, tmpLock.entry.owner);
				if (tmpLock.entry.version != lockSet.getByIndex(i)->version || \
					tmpLock.entry.locked == 1 || \
					(tmpLock.entry.pre_locked == 1 && tmpLock.entry.owner < uniqueIndex()))
				{
					releaseLocks();
					return false;
				}
				preLockVal = calcPreLockedVal(tmpLock.entry.version, uniqueIndex());
			} while (atomicCAS((int*)g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index), \
				tmpLock.i, preLockVal.i) != tmpLock.i);
		}
		return true;
	}


	__device__ bool tryLock()
	{
		GlobalLockEntry tmpLock;
		GlobalLockEntry preLockVal;
		GlobalLockEntry finalLockVal;
		unsigned int length = lockSet.getCount();
		for (size_t i = 0; i < length; ++i)
		{
			tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
			preLockVal = calcPreLockedVal(tmpLock.entry.version, uniqueIndex());
			finalLockVal = calcLockedVal(tmpLock.entry.version);
			if (atomicCAS((int*)(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index)), \
				preLockVal.i, finalLockVal.i) != preLockVal.i)
			{
				releaseLocks();
				return false;
			}
			lockSet.getByIndex(i)->locked = 1;
		}
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
		readSet = CUDASet<ReadEntry<T> >();
		writeSet = CUDASet<WriteEntry<T> >();
		lockSet = CUDASet<LocalLockEntry>();
		abort = false;
	}

	__device__ T txRead(T* ptr)
	{
		T val;
		if(g_lock->getEntryAt<T>(ptr).entry.locked == 0)
		{
			bool isFound = false;
			unsigned int length = writeSet.getCount();
			for (size_t i = 0; i < length; i++)
			{
				if (writeSet.getByIndex(i)->cudaPtr == ptr)
				{
					isFound = true;
					val = writeSet.getByIndex(i)->value;
					break;
				}
			}
			if(!isFound)
			{
				ReadEntry<T> tmpReadEntry;
				tmpReadEntry.cudaPtr = ptr;
				tmpReadEntry.value = *ptr;
				tmpReadEntry.version = g_lock->getEntryAt<T>(ptr).entry.version;
				readSet.Add(tmpReadEntry);
				val = tmpReadEntry.value;
			}
		}
		else
		{
			val = 0;
			abort = true;
		}
		return val;
	}

	__device__ void txWrite(T* ptr, T val)
	{
		if(g_lock->getEntryAt(ptr).entry.locked == 0)
		{
			bool isFound = false;
			unsigned int length = writeSet.getCount();
			for (size_t i = 0; i < length; i++)
			{
				if (writeSet.getByIndex(i)->cudaPtr == ptr)
				{
					isFound = true;
					writeSet.getByIndex(i)->value = val;
					break;
				}
			}
			if(!isFound)
			{
				WriteEntry<T> tmpWriteEntry;
				tmpWriteEntry.value = val;
				tmpWriteEntry.cudaPtr = ptr;
				writeSet.Add(tmpWriteEntry);
				LocalLockEntry tmpLocalLockEntry;
				tmpLocalLockEntry.index = g_lock->hash(ptr);
				tmpLocalLockEntry.version = g_lock->getEntryAt<T>(ptr).entry.version;
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
				if (g_lock->getEntryAt<T>(readSet.getByIndex(i)->cudaPtr).entry.version != readSet.getByIndex(i)->version)
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
		for (size_t i = 0; i < length; i++)
		{
			*(writeSet.getByIndex(i)->cudaPtr) = writeSet.getByIndex(i)->value;

			//memcpy(writeSet.getByIndex(i)->cudaPtr, &(writeSet.getByIndex(i)->value), sizeof(T));

			/*WriteEntry<T> tmp;
			writeSet.getByIndex(i, &tmp);
			*(writeSet.getByIndex(i)->cudaPtr) = tmp.value;*/
		}
		__threadfence();
		length = lockSet.getCount();
		for (size_t i = 0; i < length; i++)
		{
			if (lockSet.getByIndex(i)->version < MAX_VERSION)
			{
				(*(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index))).entry.version += 1;
			}
			else
			{
				(*(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index))).entry.version = 0;
			}
		}
	}


	__device__ void releaseLocks()
	{
		unsigned int length = lockSet.getCount();
		GlobalLockEntry preLockVal;
		GlobalLockEntry unLockVal;
		GlobalLockEntry tmpLock;

		for (size_t i = 0; i < length; i++)
		{
			tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
			unLockVal = calcUnlockVal(tmpLock.entry.version);
			if (tmpLock.entry.pre_locked == 1)
			{
				preLockVal = calcPreLockedVal(tmpLock.entry.version, uniqueIndex());
				atomicCAS((int*)g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index), \
					preLockVal.i, unLockVal.i);
			}
		}

		for (size_t i = 0; i < length; i++)
		{
			tmpLock = g_lock->getEntryAt(lockSet.getByIndex(i)->index);
			unLockVal = calcUnlockVal(tmpLock.entry.version);
			if (lockSet.getByIndex(i)->locked == 1)
			{
				*(g_lock->getEntryAtPtr(lockSet.getByIndex(i)->index)) = unLockVal;
			}
		}

		readSet.Clear();
		writeSet.Clear();
		lockSet.Clear();
	}

	__device__ bool isAborted()
	{
		return abort;
	}

	__device__ CUDASet<ReadEntry<T> >* getReadSet()
	{
		return &readSet;
	}

	__device__ CUDASet<WriteEntry<T> >* getWriteSet()
	{
		return &writeSet;
	}

	__device__ CUDASet<LocalLockEntry>* getLockSet()
	{
		return &lockSet;
	}
};




__host__ int hey();
__host__ int hey2();
__host__ int hey3();
__host__ int testGlt();
__global__ void testGltKernel(GlobalLockTable g_lock, int* cudaPtr, int* val);
__global__ void changeArray(CUDAArray<WriteEntry<int> > arr, int* ptr, int val);
__global__ void testCorrectSTM(GlobalLockTable g_lock, double* cudaPtr);

#endif
