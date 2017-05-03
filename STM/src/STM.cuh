#ifndef CUDASTM
#define CUDASTM
#include "helper/helper.cuh"
#include "structures/CUDAStructures.cuh"

#define MAX_VERSION 524288

typedef	struct
{
	unsigned version : 11;
	unsigned owner : 19;
	unsigned locked : 1;
	unsigned pre_locked : 1;
}GlobalLockEntry;

typedef	struct
{
	unsigned version : 11;
	unsigned index : 21;
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

template<typename T>
class GlobalLockTable
{
private:
	CUDAArray<GlobalLockEntry> _glt;
	T* _sharedMemPtr;
	size_t _memSize;
	size_t _wordSize;
	size_t _numberWordLock;
	size_t _length;
public:

	__host__ GlobalLockTable(T* sharedMemPtr, size_t memSize, size_t numberWordLock)
	{
		_sharedMemPtr = sharedMemPtr;
		_wordSize = sizeof(T);
		_memSize = memSize*_wordSize;
		_numberWordLock = numberWordLock;
		_length = _memSize/(_wordSize*_numberWordLock);
		_glt = CUDAArray<GlobalLockEntry>(_length);
	}

	__device__ size_t getLength()
	{
		return _length;
	}

	__device__ GlobalLockEntry getEntryAt(unsigned int index)
	{
		return _glt.At(index);
	}

	__device__ GlobalLockEntry* getEntryAtPtr(unsigned int index)
	{
		return _glt.AtPtr(index);
	}

	__device__ GlobalLockEntry getEntryAt(T* cudaPtr)
	{
		return _glt.At(hash(cudaPtr));
	}

	__device__ GlobalLockEntry* getEntryAtPtr(T* cudaPtr)
	{
		return _glt.AtPtr(hash(cudaPtr));
	}

	__device__ void setEntryAt(T* cudaPtr, GlobalLockEntry entry)
	{
		_glt.SetAt(hash(cudaPtr), entry);
	}

	__host__ __device__ unsigned long hash(T* cudaPtr)
	{
		//TODO control range
		unsigned long tmp = (uintptr_t(cudaPtr) - (uintptr_t(_sharedMemPtr)))/(_wordSize*_numberWordLock);
		return tmp;
	}
};

template<typename T>
class LocalMetadata
{
private:
	GlobalLockTable<T> g_lock;
	CUDALinkedList<ReadEntry<T> > readSet;
	CUDALinkedList<WriteEntry<T> > writeSet;
	CUDALinkedList<LocalLockEntry> lockSet;
	bool abort;

	GlobalLockEntry calcPreLockedVal(unsigned int version, unsigned int index)
	{
		GlobalLockEntry tmp;
		tmp.version = version;
		tmp.owner = index;
		tmp.pre_locked = 1;
		tmp.locked = 0;
		return tmp;
	}

	GlobalLockEntry calcLockedVal(unsigned int version)
	{
		GlobalLockEntry tmp;
		tmp.version = version;
		tmp.owner = 0;
		tmp.pre_locked = 0;
		tmp.locked = 1;
		return tmp;
	}

	GlobalLockEntry calcUnlockVal(unsigned int version)
	{
		GlobalLockEntry tmp;
		tmp.version = version;
		tmp.owner = 0;
		tmp.pre_locked = 0;
		tmp.locked = 0;
		return tmp;
	}

	__device__ bool tryPreLock()
	{
		Node<LocalLockEntry>* tmpNode = lockSet.getHead();
		GlobalLockEntry tmpLock;
		GlobalLockEntry preLockVal;
		while(tmpNode != NULL)
		{
			do
			{
				tmpLock = g_lock.getAt(tmpNode->value.index);
				if(tmpLock.version != tmpNode->value.version || \
					tmpLock.locked == 1 || \
					(tmpLock.pre_locked == 1 &&	tmpLock.owner < uniqueIndex()))
				{
					releaseLocks();
					return false;
				}
				preLockVal = calcPreLockedVal(tmpLock.version, uniqueIndex());
			} while(atomicCAS(g_lock.getAtPrt(tmpNode->value.index), \
					tmpLock, preLockVal) != tmpLock);
			tmpNode = tmpNode->next;
		}
		return true;
	}


	__device__ bool tryLock()
	{
		Node<LocalLockEntry>* tmpNode = lockSet.getHead();
		GlobalLockEntry tmpLock;
		GlobalLockEntry preLockVal;
		GlobalLockEntry finalLockVal;
		while(tmpNode != NULL)
		{
			preLockVal = calcPreLockedVal(tmpLock.version, uniqueIndex());
			finalLockVal = calcLockedVal(tmpLock.version);
			if(atomicCAS(g_lock.getAtPrt(tmpNode->value.index), \
					preLockVal, finalLockVal) != preLockVal)
			{
				releaseLocks();
				return false;
			}
			tmpNode = tmpNode->next;
		}
		return true;
	}

	__device__ void releaseLocks()
	{
		Node<LocalLockEntry>* tmpNode = lockSet.getHead();
		GlobalLockEntry preLockVal;
		GlobalLockEntry unLockVal;
		GlobalLockEntry tmpLock;
		while(tmpNode != NULL)
		{
			tmpLock = g_lock.getAt(tmpNode->value.index);
			unLockVal = calcUnlockVal(tmpLock.version);
			if(tmpLock.pre_locked == 1)
			{
				preLockVal = calcPreLockedVal(tmpLock.version, uniqueIndex());
				atomicCAS(g_lock.getAtPrt(tmpNode->value.index), preLockVal, unLockVal);
			}
			if(tmpLock.locked == 1)
			{
				*(g_lock.getAtPtr(tmpNode->value.index)) = unLockVal;
			}
			tmpNode = tmpNode->next;
		}
	}

public:

	LocalMetadata(GlobalLockTable<T> glt)
	{
		txStart();
		g_lock = glt;
	}

	void txStart()
	{
		readSet = CUDALinkedList<ReadEntry<T> >();
		writeSet = CUDALinkedList<WriteEntry<T> >();
		lockSet = CUDALinkedList<LocalLockEntry>();
		abort = 0;
	}

	T txRead(T* ptr)
	{
		T val;
		if(g_lock.getEntryAt(ptr).locked == 0)
		{
			bool isFound = false;
			Node<WriteEntry<T> >* tmpNode = writeSet.getHead();
			while((tmpNode != 0) && !isFound)
			{
				if(tmpNode->value.cudaPtr == ptr)
				{
					isFound = true;
					val = tmpNode->value.value;
				}
			}
			if(!isFound)
			{
				ReadEntry<T> tmpReadEntry;
				tmpReadEntry.value = *ptr;
				tmpReadEntry.version = g_lock.getEntryAt(ptr).version;
				readSet.push(tmpReadEntry);
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

	void txWrite(T* ptr, T val)
	{
		if(g_lock.getEntryAt(ptr).locked == 0)
		{
			bool isFound = false;
			Node<WriteEntry<T> >* tmpNode = writeSet.getHead();
			while((tmpNode != NULL) && !isFound)
			{
				if(tmpNode->value.cudaPtr == ptr)
				{
					isFound = true;
					tmpNode->value.value = val;
				}
			}
			if(!isFound)
			{
				WriteEntry<T> tmpWriteEntry;
				tmpWriteEntry.value = val;
				tmpWriteEntry.cudaPtr = ptr;
				writeSet.push(tmpWriteEntry);

				LocalLockEntry tmpLocalLockEntry;
				tmpLocalLockEntry.index = g_lock.hash(ptr);
				tmpLocalLockEntry.version = g_lock.getEntryAt(ptr).version;
				lockSet.push(tmpLocalLockEntry);
			}
		}
		else
		{
			abort = true;
		}
	}

	bool txValidate()
	{
		if(tryPreLock() == true)
		{
			Node<ReadEntry<T> >* tmpNode = readSet.getHead();
			while(tmpNode != NULL)
			{
				if(g_lock.getAt(tmpNode->value.cudaPtr).version != tmpNode->value.version)
				{
					return false;
				}
				tmpNode = tmpNode->next;
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
		Node<WriteEntry<T> >* tmpNode = writeSet.getHead();
		while(tmpNode != NULL)
		{
			*(tmpNode->value.cudaPtr) = tmpNode->value.val;
			tmpNode = tmpNode->next;
		}
		__threadfence();
		Node<LocalLockEntry>* tmpLockNode = lockSet.getHead();
		while(tmpLockNode != NULL)
		{
			if(tmpLockNode->value.version < MAX_VERSION)
			{
				*(g_lock.getAtPtr(tmpLockNode->value.index)).version +=1;
			}
			else
			{
				*(g_lock.getAtPtr(tmpLockNode->value.index)).version = 0;
			}
			tmpLockNode = tmpLockNode->next;
		}

	}

	bool isAborted()
	{
		return abort;
	}

	CUDALinkedList<ReadEntry<T> > getReadSet()
	{
		return readSet;
	}

	CUDALinkedList<WriteEntry<T> > getWriteSet()
	{
		return writeSet;
	}

	CUDALinkedList<LocalLockEntry> getLockSet()
	{
		return lockSet;
	}
};




__host__ int hey();
__host__ int hey2();
__host__ int testGlt();
__global__ void testGltKernel(GlobalLockTable<int> g_lock, int* cudaPtr, int* val);
__global__ void changeArray(CUDAArray<WriteEntry<int> > arr, int* ptr, int val);

#endif
