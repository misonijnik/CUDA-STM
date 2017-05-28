#ifndef CUDASTRUCTURES
#define CUDASTRUCTURES

#include "helper.cuh"
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<typename V>
struct Node
{
    V value;
    struct Node<V>* next;
};

template<typename T>
class CUDALinkedList
{
private:
	unsigned int Count;
	Node<T>* head;

public:

	__device__ CUDALinkedList()
	{
		Count = 0;
		head = NULL;
	}

	__device__ unsigned int getCount()
	{
		return Count;
	}

	__device__ Node<T>* getHead()
	{
		return head;
	}

	__device__ void push(T val)
	{
		Node<T>* tmp = (Node<T>*)malloc(sizeof(Node<T>));
		tmp->value = val;
		tmp->next = head;
		head = tmp;
		++Count;
	}

	__device__ void Dispose()
	{
		if (Count > 0)
		{
			Node<T>* current = head; 
			Node<T>* next;
			while(current != NULL)
			{
				next = current->next;				
				free(current);
				current = next;
			}
			Count = 0;
		}
	}

	__device__  ~CUDALinkedList()
	{
	}
};

#endif
