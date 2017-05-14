#ifndef CUDASTRUCTURES
#define CUDASTRUCTURES

#include "../helper/helper.cuh"
#include <stdint.h>
#include <stdio.h>

template<typename V>
struct Node
{
    V value;
    struct Node<V>* next;
    struct Node<V>* prev;
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
		tmp->prev = NULL;
		if(Count > 0)
		{
			head->prev = tmp;
		}
		head = tmp;
		++Count;
	}

	__device__ void deleteValue(Node<T>* node)
	{
		if(Count > 0)
		{
			node->prev->next = node->next;
			node->next->prev = node->prev;
			free(node);
			--Count;
		}
	}

	__device__ void Dispose()
	{
		if (Count > 0)
		{
			Node<T>* tmp = head;
			while(head != NULL)
			{
				head = tmp->next;
				free(tmp);
			}
		}
	}

	__device__  ~CUDALinkedList()
	{
		Dispose();
	}
};

#endif
