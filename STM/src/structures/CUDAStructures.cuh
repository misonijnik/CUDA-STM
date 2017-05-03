#ifndef CUDASTRUCTURES
#define CUDASTRUCTURES

#include "../helper/helper.cuh"
#include <stdint.h>

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

	CUDALinkedList()
	{
		Count = 0;
		head = NULL;
	}

	Node<T>* getHead()
	{
		return head;
	}

	void push(T val)
	{
		Node<T>* tmp = (Node<T>*)malloc(sizeof(Node<T>));
		tmp->value = val;
		tmp->next = head;
		tmp->prev = NULL;
		head->prev = tmp;
		head = tmp;
		Count++;
	}

	void deleteValue(Node<T>* node)
	{
		if(Count > 0)
		{
			node->prev->next = node->next;
			node->next->prev = node->prev;
			free(node);
			--Count;
		}
	}

	void Dispose()
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

	__host__ __device__  ~CUDALinkedList()
	{

	}
};

#endif
