#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct pair_t {
    float key;
    int val;
} pair_t;

void swapCPU(pair_t *a, pair_t *b) {
    pair_t temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void swap(pair_t *a, pair_t *b) {
    pair_t temp = *a;
    *a = *b;
    *b = temp;
}

float *Input;
pair_t *InputPair;
pair_t *Output;

int compare_pair(const void *a, const void *b)
{
    float floatA = ((pair_t *)a)->key;
    float floatB = ((pair_t *)b)->key;

    if (floatA < floatB) return -1;
    if (floatA > floatB) return 1;
    return 0;
}

int compare_verify(const void *p1, const void *p2)
{
    pair_t *a = (pair_t *)p1;
    pair_t *b = (pair_t *)p2;

    if(a->key == b->key)
    {
        return a->val - b->val;
    }
    else if(a->key < b->key)
    {
        return -1;
    }
    else
    {
        return 1;
    }
}

__device__ void maxHeapify(pair_t heap[], int size, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < size && heap[left].key > heap[largest].key) {
        largest = left;
    }

    if (right < size && heap[right].key > heap[largest].key) {
        largest = right;
    }

    if (largest != i) {
        swap(&heap[i], &heap[largest]);
        maxHeapify(heap, size, largest);
    }
}

__device__ void heapifyUp(pair_t heap[], int *size, int pos) {
    int parent = (pos - 1) / 2;

    while (pos > 0 && heap[parent].key < heap[pos].key) {
        swap(&heap[parent], &heap[pos]);
        pos = parent;
        parent = (pos - 1) / 2;
    }
}

__device__ void insert(pair_t heap[], int *size, pair_t element) {
    if (*size == 0) {
        heap[0] = element;
        (*size)++;
    } else {
        heap[*size] = element;
        (*size)++;
        heapifyUp(heap, size, *size - 1);
    }
}

__device__ void decreaseMax(pair_t heap[], int size, pair_t element) {
    if (size == 0) return;

    if (heap[0].key > element.key) {
        heap[0].key = element.key;
        heap[0].val = element.val;
        maxHeapify(heap, size, 0);
    }
}

__global__ void findKSmallestKernel(pair_t *InputPair, pair_t *Output, int nTotalElements, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ pair_t localHeap[256]; 
    int localHeapSize = 0;

    for (int i = idx; i < nTotalElements; i += stride) {
        if (localHeapSize < k) {
            insert(localHeap, &localHeapSize, InputPair[i]);
        } else if (InputPair[i].key < localHeap[0].key) {
            decreaseMax(localHeap, localHeapSize, InputPair[i]);
        }
    }
    
    __syncthreads();

    if (threadIdx.x == 0) { 
        for (int i = 0; i < localHeapSize; i++) {
            if (Output[0].key > localHeap[i].key) {
                decreaseMax(Output, k, localHeap[i]);
            }
        }
    }
}

void findKSmallest(int nTotalElements, int k) {
    pair_t *d_InputPair, *d_Output;
    int blockSize = 256;
    int gridSize = (nTotalElements + blockSize - 1) / blockSize;

    cudaMalloc(&d_InputPair, nTotalElements * sizeof(pair_t));
    cudaMalloc(&d_Output, k * sizeof(pair_t));

    cudaMemcpy(d_InputPair, InputPair, nTotalElements * sizeof(pair_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Output, Output, k * sizeof(pair_t), cudaMemcpyHostToDevice);

    findKSmallestKernel <<< gridSize, blockSize >>> (d_InputPair, d_Output, nTotalElements, k);

    cudaMemcpy(Output, d_Output, k * sizeof(pair_t), cudaMemcpyDeviceToHost);

    cudaFree(d_InputPair);
    cudaFree(d_Output);
}

void heapifyUpCPU(pair_t heap[], int *size, int pos) {
    int parent = (pos - 1) / 2;

    while (pos > 0 && heap[parent].key < heap[pos].key) {
        swapCPU(&heap[parent], &heap[pos]);
        pos = parent;
        parent = (pos - 1) / 2;
    }
}

void insertCPU(pair_t heap[], int *size, pair_t element) {
    if (*size == 0) {
        heap[0] = element;
        (*size)++;
    } else {
        heap[*size] = element;
        (*size)++;
        heapifyUpCPU(heap, size, *size - 1);
    }
}

void fillArrayRandom(int nTotalElements) {
    pair_t element;
    int size = 0;
    for(int i = 0; i < nTotalElements; i++) {
        int a = rand();
        int b = rand();

        float v = a * 100.0 + b;
        Input[i] = v;

        element.key = v;
        element.val = i;

        // Use CPU-based insert here since this is a host function.
        insertCPU(InputPair, &size, element);
    }
}

int binary_search(pair_t *array, int size, pair_t target) {
    int low = 0;
    int high = size - 1;
    
    while (low <= high) {
        int mid = (low + high) / 2;
        if (compare_verify(&target, &array[mid]) == 0) {
            return mid;  // Target found
        } else if (compare_verify(&target, &array[mid]) < 0) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    
    return -1;  // Target not found
}

void verifyOutput(const float *Input, const pair_t *Output, int nTotalElmts, int k) {
    int ok = 1;
    pair_t Answers[k];

    // 1) Create an array I of pairs (key, value)
    pair_t *I = (pair_t *)malloc(nTotalElmts * sizeof(pair_t));

    for(int i = 0; i < nTotalElmts; i++) {
        I[i].key = Input[i];
        I[i].val = i;
    }

    // 2) Sort the array I in ascending order
    qsort(I, nTotalElmts, sizeof(pair_t), compare_pair);

    // 3) Insert first k values of I into the Answers array
    for (int i = 0; i < k; i++) {
        Answers[i].key = I[i].key;
        Answers[i].val = I[i].val;
    }

    // 3) For each pair (ki,vi) belonging to the Output array
    for (int i = 0; i < k; i++) {
        if (binary_search(Answers, k, Output[i]) == -1) {
            ok = 0;
            break;
        }
    }

    if (ok) {
        printf("\nOutput set verified correctly.\n");
    } else {
        printf("\nOutput set DID NOT compute correctly!!!\n");
    }

    free(I);
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <nTotalElements> <k>\n", argv[0]);
        return 1;
    }

    int nTotalElements = atoi(argv[1]);
    int k = atoi(argv[2]);
    clock_t start, end;
    double cpu_time_used;

    if (nTotalElements <= 0) {
        printf("Error: nTotalElements must be a positive integer.\n");
        return 1;
    }

    if (k > nTotalElements || k <= 0) {
        printf("Error: k must be a positive integer and less than or equal to nTotalElements.\n");
        return 1;
    }

    Input = (float *)malloc(nTotalElements * sizeof(float));
    InputPair = (pair_t *)malloc(nTotalElements * sizeof(pair_t));
    Output = (pair_t *)malloc(k * sizeof(pair_t));
    fillArrayRandom(nTotalElements);

    start = clock();
    findKSmallest(nTotalElements, k);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f seconds\n", cpu_time_used);

    verifyOutput(Input, Output, nTotalElements, k);

    free(Input);
    free(InputPair);
    free(Output);

    return 0;
}
