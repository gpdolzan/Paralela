#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

typedef struct pair_t {
    float key;
    int val;
} pair_t;

typedef struct thread_data_t {
    pair_t *localHeap;
    int start;
    int end;
    int k;
    int localHeapSize;
    int localMaxSize;
} thread_data_t;

float *Input;
pair_t *InputPair;
pair_t *Output;

void swap(pair_t *a, pair_t *b) {
    pair_t temp = *a;
    *a = *b;
    *b = temp;
}

// Comparison function for qsort
int compare(const void *a, const void *b)
{
    float floatA = *(float *)a;
    float floatB = *(float *)b;

    if (floatA < floatB) return -1;
    if (floatA > floatB) return 1;
    return 0;
}

// Make a compare function that tests val and key
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

// Comparison function for qsort
int compare_pair(const void *a, const void *b)
{
    float floatA = ((pair_t *)a)->key;
    float floatB = ((pair_t *)b)->key;

    if (floatA < floatB) return -1;
    if (floatA > floatB) return 1;
    return 0;
}

void maxHeapify(pair_t heap[], int size, int i) {
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

void heapifyUp(pair_t heap[], int *size, int pos) {
    int parent = (pos - 1) / 2;

    while (pos > 0 && heap[parent].key < heap[pos].key) {
        swap(&heap[parent], &heap[pos]);
        pos = parent;
        parent = (pos - 1) / 2;
    }
}

void insert(pair_t heap[], int *size, pair_t element)
{
    if (*size == 0) {
        heap[0] = element;
        (*size)++;
    } else {
        heap[*size] = element;
        (*size)++;
        heapifyUp(heap, size, *size - 1);
    }
}

void decreaseMax(pair_t heap[], int size, pair_t element)
{
    if (size == 0) return;

    if (heap[0].key > element.key)
    {
        heap[0].key = element.key;
        heap[0].val = element.val;
        maxHeapify(heap, size, 0);
    }
}

void fillArrayRandom(int nTotalElements)
{
    pair_t element;
    int size = 0;
    for(int i = 0; i < nTotalElements; i++)
    {
        int a = rand();
        int b = rand();

        float v = a * 100.0 + b;
        Input[i] = v;

        element.key = v;
        element.val = i;

        insert(InputPair, &size, element);
    }
}

int binary_search(pair_t *array, int size, pair_t target)
{
    int low = 0;
    int high = size - 1;
    
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (compare_verify(&target, &array[mid]) == 0)
        {
            return mid;  // Target found
        }
        else if (compare_verify(&target, &array[mid]) < 0)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
    
    return -1;  // Target not found
}

void verifyOutput(const float *Input, const pair_t *Output, int nTotalElmts, int k)
{
    int ok = 1;
    pair_t Answers[k];

    // 1) Create an array I of pairs (key, value)
    pair_t *I = (pair_t *)malloc(nTotalElmts * sizeof(pair_t));

    for(int i = 0; i < nTotalElmts; i++)
    {
        I[i].key = Input[i];
        I[i].val = i;
    }

    // 2) Sort the array I in ascending order
    qsort(I, nTotalElmts, sizeof(pair_t), compare_pair);

    // 3) Insert first k values of I into the Answers array
    for (int i = 0; i < k; i++)
    {
        Answers[i].key = I[i].key;
        Answers[i].val = I[i].val;
    }

    // 3) For each pair (ki,vi) belonging to the Output array
    for (int i = 0; i < k; i++)
    {
        if (binary_search(Answers, k, Output[i]) == -1)
        {
            ok = 0;
            break;
        }
    }

    if (ok)
    {
        printf("\nOutput set verified correctly.\n");
    } else
    {
        printf("\nOutput set DID NOT compute correctly!!!\n");
    }

    free(I);
}

void *thread_function(void *arg)
{
    thread_data_t *data = (thread_data_t *)arg;
    for (int i = data->start; i < data->end; i++)
    {
        if (data->localHeapSize < data->k)
        {
            insert(data->localHeap, &(data->localHeapSize), InputPair[i]);
        }
        else if (InputPair[i].key < data->localHeap[0].key)
        {
            decreaseMax(data->localHeap, data->localHeapSize, InputPair[i]);
        }
    }
    return NULL;
}

__device__ void swapDevice(pair_t *a, pair_t *b) {
    pair_t temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void maxHeapifyDevice(pair_t heap[], int size, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    while (i < size) {
        largest = i;

        if (left < size && heap[left].key > heap[largest].key)
            largest = left;

        if (right < size && heap[right].key > heap[largest].key)
            largest = right;

        if (largest == i)
            break;

        swapDevice(&heap[i], &heap[largest]);
        i = largest;
        left = 2 * i + 1;
        right = 2 * i + 2;
    }
}

__device__ void heapifyUpDevice(pair_t heap[], int pos) {
    int parent = (pos - 1) / 2;

    while (pos > 0 && heap[parent].key < heap[pos].key) {
        swapDevice(&heap[parent], &heap[pos]);
        pos = parent;
        parent = (pos - 1) / 2;
    }
}

__device__ void insertDevice(pair_t heap[], int *size, pair_t element)
{
    if (*size == 0) {
        heap[0] = element;
        (*size)++;
    } else {
        heap[*size] = element;
        (*size)++;
        heapifyUpDevice(heap, *size - 1);
    }
}

__device__ void decreaseMaxDevice(pair_t heap[], int size, pair_t element)
{
    if (size == 0) return;

    if (heap[0].key > element.key)
    {
        heap[0].key = element.key;
        heap[0].val = element.val;
        maxHeapifyDevice(heap, size, 0);
    }
}

__global__ void findKSmallestKernel(pair_t *InputPair, pair_t *d_intermediateResults, int nTotalElements, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    extern __shared__ pair_t sharedMemory[];
    pair_t* localHeap = &sharedMemory[threadIdx.x * k];

    int localHeapSize = 0;

    for (int i = idx; i < nTotalElements; i += stride)
    {
        if (localHeapSize < k)
        {
            insertDevice(localHeap, &localHeapSize, InputPair[i]);
        }
        else if (InputPair[i].key < localHeap[0].key)
        {
            decreaseMaxDevice(localHeap, localHeapSize, InputPair[i]);
        }
    }

    // Write local heap results to global intermediate results.
    for (int i = 0; i < k; i++)
    {
        d_intermediateResults[idx * k + i] = localHeap[i];
    }
}

__global__ void mergeKSmallestKernel(pair_t *d_intermediateResults, pair_t *d_Output, int k, int totalThreads, int numBlocks)
{
    // One block will handle the merging. Each thread in the block will handle one value.
    extern __shared__ pair_t sharedHeap[];

    int idx = threadIdx.x;

    // Each thread in this block will insert elements from all the intermediate heaps.
    for (int i = 0; i < min(numBlocks, k); i++)
    {
        if (idx < k)
        {
            pair_t currentPair = d_intermediateResults[i * k + idx];

            if (idx == 0)
            {
                // Initialize the shared heap
                for (int j = 0; j < k; j++)
                {
                    sharedHeap[j] = currentPair;
                }
            }
            else
            {
                if (currentPair.key < sharedHeap[0].key)
                {
                    sharedHeap[0] = currentPair;
                    maxHeapifyDevice(sharedHeap, k, 0);
                }
            }
        }
        __syncthreads();
    }

    // The sharedHeap now has the k smallest values.
    if (idx < k)
    {
        d_Output[idx] = sharedHeap[idx];
    }
}

void findKSmallest(int nTotalElements, int k)
{
    int blockSize = min(k, 256);
    int numBlocks = (nTotalElements + blockSize - 1) / blockSize;

    pair_t *d_InputPair, *d_intermediateResults, *d_Output;

    cudaMalloc(&d_InputPair, nTotalElements * sizeof(pair_t));
    cudaMalloc(&d_intermediateResults, numBlocks * blockSize * k * sizeof(pair_t));
    cudaMalloc(&d_Output, k * sizeof(pair_t));

    cudaMemcpy(d_InputPair, InputPair, nTotalElements * sizeof(pair_t), cudaMemcpyHostToDevice);

    findKSmallestKernel<<<numBlocks, blockSize>>>(d_InputPair, d_intermediateResults, nTotalElements, k);

    // Now merge the results.
    mergeKSmallestKernel<<<1, k, numBlocks * k * sizeof(pair_t)>>>(d_intermediateResults, d_Output, k, numBlocks);

    cudaMemcpy(Output, d_Output, k * sizeof(pair_t), cudaMemcpyDeviceToHost);

    cudaFree(d_InputPair);
    cudaFree(d_intermediateResults);
    cudaFree(d_Output);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <nTotalElements> <k>\n", argv[0]);
        return 1;
    }

    // Convert the arguments from strings to integers
    int nTotalElements = atoi(argv[1]);
    int k = atoi(argv[2]);
    clock_t start, end;
    double cpu_time_used;

    // Check the conditions
    if (nTotalElements <= 0)
    {
        printf("Error: nTotalElements must be a positive integer.\n");
        return 1;
    }

    if (k > nTotalElements || k <= 0)
    {
        printf("Error: k must be a positive integer and less than or equal to nTotalElements.\n");
        return 1;
    }

    // Program can begin
    Input = (float *)malloc(nTotalElements * sizeof(float));
    InputPair = (pair_t *)malloc(nTotalElements * sizeof(pair_t));
    Output = (pair_t *)malloc(k * sizeof(pair_t));
    fillArrayRandom(nTotalElements);

    start = clock();
    findKSmallest(nTotalElements, k);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f seconds", cpu_time_used);

    // Verify the output
    verifyOutput(Input, Output, nTotalElements, k);

    free(Input);
    free(InputPair);
    free(Output);

    return 0;
}