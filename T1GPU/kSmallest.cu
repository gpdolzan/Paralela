#include <cuda_runtime.h>
#include <stdio.h>

typedef struct pair_t {
    float key;
    int val;
} pair_t;

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

// Device function to partition array based on the key in pair_t
__device__ void partition(pair_t* data, int left, int right, int pivotIndex, int& pivotNewIndex) {
    float pivotValue = data[pivotIndex].key;
    int i = left;
    for (int j = left; j < right; j++) {
        if (data[j].key < pivotValue) {
            // Swap data[i] and data[j]
            pair_t temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
        }
    }
    // Swap data[i] and data[pivotIndex]
    pair_t temp = data[i];
    data[i] = data[pivotIndex];
    data[pivotIndex] = temp;
    pivotNewIndex = i;
}

__global__ void findKSmallest(pair_t* data, int k, int n, pair_t* results) {
    int tid = threadIdx.x;
    int pivotIndex = tid;
    
    __shared__ int shared_pivotNewIndex;
    
    if (tid == 0) {
        shared_pivotNewIndex = 0;
    }

    __syncthreads();

    if (tid < n) {
        partition(data, 0, n, pivotIndex, shared_pivotNewIndex);
        
        if (shared_pivotNewIndex == k) {
            results[tid] = data[shared_pivotNewIndex];
        }
    }

    __syncthreads();
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

    // Print both Answers and Output
    printf("\nAnswers:\n");
    for (int i = 0; i < k; i++)
    {
        printf("(%f, %d) ", Answers[i].key, Answers[i].val);
    }
    printf("\n\n");

    printf("Output:\n");
    for (int i = 0; i < k; i++)
    {
        printf("(%f, %d) ", Output[i].key, Output[i].val);
    }
    printf("\n\n");


    if (ok)
    {
        printf("\nOutput set verified correctly.\n");
    } else
    {
        printf("\nOutput set DID NOT compute correctly!!!\n");
    }

    free(I);
}

int main(int argc, char *argv[])
{
    int n = 10; // Sample size
    int k = 5; // We're finding the 5 smallest
    pair_t *d_data;
    pair_t *d_results;

    if (argc != 3)
    {
        printf("Usage: %s <nTotalElements> <k>\n", argv[0]);
        return 1;
    }

    // Convert the arguments from strings to integers
    //int nTotalElements = atoi(argv[1]);
    //int k = atoi(argv[2]);
    //clock_t start, end;
    //double cpu_time_used;

    // Check the conditions
    if (n <= 0)
    {
        printf("Error: nTotalElements must be a positive integer.\n");
        return 1;
    }

    if (k > n || k <= 0)
    {
        printf("Error: k must be a positive integer and less than or equal to nTotalElements.\n");
        return 1;
    }

    Input = (float*)malloc(n * sizeof(float));
    //h_data = (pair_t*)malloc(n * sizeof(pair_t));
    InputPair = (pair_t*)malloc(n * sizeof(pair_t));
    //h_results = (pair_t*)malloc(k * sizeof(pair_t));
    Output = (pair_t*)malloc(k * sizeof(pair_t));

    // Fill h_data with your max heap data...
    fillArrayRandom(n);

    cudaMalloc((void**)&d_data, n * sizeof(pair_t));
    cudaMalloc((void**)&d_results, k * sizeof(pair_t));

    cudaMemcpy(d_data, InputPair, n * sizeof(pair_t), cudaMemcpyHostToDevice);
    
    findKSmallest<<<1, n>>>(d_data, k, n, d_results);
    
    cudaMemcpy(Output, d_results, k * sizeof(pair_t), cudaMemcpyDeviceToHost);

    // h_results now contains k smallest values
    // Verify the results
    verifyOutput(Input, Output, n, k);

    // Cleanup
    free(Input);
    free(InputPair);
    free(Output);
    cudaFree(d_data);
    cudaFree(d_results);

    return 0;
}