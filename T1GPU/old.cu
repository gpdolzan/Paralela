#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

typedef struct pair_t {
    float key;
    int val;
} pair_t;

// Global arrays
float *Input;
pair_t *InputPair;
pair_t *Output;

__device__ void swap(pair_t *a, pair_t *b) {
    pair_t temp = *a;
    *a = *b;
    *b = temp;
}

// Comparison function for qsort
int compare(const void *a, const void *b)
{
    float floatA = ((pair_t *)a)->key;
    float floatB = ((pair_t *)b)->key;

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

__global__ void findKSmallestKernel(pair_t *InputPair, int nTotalElements, pair_t *Output, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < nTotalElements; i += stride) {
        if (i < k)
        {
            // Just insert directly
            Output[i] = InputPair[i];
        }
        else if (InputPair[i].key < Output[0].key)
        {
            // TODO: This needs a more robust mechanism for distributed decreaseMax in CUDA
            // For simplicity, the current version just updates the max
            Output[0] = InputPair[i];
        }
    }
}

void fillArrayRandom(int nTotalElements) {
    for (int i = 0; i < nTotalElements; i++) {
        int a = rand();
        int b = rand();

        float v = a * 100.0 + b;
        Input[i] = v;

        InputPair[i].key = v;
        InputPair[i].val = i;  // Store the corresponding index
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <nTotalElements> <k>\n", argv[0]);
        return 1;
    }

    // Convert the arguments from strings to integers
    int nTotalElements = atoi(argv[1]);
    int k = atoi(argv[2]);

    // Check the conditions
    if (nTotalElements <= 0) {
        printf("Error: nTotalElements must be a positive integer.\n");
        return 1;
    }

    if (k > nTotalElements || k <= 0) {
        printf("Error: k must be a positive integer and less than or equal to nTotalElements.\n");
        return 1;
    }

    // Allocate memory
    Input = (float *)malloc(nTotalElements * sizeof(float));
    InputPair = (pair_t *)malloc(nTotalElements * sizeof(pair_t));
    Output = (pair_t *)malloc(k * sizeof(pair_t));

    fillArrayRandom(nTotalElements);

    // Device arrays
    pair_t *d_InputPair;
    pair_t *d_Output;

    cudaMalloc((void **)&d_InputPair, nTotalElements * sizeof(pair_t));
    cudaMalloc((void **)&d_Output, k * sizeof(pair_t));

    cudaMemcpy(d_InputPair, InputPair, nTotalElements * sizeof(pair_t), cudaMemcpyHostToDevice);

    // Determine the launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (nTotalElements + threadsPerBlock - 1) / threadsPerBlock;

    clock_t start = clock();
    findKSmallestKernel<<<blocksPerGrid, threadsPerBlock>>>(d_InputPair, nTotalElements, d_Output, k);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    cudaMemcpy(Output, d_Output, k * sizeof(pair_t), cudaMemcpyDeviceToHost);

    printf("Time spent: %f\n", time_spent);
    verifyOutput(Input, Output, nTotalElements, k);

    // Cleanup
    free(Input);
    free(InputPair);
    free(Output);
    cudaFree(d_InputPair);
    cudaFree(d_Output);

    return 0;
}