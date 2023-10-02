#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    // Read first argument, it is an int
    int nElements = atoi(argv[1]);

    double sum = 0.0;
    double answer = 0.0;

    // Check if the number of supplied arguments matches nElements
    if (argc != nElements + 2) {
        fprintf(stderr, "Error: Number of elements does not match the supplied value.\n");
        return 1;
    }

    // Sum all elements which are passed as arguments
    for (int i = 2; i < nElements + 2; i++)
    {  
        sum += atof(argv[i]);
    }

    // Calculate average
    answer = sum / nElements;

    // Print average
    printf("%lf\n", answer);

    return 0;
}