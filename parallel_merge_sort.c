#include <errno.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/****************************************************************************************************
Parallel merge sort:

 The array is split into sub-arrays until each sub-array only holds one element. This way, all 
 sub-arrays are sorted (since they only have 1 element). Then, all sub-array are merged with their 
 "neighbouring" sub-array, and so on, until all are merged to one array.

 Therefore, when starting the merge operation we have a very large number of very small arrays.
 Merging them requires very little CPU time. That's why it makes sense to merge them sequentially 
 up to a certain array length (2000 - found by trying out different numbers), in order to avoid 
 the overheads of parallelism. Once the arrays have reached this certain size, the operation takes 
 much more time and it becomes reasonable to parallelize it.

 ****************************************************************************************************/


/**
 * @brief Parallel function to merge to sub-arrays (arr[left..mid] and arr[mid+1..right])
 * @param: arr
 * @param: left = index of the left end
 * @param: left = index of the middle of the array
 * @param: left = index of the right end
 */
void merge_parallel(int arr[], int left, int mid, int right)
{

    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // temp arrays
    int32_t *L;
    L = (int32_t *) calloc(n1, sizeof(int32_t));

    int32_t *R;
    R = (int32_t *) calloc(n2, sizeof(int32_t));

    // Copy data to temp arrays L[] and R[]
#pragma omp parallel shared(i, j, k, n1, n2)
    {
#pragma omp for
        for (i = 0; i < n1; i++)
            L[i] = arr[left + i];

#pragma omp for
        for (j = 0; j < n2; j++)
            R[j] = arr[mid + 1 + j];
    }

    // Merge the temp arrays back into arr[l..r]

    i = 0;      // Initial index of first subarray
    j = 0;      // Initial index of second subarray
    k = left;   // Initial index of merged subarray

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        } else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    free(L);
    free(R);
}

/**
 * @brief Non-parallel function to merge to sub-arrays (arr[left..mid] and arr[mid+1..right])
 * @param: arr
 * @param: left = index of the left end
 * @param: left = index of the middle of the array
 * @param: left = index of the right end
 */
void merge_sequential(int arr[], int left, int mid, int right)
{

    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // temp arrays
    int32_t L[n1];
    int32_t R[n2];

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // Merge the temp arrays back into arr[l..r]

    i = 0;      // Initial index of first subarray
    j = 0;      // Initial index of second subarray
    k = left;   // Initial index of merged subarray

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        } else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/**
 * @brief Recursive and sequential merge sort algorithm.
 * @param: arr = array to sort
 * @param: left = index of the left end
 * @param: right = index of the right end
 */
void merge_sort_recursive(int32_t arr[], int left, int right)
{

    if (left < right)
    {
        int size = right - left;
        int mid = left + size / 2;

        if (size < 2000)
        {
            merge_sort_recursive(arr, left, mid);
            merge_sort_recursive(arr, mid + 1, right);

            merge_sequential(arr, left, mid, right);
        } else
        {
            // Splitting in two tasks. Taskwait will then wait for both tasks to finish.
#pragma omp task
            merge_sort_recursive(arr, left, mid);

#pragma omp task
            merge_sort_recursive(arr, mid + 1, right);

#pragma omp taskwait
            merge_parallel(arr, left, mid, right);
        }
    }
}

/**
 * @brief Function for checking whether or not an array's elements are ordered.
 * (https://www.geeksforgeeks.org/program-check-array-sorted-not-iterative-recursive/)
 * @param: arr
 * @param: n = length of the array
 */
int is_array_sorted(int arr[], int n)
{
    if (n == 1 || n == 0)
        return 1;

    // Unsorted pair found (Equal values allowed)
    if (arr[n - 1] < arr[n - 2])
    {
        return 0;
    }

    // recursive call
    return is_array_sorted(arr, n - 1);
}


int main(int argc, char **argv)
{

    /****************** handle input ******************/
    if (argc != 2)
    {
        fprintf(stderr, "Error: usage: %s <n>\n", argv[0]);
        return EXIT_FAILURE;
    }
    errno = 0;
    char *str = argv[1];
    char *endptr;
    long n = strtol(str, &endptr, 0);
    if (errno != 0)
    {
        perror("strtol");
        return EXIT_FAILURE;
    }
    if (endptr == str)
    {
        fprintf(stderr, "Error: no digits were found!\n");
        return EXIT_FAILURE;
    }
    if (n < 0)
    {
        fprintf(stderr, "Error: matrix size must not be negative!\n");
        return EXIT_FAILURE;
    }
    /******************************************************************************/

    /******** allocation of the array and filling it with random numbers **********/

    int32_t *arr;
    arr = (int32_t *) calloc(n, sizeof(int32_t));

    if (arr == NULL)
    {
        printf("MALLOC ERROR\n");
        return EXIT_FAILURE;
    }

    // fill array with pseudo random numbers
    srand(0);

    double start_time;
    double end_time;

#pragma omp parallel // parallelize the filling of the array, with an own seed for each thread
    {
        unsigned int my_seed = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            arr[i] = rand_r(&my_seed) / 10000000;
        }
        /******************************************************************************/

    }
    // print array
    printf("Before: \n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");

    start_time = omp_get_wtime();

    merge_sort_recursive(arr, 0, n - 1);

    end_time = omp_get_wtime();

    printf("After: \n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");

    assert(is_array_sorted(arr, n));

    printf("time: %2.2f seconds\n", end_time - start_time);
    return EXIT_SUCCESS;
}

