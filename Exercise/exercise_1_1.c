#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <math.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "array.h"

static unsigned int const seed = 1234;
static int const dimensions[] = {128*1, 128*2, 128*4, 128*8};
static int const n_dimensions = sizeof(dimensions)/sizeof(int);
static double const epsilon = 1e-10;

typedef void (*BLAS_DAXPY)(
    int const n, double const alpha, double* const x, int const incx, double* const y, int const incy
);

typedef double (*BLAS_DDOT)(
    int const n, double* const x, int const incx, double* const y, int const incy
);

static void populate_random_DDOT_operants(
    int const n,
    int const seed,
    double* const x, double* const y)
{
    set_initilize_rand_seed(seed);

    initialize_1d_double_rand(x, n);
    initialize_1d_double_rand(y, n);
}

static void populate_random_DAXPY_operants(
    int const n,
    int const seed,
    double* const alpha,
    double* const x, double* const y)
{
    set_initilize_rand_seed(seed);

    *alpha = get_double_rand();
    initialize_1d_double_rand(x, n);
    initialize_1d_double_rand(y, n);
}

static void initialize_problem_operants(
    int const n,
    double** const x, double** const y)
{
    *x = allocate_1d_double(n);
    *y = allocate_1d_double(n);
}

static void destroy_problem_operants(double** const x, double** const y)
{
    *x = free_1d_double(*x);
    *y = free_1d_double(*y);
}

static bool test_DAXPY(int const n, BLAS_DAXPY axpy, double const epsilon, unsigned int const seed)
{
    double* x = NULL;
    double* y = NULL;
    double alpha = 0;
    initialize_problem_operants(n, &x, &y);
    populate_random_DAXPY_operants(n, seed, &alpha, x, y);

    double* y_test = allocate_1d_double(n);
    for (int i = 0; i < n; i++) {
        y_test[i] = y[i];
    }

    axpy(n, alpha, x, 1, y, 1);
    cblas_daxpy(n, alpha, x, 1, y_test, 1);
    double err = 0.0;
    for (int i = 0; i < n; i++) {
        double const diff = y[i] - y_test[i];
        err += diff*diff;
    }
    err = sqrt(err);
    
    bool result_is_correct = err < epsilon;

    y_test = free_1d_double(y_test);
    destroy_problem_operants(&x, &y);

    return result_is_correct;
}

static bool test_DDOT(int const n, BLAS_DDOT dot, double const epsilon, unsigned int const seed)
{
    double* x = NULL;
    double* y = NULL;
    initialize_problem_operants(n, &x, &y);
    populate_random_DDOT_operants(n, seed, x, y);

    double const d = dot(n, x, 1, y, 1);
    double const d_test = cblas_ddot(n, x, 1, y, 1);
    
    double const diff = d - d_test;
    double const err = sqrt(diff*diff);
    
    bool result_is_correct = err < epsilon;

    destroy_problem_operants(&x, &y);

    return result_is_correct;
}

// In the implementation of functions "DAXPY" and "DDOT" replace the call to
// the corresponding BLAS function with your own implementation.
// y = alpha * x + y
void DAXPY(int const n, double const alpha, double* const x, int const incx, double* const y, int const incy)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int const base_count = n / size;
    int const remainder = n % size;

    // MAGICAL DISTRIBUTION OF WORK
    // -- Number of elements for current rank
    int const local_count = base_count + (rank < remainder ? 1: 0);
    // -- Global starting index for current rank
    int const global_start = rank * base_count + (rank < remainder ? rank : remainder);

    // Local y = alpha * x + y Calculation
    for (int i = 0; i < local_count; i++) {
        int const global_idx = global_start + i;
        y[global_idx * incy] = alpha * x[global_idx * incx] + y[global_idx * incy];
    }

    // In order to gather different result chunks from different ranks, I will use Allgatherv
    // which is similar to Allgather but "varied" message sizes
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0); 
        displs[i] = i * base_count + (i < remainder ? i : remainder);
    }

    // Here we can use MPI_IN_PLACE option, since the test cases are using **incy == 1**. (NO STRIDE)
    // Each process has already written to its portion of y (This is more performant obviously)
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, y, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);
}

double DDOT(int const n, double* const x, int const incx, double* const y, int const incy)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int const base_count = n / size;
    int const remainder = n % size;

    // Work Distribution is same as DAXPY above -----^
    int const local_count = base_count + (rank < remainder ? 1: 0);
    int const global_start = rank * base_count + (rank < remainder ? rank : remainder);

    double local_dot = 0.0;
    for (int i = 0; i < local_count; i++) {
        int const global_index = global_start + i;
        local_dot += x[global_index * incx] * y[global_index * incy];
    }

    double global_dot = 0.0;

    // To reduce all the local_dots from all the ranks, I will use MPI_Allreduce
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_dot;
}

static bool generate_operand_dimension(int* const n)
{
    int const max_dim = n_dimensions;
    static int dim = 0;

    if (dim >= max_dim) {
        return false;
    }

    *n = dimensions[dim];

    dim++;

    return true;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    bool all_tests_pass = true;

    int n = 0;

    while (generate_operand_dimension(&n)) {
        bool const test_DAXPY_pass = test_DAXPY(n, DAXPY, epsilon, seed);
        if (!test_DAXPY_pass) {
            printf("DAXPY failed for: n=%d\n", n);
            all_tests_pass = false;
        }
        bool const test_DDOT_pass = test_DDOT(n, DDOT, epsilon, seed);
        if (!test_DDOT_pass) {
            printf("DDOT failed for: n=%d\n", n);
            all_tests_pass = false;
        }
    }

    MPI_Finalize();

    if (!all_tests_pass) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
