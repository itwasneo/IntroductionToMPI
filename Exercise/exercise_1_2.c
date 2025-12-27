#include "mpi.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "array.h"

static unsigned int const seed         = 1234;
static int const          dimensions[] = {32 * 1, 32 * 8, 32 * 64, 32 * 512};
static int const          n_dimensions = sizeof(dimensions) / sizeof(int);
static double const       epsilon      = 1e-10;

#ifdef ACCELERATE_NEW_LAPACK // To handle deprecated lib names in APPLE
typedef enum CBLAS_ORDER     CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
#endif

typedef void (*BLAS_DGEMV)(int const m, int const n, double const alpha, double const* const A,
                           int const ldA, double const* const x, int const incx, double const beta,
                           double* const y, int const incy);

static void populate_random_operants(int const m, int const n, int const seed, double* const A,
                                     double* const x, double* const y, double* const alpha,
                                     double* const beta) {
    set_initilize_rand_seed(seed);

    initialize_1d_double_rand(x, n);
    initialize_1d_double_rand(y, m);
    initialize_2d_double_blocked_rand(A, m, n);

    *alpha = get_double_rand();
    *beta  = get_double_rand();
}

static void initialize_problem_operants(int const m, int const n, double** const A,
                                        double** const x, double** const y) {
    *A = allocate_2d_double_blocked(m, n);
    *x = allocate_1d_double(n);
    *y = allocate_1d_double(m);
}

static void destroy_problem_operants(double** const A, double** const x, double** const y) {
    *A = free_2d_double_blocked(*A);
    *x = free_1d_double(*x);
    *y = free_1d_double(*y);
}

static bool test_DGEMV(int const m, int const n, BLAS_DGEMV dgemv, double const epsilon,
                       unsigned int const seed, double* const duration) {
    double* A     = NULL;
    double* x     = NULL;
    double* y     = NULL;
    double  alpha = 0.0;
    double  beta  = 0.0;

    initialize_problem_operants(m, n, &A, &x, &y);
    populate_random_operants(m, n, seed, A, x, y, &alpha, &beta);
    double* y_test = allocate_1d_double(m);
    for (int i = 0; i < m; i++) {
        y_test[i] = y[i];
    }

    CBLAS_LAYOUT const    layout = CblasColMajor;
    CBLAS_TRANSPOSE const transA = CblasNoTrans;
    int const             ldA    = m;
    int const             incx   = 1;
    int const             incy   = 1;

    cblas_dgemv(layout, transA, m, n, alpha, A, ldA, x, incx, beta, y_test, incy);

    // Time the execution of dgemv
    clock_t const start = clock();
    dgemv(m, n, alpha, A, ldA, x, incx, beta, y, incy);
    clock_t const end = clock();

    double err = 0.0;
    for (int i = 0; i < m; i++) {
        double const diff = y[i] - y_test[i];
        err += diff * diff;
    }
    err = sqrt(err);

    bool result_is_correct = err < epsilon;

    y_test = free_1d_double(y_test);
    destroy_problem_operants(&A, &x, &y);

    *duration = ((double)(end - start)) / CLOCKS_PER_SEC;

    return result_is_correct;
}

// In the implementation of functions "DGEMV" and "rowwise_DGEMV", replace the
//  call to the BLAS function with your own implementation.
//  iterate over columns, accumulate into y
void DGEMV(int const m, int const n, double const alpha, double const* const A, int const ldA,
           double const* const x, int const incx, double const beta, double* const y,
           int const incy) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // storing the original values
    double* y_original = allocate_1d_double(m);
    for (int i = 0; i < m; i++) {
        y_original[i] = y[i * incy];
    }

    // distribute "columns" across ranks
    int const base_count = n / size;
    int const remainder  = n % size;

    int const local_col_count = base_count + (rank < remainder ? 1 : 0);
    int const local_col_start = rank * base_count + (rank < remainder ? rank : remainder);

    double* local_y = allocate_1d_double(m);
    for (int i = 0; i < m; i++) {
        local_y[i] = 0.0;
    }

    // Column-wise computation: for each column j, compute alpha * A(:,j) * x[j]
    for (int j = 0; j < local_col_count; j++) {
        int const    global_col = local_col_start + j;
        double const x_val      = x[global_col * incx];

        // Add alpha * x[j] * A(:,j) to local_y
        for (int i = 0; i < m; i++) {
            // Column-major: A[i,j] is at A[i + j*ldA]
            local_y[i] += alpha * x_val * A[i + global_col * ldA];
        }
    }

    // reduce
    double* global_y = allocate_1d_double(m);
    MPI_Allreduce(local_y, global_y, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < m; i++) {
        y[i * incy] = global_y[i] + beta * y_original[i];
    }

    y_original = free_1d_double(y_original);
    local_y    = free_1d_double(local_y);
    global_y   = free_1d_double(global_y);
}

void rowwise_DGEMV(int const m, int const n, double const alpha, double const* const A,
                   int const ldA, double const* const x, int const incx, double const beta,
                   double* const y, int const incy) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // distribute "rows" across ranks
    int const base_count = m / size;
    int const remainder  = m % size;

    int const local_row_count  = base_count + (rank < remainder ? 1 : 0);
    int const global_row_start = rank * base_count + (rank < remainder ? rank : remainder);

    for (int i = 0; i < local_row_count; i++) {
        int const global_row = global_row_start + i;

        // dot product of row i with x
        double dot = 0.0;
        for (int j = 0; j < n; j++) {
            dot += A[global_row + j * ldA] * x[j * incx];
        }

        y[global_row * incy] = alpha * dot + beta * y[global_row * incy];
    }

    // as in DDOT
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs     = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i]     = i * base_count + (i < remainder ? i : remainder);
    }

    // again I'm using Allgatherv to collect and sum all the dot products
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, y, sendcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    free(sendcounts);
    free(displs);
}

static bool generate_operand_dimensions(int* const m, int* const n) {
    int const  max_dim = n_dimensions;
    static int m_dim   = 0;
    static int n_dim   = 0;

    if (n_dim >= max_dim) {
        return false;
    }

    *m = dimensions[m_dim];
    *n = dimensions[n_dim];

    m_dim++;
    if (m_dim >= max_dim) {
        m_dim = 0;
        n_dim++;
    }

    return true;
}

int main(int argc, char* argv[]) {
    bool all_test_pass = true;

    MPI_Init(&argc, &argv);
    int n = 0;
    int m = 0;

    while (generate_operand_dimensions(&m, &n)) {
        double     columnwise_duration = 0.0;
        bool const test_DGEMV_pass = test_DGEMV(m, n, DGEMV, epsilon, seed, &columnwise_duration);
        if (!test_DGEMV_pass) {
            fprintf(stderr, "DGENV failed for: m=%d, n=%d\n", m, n);
            all_test_pass = false;
        }
        double     rowwise_duration = 0.0;
        bool const test_rowwise_DGEMV_pass =
            test_DGEMV(m, n, rowwise_DGEMV, epsilon, seed, &rowwise_duration);
        if (!test_rowwise_DGEMV_pass) {
            fprintf(stderr, "rowwise_DGEMV failed for: m=%d, n=%d\n", m, n);
            all_test_pass = false;
        }
        printf("Duration of case m=%d, n=%d: columnwise=%lf, rowwise=%lf\n", m, n,
               columnwise_duration, rowwise_duration);
    }

    MPI_Finalize();

    if (!all_test_pass) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
