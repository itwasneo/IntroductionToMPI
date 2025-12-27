#include "array.h"
#include "multiply.h"

#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned int const seed         = 1234;
static int const          dimensions[] = {128 * 1, 128 * 2, 128 * 4, 128 * 8};
static int const          n_dimensions = sizeof(dimensions) / sizeof(int);
static double const       epsilon      = 1e-10;

typedef void (*GEMM)(int const m, int const k, int const n, double const* const A,
                     double const* const B, double* const C);

static void populate_compatible_random_matrix_pairs(int const m, int const k, int const n,
                                                    int const seed, double* const A,
                                                    double* const B) {
    set_initilize_rand_seed(seed);

    initialize_2d_double_blocked_rand(A, m, k);
    initialize_2d_double_blocked_rand(B, k, n);
}

static void initialize_problem_matrices(int const m, int const k, int const n, double** const A,
                                        double** const B, double** const C) {
    *A = allocate_2d_double_blocked(m, k);
    *B = allocate_2d_double_blocked(k, n);
    *C = allocate_2d_double_blocked(m, n);
}

static void destroy_problem_matrices(double** const A, double** const B, double** const C) {
    *A = free_2d_double_blocked(*A);
    *C = free_2d_double_blocked(*C);
    *C = free_2d_double_blocked(*C);
}

static bool test_muptiply(int const m, int const k, int const n, GEMM gemm, double const epsilon,
                          unsigned int const seed) {
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);

    gemm(m, k, n, A, B, C);
    bool result_is_correct = is_product(m, k, n, A, B, C, epsilon);

    destroy_problem_matrices(&A, &B, &C);

    return result_is_correct;
}

// Implement a function "parallel_gemm" of type GEMM, that implements the
// matrix multiplication operation.
//
void parallel_gemm(int const m, int const k, int const n, double const* const A,
                   double const* const B, double* const C) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // distributing columns of B accross processes
    int const cols_per_process = n / size;
    int const local_col_start  = rank * cols_per_process;
    int const local_col_count  = cols_per_process;

    // allocate local portion of B (k * local_col_count)
    double* B_local = allocate_2d_double_blocked(k, local_col_count);

    // scatter B: copy the columns assigned to this rank
    for (int j = 0; j < local_col_count; j++) {
        int const global_col_index = local_col_start + j;
        for (int i = 0; i < k; i++) {
            B_local[i + j * k] = B[i + global_col_index * k];
        }
    }

    // computing directly into C at the correct positions
    for (int j = 0; j < local_col_count; j++) {
        int const global_col_index = local_col_start + j;
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += A[i + l * m] * B_local[l + j * k];
            }
            C[i + global_col_index * m] = sum;
        }
    }

    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs     = (int*)malloc(size * sizeof(int));

    for (int p = 0; p < size; p++) {
        recvcounts[p] = m * cols_per_process;
        displs[p]     = p * m * cols_per_process;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, C, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    B_local = free_2d_double_blocked(B_local);
    free(recvcounts);
    free(displs);
}
//
// Then set "tested_gemm" to the address of your funtion
// GEMM const tested_gemm = &multiply_matrices;
GEMM const tested_gemm = &parallel_gemm;

static bool generate_square_matrix_dimension(int* const m, int* const k, int* const n) {
    int const  max_dim = n_dimensions;
    static int dim     = 0;

    if (dim >= max_dim) {
        return false;
    }

    *m = dimensions[dim];
    *k = dimensions[dim];
    *n = dimensions[dim];

    dim++;

    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    bool all_test_pass = true;

    int m = 0;
    int k = 0;
    int n = 0;

    while (generate_square_matrix_dimension(&m, &k, &n)) {
        bool const test_pass = test_muptiply(m, k, n, tested_gemm, epsilon, seed);
        if (!test_pass) {
            printf("Multiplication failed for: m=%d, k=%d, n=%d\n", m, k, n);
            all_test_pass = false;
        }
    }

    MPI_Finalize();

    if (!all_test_pass) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
