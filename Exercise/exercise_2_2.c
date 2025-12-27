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
    *B = free_2d_double_blocked(*B);
    *C = free_2d_double_blocked(*C);
}

static bool test_muptiply(int const m, int const k, int const n, GEMM gemm, double const epsilon,
                          unsigned int const seed) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);

    gemm(m, k, n, A, B, C);

    // Broadcast result from master to all workers for verification
    MPI_Bcast(C, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    bool result_is_correct = is_product(m, k, n, A, B, C, epsilon);

    destroy_problem_matrices(&A, &B, &C);

    return result_is_correct;
}

// MPI message tags
#define TAG_WORK_REQUEST 1
#define TAG_WORK_INDEX 2
#define TAG_WORK_DATA 3
#define TAG_RESULT_INDEX 4
#define TAG_RESULT_DATA 5
#define NO_MORE_WORK -1

// Dynamic scattering matrix multiplication using master-worker pattern
void parallel_gemm(int const m, int const k, int const n, double const* const A,
                   double const* const B, double* const C) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // === MASTER ===
        int next_column        = 0;
        int results_received   = 0;
        int workers_terminated = 0;

        // Initialize C to zero
        for (int i = 0; i < m * n; i++)
            C[i] = 0.0;

        // Process requests until all results received and all workers terminated
        while (results_received < n || workers_terminated < size - 1) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_WORK_REQUEST) {
                // Worker requesting work
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                if (next_column < n) {
                    // Send work: column index and data
                    MPI_Send(&next_column, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_INDEX,
                             MPI_COMM_WORLD);
                    MPI_Send(&B[next_column * k], k, MPI_DOUBLE, status.MPI_SOURCE, TAG_WORK_DATA,
                             MPI_COMM_WORLD);
                    next_column++;
                } else {
                    // No more work - send termination signal
                    int terminate = NO_MORE_WORK;
                    MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK_INDEX,
                             MPI_COMM_WORLD);
                    workers_terminated++;
                }
            } else if (status.MPI_TAG == TAG_RESULT_INDEX) {
                // Worker sending result
                int col_idx;
                MPI_Recv(&col_idx, 1, MPI_INT, status.MPI_SOURCE, TAG_RESULT_INDEX, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Recv(&C[col_idx * m], m, MPI_DOUBLE, status.MPI_SOURCE, TAG_RESULT_DATA,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                results_received++;
            }
        }
    } else {
        // === WORKER ===
        double* B_col = (double*)malloc(k * sizeof(double));
        double* C_col = (double*)malloc(m * sizeof(double));

        while (true) {
            // Request work
            MPI_Send(&rank, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD);

            // Receive column index
            int col_idx;
            MPI_Recv(&col_idx, 1, MPI_INT, 0, TAG_WORK_INDEX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Check for termination
            if (col_idx == NO_MORE_WORK)
                break;

            // Receive column data
            MPI_Recv(B_col, k, MPI_DOUBLE, 0, TAG_WORK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Compute C(:, col_idx) = A * B(:, col_idx)
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * m] * B_col[l];
                }
                C_col[i] = sum;
            }

            // Send result back to master
            MPI_Send(&col_idx, 1, MPI_INT, 0, TAG_RESULT_INDEX, MPI_COMM_WORLD);
            MPI_Send(C_col, m, MPI_DOUBLE, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
        }

        free(B_col);
        free(C_col);
    }
}

GEMM const tested_gemm = &parallel_gemm;
// GEMM const tested_gemm = &multiply_matrices;

static bool generate_square_matrix_dimension(int* const m, int* const k, int* const n) {
    static int dim = 0;
    if (dim >= n_dimensions)
        return false;

    *m = *k = *n = dimensions[dim];
    dim++;
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool all_test_pass = true;
    int  m = 0, k = 0, n = 0;

    while (generate_square_matrix_dimension(&m, &k, &n)) {
        bool test_pass = test_muptiply(m, k, n, tested_gemm, epsilon, seed);

        // Synchronize test result across all processes
        int local_pass = test_pass ? 1 : 0;
        int global_pass;
        MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (global_pass == 0) {
            if (rank == 0)
                printf("Multiplication failed for: m=%d, k=%d, n=%d\n", m, k, n);
            all_test_pass = false;
        }
    }

    MPI_Finalize();
    return all_test_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
