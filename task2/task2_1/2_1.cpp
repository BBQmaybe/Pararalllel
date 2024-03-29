#include <iostream>
#include <memory>
#include <omp.h>
#include <inttypes.h>

int num_pot = 40;

void matrix_vector_product_omp(double* a, double* b, double* c, int m, int n)
{
#pragma omp parallel num_threads(num_pot)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++) {
                c[i] += a[i * n + j] * b[j];
            }
        }
    }
}

void parallel_init(double* a, double* b, double* c, int m, int n) {
#pragma omp parallel num_threads(num_pot)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i < ub; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
        }
        for (int j = 0; j < n; j++) {
            b[j] = j;
        }
    }
}

void run_parallel(int m, int n)
{
    std::unique_ptr<double[]> a(new double[m * n]);
    std::unique_ptr<double[]> b(new double[n]);
    std::unique_ptr<double[]> c(new double[m]);

    double t1 = omp_get_wtime();

    parallel_init(a.get(), b.get(), c.get(), m, n);

    t1 = omp_get_wtime() - t1;
    printf("Elapsed time (parallel): %.2f sec.\n", t1);
    double t;
    t = omp_get_wtime();
    matrix_vector_product_omp(a.get(), b.get(), c.get(), m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time (parallel): %.2f sec.\n", t);
    printf("%.2f           1", 2.02 / t);
}

int main(int argc, char** argv)
{
    int m = 20000, n = 20000;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    run_parallel(m, n);

    return 0;
}
