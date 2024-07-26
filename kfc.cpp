// g++ -mavx512f -fopenmp -shared -O3 -std=c++11 -DEzio kfc.cpp -ldl -o kfc.so -lopenblas 
// g++ -mavx512f -fopenmp -O3 -std=c++11 -DEzio kfc.cpp -o kfc -lopenblas 

#include <cblas.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

void gemv(double*, double*, double*, int);
void gevv(double*, double*, double*, int);
void vecscale(double*, const double&, int);
void vecadd(double*, double*, int);
void vecaddscaled(double*, double*, const double&, double*, int);
void matadd(double*, double*, int);
void matsub(double*, double*, int);
void mattrans(const double*, double*, int);
double vecdot(double*, double*, int);

void gemm(const double* A, const double* B, double* C, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n,
                B, n, 0.0, C, n);
}

// Q [num_vars, num_vars]
// C [1, num_vars]
// A [num_vars, num_vars]
// R scalar

// P [num_steps, num_vars, num_vars]
// x [num_steps, num_vars]
// y [num_steps]
// tmp [num_steps, num_vars]

// kfc means Kalman Filter written in C. It may works more efficiently on
// Thursday.
extern "C" void kfc(double* Q,
                    double* C,
                    double* A,
                    double R,
                    double* P,
                    double* x,
                    double* y,
                    double* tmp,
                    int num_steps,
                    int num_vars) {
#ifdef Ezio
#define gemm(A, B, C, n) gemm(A, B, C, n)
#endif
    double* xt1t = (double*)malloc(num_vars * sizeof(double));
    double* Pt1t = (double*)malloc(num_vars * num_vars * sizeof(double));
    double* K = (double*)malloc(num_vars * sizeof(double));
    double* AT = (double*)malloc(num_vars * num_vars * sizeof(double));
    double* tr = (double*)malloc(4 * num_vars * sizeof(double));

    mattrans(A, AT, num_vars);

    double dt;
    double* M = (double*)malloc(num_vars * num_vars * sizeof(double));
    double* M2 = (double*)malloc(num_vars * num_vars * sizeof(double));
    for (int t = 1; t < num_steps; t++) {
        // xt1t = A @ x[t - 1] + tmp[t]
        // [num_vars, 1]
        gemv(A, x + num_vars * (t - 1), xt1t, num_vars);
        vecadd(xt1t, tmp + num_vars * t, num_vars);

        // Pt1t = A @ P[t - 1] @ A.T + Q
        // [num_vars, num_vars]
        gemm(A, P + num_vars * num_vars * (t - 1), M, num_vars);
        gemm(M, AT, Pt1t, num_vars);
        matadd(Pt1t, Q, num_vars);

        // K = Pt1t @ C.T @ (1.0/(C @ Pt1t @ C.T + R))
        // [num_vars, 1]
        gemv(Pt1t, C, K, num_vars);
        dt = 1.0 / (R + vecdot(C, K, num_vars));
        vecscale(K, dt, num_vars);

        // x[t] = xt1t + K @ (y[t] - C @ xt1t)
        dt = y[t] - vecdot(C, xt1t, num_vars);
        vecaddscaled(xt1t, K, dt, x + num_vars * t, num_vars);

        // P[t] = Pt1t - K @ C @ Pt1t
        gevv(K, C, M, num_vars);
        gemm(M, Pt1t, M2, num_vars);
        matsub(Pt1t, M2, num_vars);
        memcpy(P + num_vars * num_vars * t, Pt1t,
               num_vars * num_vars * sizeof(double));
    }
    free(xt1t);
    free(Pt1t);
    free(K);
    free(AT);
    free(tr);
    free(M);
    free(M2);
#ifdef Ezio
#undef gemm
#endif
}

// [n, n] x [n, 1] -> [n, 1]
void gemv(double* A, double* x, double* y, int n) {
    memset(y, 0, n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            y[i] += A[i * n + j] * x[j];
}

// [n, 1] x [1, n] -> [n, n]
void gevv(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i] * B[j];
}

void vecscale(double* A, const double& x, int n) {
    for (int i = 0; i < n; i++)
        A[i] *= x;
}

void vecadd(double* x, double* y, int n) {
    for (int i = 0; i < n; i++)
        x[i] += y[i];
}

void vecaddscaled(double* x, double* y, const double& a, double* z, int n) {
    for (int i = 0; i < n; i++)
        z[i] = x[i] + a * y[i];
}

void matadd(double* A, double* B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] += B[i * n + j];
}

void matsub(double* A, double* B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] -= B[i * n + j];
}

void mattrans(const double* A, double* AT, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AT[i * n + j] = A[j * n + i];
}

double vecdot(double* x, double* y, int n) {
    double res = 0;
    for (int i = 0; i < n; i++)
        res += x[i] * y[i];
    return res;
}

int main() {
    const int n = 50;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    double* tr = (double*)malloc(4 * n * sizeof(double));
    double* TT = (double*)malloc(n * n * sizeof(double));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n * n; i++) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    for (int i = 0; i < 100000; i++)
        gemm(A, B, C, n);

    return 0;
}