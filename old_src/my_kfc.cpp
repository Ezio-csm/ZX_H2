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

void gemm(double*, double*, double*, int);
void gemv(double*, double*, double*, int);
void gevv(double*, double*, double*, int);
void vecscale(double*, const double&, int);
void vecadd(double*, double*, int);
void vecaddscaled(double*, double*, const double&, double*, int);
void matadd(double*, double*, int);
void matsub(double*, double*, int);
void mattrans(const double*, double*, int);
double vecdot(double*, double*, int);

void gemm_2(double*, double*, double*, double*, int);
void gemm4kernel(double*, double*, double*, int, int, int);
void gemm_3(const double*, const double*, double*, double*, int);

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
#define gemm(A, B, C, n) gemm_4(A, B, C, n)
#endif
    double* xt1t = (double*)malloc(num_vars * sizeof(double));
    double* Pt1t = (double*)malloc(num_vars * num_vars * sizeof(double));
    double* K = (double*)malloc(num_vars * sizeof(double));
    double* AT = (double*)malloc(num_vars * num_vars * sizeof(double));
    double* TT = (double*)malloc(num_vars * num_vars * sizeof(double));
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
    free(TT);
    free(tr);
    free(M);
    free(M2);
#ifdef Ezio
#undef gemm
#endif
}

// [n, n] x [n, n] -> [n, n]
void gemm(double* A, double* B, double* C, int n) {
    memset(C, 0, n * n * sizeof(double));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
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

void gemm4kernel(double* A, double* B, double* C, int row, int col, int n) {
    register double t0(0), t1(0), t2(0), t3(0), t4(0), t5(0), t6(0), t7(0),
        t8(0), t9(0), t10(0), t11(0), t12(0), t13(0), t14(0), t15(0);
    double *a0(A + row * n), *a1(a0 + n), *a2(a1 + n), *a3(a2 + n), *b0(B),
        *b1(b0 + n), *b2(b1 + n), *b3(b2 + n), *end = b0 + n;
    do {
        t0 += *(a0) * *(b0);
        t1 += *(a0) * *(b1);
        t2 += *(a0) * *(b2);
        t3 += *(a0++) * *(b3);
        t4 += *(a1) * *(b0);
        t5 += *(a1) * *(b1);
        t6 += *(a1) * *(b2);
        t7 += *(a1++) * *(b3);
        t8 += *(a2) * *(b0);
        t9 += *(a2) * *(b1);
        t10 += *(a2) * *(b2);
        t11 += *(a2++) * *(b3);
        t12 += *(a3) * *(b0++);
        t13 += *(a3) * *(b1++);
        t14 += *(a3) * *(b2++);
        t15 += *(a3++) * *(b3++);
    } while (b0 != end);
    C[row * n + col] = t0;
    C[row * n + col + 1] = t1;
    C[row * n + col + 2] = t2;
    C[row * n + col + 3] = t3;
    C[(row + 1) * n + col] = t4;
    C[(row + 1) * n + col + 1] = t5;
    C[(row + 1) * n + col + 2] = t6;
    C[(row + 1) * n + col + 3] = t7;
    C[(row + 2) * n + col] = t8;
    C[(row + 2) * n + col + 1] = t9;
    C[(row + 2) * n + col + 2] = t10;
    C[(row + 2) * n + col + 3] = t11;
    C[(row + 3) * n + col] = t12;
    C[(row + 3) * n + col + 1] = t13;
    C[(row + 3) * n + col + 2] = t14;
    C[(row + 3) * n + col + 3] = t15;
}

void gemm_2(double* A, double* B, double* C, double* tr, int n) {
    int tn = n - n % 4;
    for (int j = 0; j < tn; j += 4) {
        for (int i = 0; i < n; i++) {
            tr[0 * n + i] = B[i * n + j + 0];
            tr[1 * n + i] = B[i * n + j + 1];
            tr[2 * n + i] = B[i * n + j + 2];
            tr[3 * n + i] = B[i * n + j + 3];
        }
        for (int i = 0; i < tn; i += 4) {
            gemm4kernel(A, tr, C, i, j, n);
        }
    }
    for (int j = tn; j < n; j++)
        for (int i = 0; i < n; i++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
    for (int i = tn; i < n; i++)
        for (int j = 0; j < tn; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
}

void gemm_3(const double* A, const double* B, double* C, double* BT, int n) {
    mattrans(B, BT, n);
    int k;
    __m512d a, b;
    __m512d c = _mm512_setzero_pd();
    int tn = n - n % 8;
    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            for (k = 0; k < tn; k += 8) {
                a = _mm512_loadu_pd(A + i * n + k);
                b = _mm512_loadu_pd(BT + j * n + k);
                c = _mm512_fmadd_pd(a, b, c);
            }
            double sum = 0.0;
            for (; k < n; k++) {
                sum += A[i * n + k] * BT[j * n + k];
            }
            C[i * n + j] = _mm512_reduce_add_pd(c) + sum;
        }
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
        gemm_2(A, B, C, tr, n);
        gemm_3(A, B, C, TT, n);

    return 0;
}