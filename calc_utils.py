import ctypes
import numpy as np

# Q [num_vars, num_vars]
# C [1, num_vars]
# A [num_vars, num_vars]
# R scalar

# P [num_steps, num_vars, num_vars]
# x [num_steps, num_vars]
# y [num_steps]
# tmp [num_steps, num_vars]

def kalman_filter_proc(Q, C, A, R, P, x, y, tmp, num_steps):
    for t in range(1, num_steps):
        xt1t = A @ x[t - 1] + tmp[t] # [num_vars, 1]
        Pt1t = A @ P[t - 1] @ A.T + Q # [num_vars, num_vars]
        K = Pt1t @ C.T @ (1.0/(C @ Pt1t @ C.T + R)) # [num_vars, 1]
        x[t] = xt1t + K @ (y[t] - C @ xt1t) 
        P[t] = Pt1t - K @ C @ Pt1t


lib = ctypes.cdll.LoadLibrary("./kfc.so")
def kalman_filter_proc_c(Q, C, A, R, P, x, y, tmp, num_steps):
    num_vars = Q.shape[0]
    lib.kfc(Q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ctypes.c_double(R), 
            P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            num_steps, num_vars)

if __name__ == '__main__':
    num_var = 15
    num_steps = 80000
    y = np.random.randn(num_steps)
    w = np.zeros((len(y), num_var))
    omega = np.full((num_var,), 1e-5)
    sigma = 1e-20
    omegaMeanshift = 0
    b = np.zeros((num_var, 3))
    c = np.zeros((num_var, 1))

    num_steps = w.shape[0]
    num_vars = b.shape[0] * 3 + 1
    x = np.zeros((num_steps, num_vars))
    P = np.zeros((num_steps, num_vars, num_vars))
    A = np.zeros((num_vars, num_vars))
    C = np.zeros((1, num_vars))
    intercept = np.zeros((num_vars,))
    beta = np.zeros((num_vars,))
    Q = np.zeros((num_vars, num_vars))
    R = sigma
    ws = np.zeros((num_steps, num_vars))

    for i in range(b.shape[0]):
        A[3 * i + 2, 3 * i + 1] = 1
        A[3 * i + 1, 3 * i] = 1
        A[3 * i, 3 * i] = b[i][0]
        A[3 * i, 3 * i + 1] = b[i][1]
        C[0, 3 * i] = 1
        intercept[3 * i] = c[i]
        beta[3 * i] = b[i, 2]
        ws[:, 3 * i] = w[:, i]
        Q[3 * i, 3 * i] = omega[i]
    Q[-1, -1] = omegaMeanshift
    C[0, -1] = 1
    A[-1, -1] = 1
    x[0] = np.full((num_vars,), y[0] / b.shape[0])
    x[0, -1] = 0

    tmp = beta * ws + intercept
    iter_num = 10

    import time
    start = time.time()
    for t in range(iter_num):
        print(f'iter: {t}')
        kalman_filter_proc(Q, C, A, R, P, x, y, tmp, num_steps)
    time1 = time.time() - start

    start = time.time()
    for t in range(iter_num):
        print(f'iter: {t}')
        kalman_filter_proc_c(Q, C, A, R, P, x, y, tmp, num_steps)
    time2 = time.time() - start

    print(f'python time: {time1}')
    print(f'c time: {time2}')