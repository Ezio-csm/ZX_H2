import numpy as np
import math
from multiprocessing import Pool

USE_MULTIPROC = True
PROC_NUM = 4
def rmse_process(t, eval_step, x_s, y, A, C, tmp):
    for s in range(t - eval_step, t):
        x_s = A @ x_s + tmp[s+1]
    return (y[t] - C @ x_s)[0] ** 2

class StatespacewithShift:
    def __init__(self, num_var, decay=1, output_Qn=False, verbose=False, eval_step=10):
        self.b = np.zeros((num_var, 3))
        self.c = np.zeros((num_var, 1))
        self.x = None
        self.omega = np.full((num_var,), 1e-5)
        self.sigma = 1e-20
        self.omegaMeanshift = 0
        self.decay = decay
        self.variance = np.zeros((num_var, 4, 4))
        self.outputQn = output_Qn
        self.verbose = verbose
        self.eval_step = eval_step

        if self.verbose and not self.outputQn:
            raise ValueError('To output Qn, must set outputQn=True')

    def fit(self, y, w, sample_T, max_iter):
        cnt = 0
        negative_Q_list = []
        for iter in range(max_iter):
            x, P = self.kalman_filter(y, w)
            self.x = x
            if self.outputQn and cnt % 50 == 0:
                negative_logQ = self.negative_Q(x, P, y, w, sample_T)
                estimated_RMSE = self.rmse(x, P, y, w, sample_T)
                if self.verbose:
                    print(f"{iter}-th iteration: Negative Qn[{negative_logQ}], RMSE(step={self.eval_step}): [{estimated_RMSE}]")
                negative_Q_list.append(negative_logQ)
            else:
                print(f"{iter}-th iteration")
            self.update(x, y, w, P, sample_T)
            cnt += 1
        if self.outputQn:
            return negative_Q_list

    def rmse(self, x, P, y, w, sample_T):
        loss = 0
        b_shape_0 = self.b.shape[0]
        C = np.zeros((1, b_shape_0 * 3 + 1))
        A = np.zeros((b_shape_0 * 3 + 1, b_shape_0 * 3 + 1))
        intercept = np.zeros((b_shape_0 * 3 + 1,))
        beta = np.zeros((b_shape_0 * 3 + 1,))
        ws = np.zeros((w.shape[0], b_shape_0 * 3 + 1))

        for i in range(b_shape_0):
            A[3 * i + 2, 3 * i + 1] = 1
            A[3 * i + 1, 3 * i] = 1
            A[3 * i, 3 * i] = self.b[i][0]
            A[3 * i, 3 * i + 1] = self.b[i][1]
            intercept[3 * i] = self.c[i]
            C[0, 3 * i] = 1
            beta[3 * i] = self.b[i, 2]
            ws[:, 3 * i] = w[:, i]
        C[0, -1] = 1
        A[-1, -1] = 1

        tmp = beta * ws + intercept
        if USE_MULTIPROC:
            with Pool(processes=PROC_NUM) as pool:
                results = pool.starmap(rmse_process, [(t, self.eval_step, x[t - self.eval_step].copy(), y, A, C, tmp) 
                                                      for t in range(self.eval_step, sample_T - 1)])
            loss = np.sum(results)
        else:
            for t in range(self.eval_step, sample_T - 1):
                x_s = x[t - self.eval_step].copy()
                for s in range(t - self.eval_step, t):
                    x_s = A @ x_s + tmp[s+1]
                loss += (y[t] - C @ x_s)[0] ** 2
        return np.sqrt(loss / sample_T)

    def negative_Q(self, x, P, y, w, sample_T):
        negative_logQ = 0
        b_shape_0 = self.b.shape[0]
        A = np.zeros((b_shape_0 * 3 + 1, b_shape_0 * 3 + 1))
        intercept = np.zeros((b_shape_0 * 3 + 1,))
        beta = np.zeros((b_shape_0 * 3 + 1,))
        ws = np.zeros((w.shape[0], b_shape_0 * 3 + 1))

        for i in range(b_shape_0):
            A[3 * i + 2, 3 * i + 1] = 1
            A[3 * i + 1, 3 * i] = 1
            A[3 * i, 3 * i] = self.b[i][0]
            A[3 * i, 3 * i + 1] = self.b[i][1]
            intercept[3 * i] = self.c[i]
            beta[3 * i] = self.b[i, 2]
            ws[:, 3 * i] = w[:, i]

        for t in range(sample_T - 1):
            tmp = A @ x[t] + beta * ws[t + 1] + intercept
            for i in range(b_shape_0):
                x_diff = x[t + 1, 3 * i] - tmp[3 * i]
                negative_logQ += (x_diff ** 2 / self.omega[i]) + (P[t + 1, 3 * i, 3 * i] / self.omega[i])
            if self.omegaMeanshift > 0:
                x_diff_meanshift = x[t + 1, -1] - x[t, -1]
                negative_logQ += (x_diff_meanshift ** 2 / self.omegaMeanshift) + (P[t + 1, -1, -1] / self.omegaMeanshift)
        return negative_logQ

    def kalman_filter(self, y, w):
        num_steps = w.shape[0]
        num_vars = self.b.shape[0] * 3 + 1
        x = np.zeros((num_steps, num_vars))
        P = np.zeros((num_steps, num_vars, num_vars))
        A = np.zeros((num_vars, num_vars))
        C = np.zeros((1, num_vars))
        intercept = np.zeros((num_vars,))
        beta = np.zeros((num_vars,))
        Q = np.zeros((num_vars, num_vars))
        R = self.sigma
        ws = np.zeros((num_steps, num_vars))

        for i in range(self.b.shape[0]):
            A[3 * i + 2, 3 * i + 1] = 1
            A[3 * i + 1, 3 * i] = 1
            A[3 * i, 3 * i] = self.b[i][0]
            A[3 * i, 3 * i + 1] = self.b[i][1]
            C[0, 3 * i] = 1
            intercept[3 * i] = self.c[i]
            beta[3 * i] = self.b[i, 2]
            ws[:, 3 * i] = w[:, i]
            Q[3 * i, 3 * i] = self.omega[i]
        Q[-1, -1] = self.omegaMeanshift
        C[0, -1] = 1
        A[-1, -1] = 1
        x[0] = np.full((num_vars,), y[0] / self.b.shape[0])
        x[0, -1] = 0

        tmp = beta * ws + intercept
        for t in range(1, num_steps):
            xt1t = A @ x[t - 1] + tmp[t]
            Pt1t = A @ P[t - 1] @ A.T + Q
            K = Pt1t @ C.T @ np.linalg.inv(C @ Pt1t @ C.T + R)
            x[t] = xt1t + K @ (y[t] - C @ xt1t)
            P[t] = Pt1t - K @ C @ Pt1t
        return x, P

    def update(self, x, y, w, P, sample_T):
        num_vars = self.b.shape[0]
        decay_factors = self.decay ** np.arange(2, sample_T)

        for i in range(num_vars):
            u = x[:, 3 * i]
            wi = w[:, i]
            A = np.zeros((4, 4))
            b = np.zeros((4,))

            # 计算矩阵 A 的各元素
            u_t_minus_1 = u[1:sample_T - 1]
            u_t_minus_2 = u[:sample_T - 2]
            wi_t = wi[2:sample_T]
            P_t_minus_1 = P[1:sample_T - 1, 3 * i]
            P_t_minus_2 = P[:sample_T - 2, 3 * i]

            A[0, 0] = np.sum(decay_factors * (u_t_minus_1 ** 2 + P_t_minus_1[:, 3 * i]))
            A[0, 1] = np.sum(decay_factors * (u_t_minus_1 * u_t_minus_2 + P_t_minus_1[:, 3 * i + 1]))
            A[0, 2] = np.sum(decay_factors * (u_t_minus_1 * wi_t))
            A[0, 3] = np.sum(decay_factors * u_t_minus_1)
            A[1, 0] = A[0, 1]
            A[1, 1] = np.sum(decay_factors * (u_t_minus_2 ** 2 + P_t_minus_2[:, 3 * i]))
            A[1, 2] = np.sum(decay_factors * (u_t_minus_2 * wi_t))
            A[1, 3] = np.sum(decay_factors * u_t_minus_2)
            A[2, 0] = A[0, 2]
            A[2, 1] = A[1, 2]
            A[2, 2] = np.sum(decay_factors * (wi_t ** 2))
            A[2, 3] = np.sum(decay_factors * wi_t)
            A[3, 0] = A[0, 3]
            A[3, 1] = A[1, 3]
            A[3, 2] = A[2, 3]
            A[3, 3] = np.sum(decay_factors)

            # 计算向量 b 的各元素
            u_t = u[2:sample_T]
            b[0] = np.sum(decay_factors * (P[2:sample_T, 3 * i, 3 * i + 1] + u_t * u_t_minus_1))
            b[1] = np.sum(decay_factors * (P[2:sample_T, 3 * i, 3 * i + 2] + u_t * u_t_minus_2))
            b[2] = np.sum(decay_factors * (u_t * wi_t))
            b[3] = np.sum(decay_factors * u_t)

            bhat = np.linalg.solve(A, b)
            self.b[i, 0] = bhat[0]
            self.b[i, 1] = bhat[1]
            self.b[i, 2] = bhat[2]
            self.c[i] = bhat[3]
            self.variance[i] = np.linalg.inv(A)
