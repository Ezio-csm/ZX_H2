import numpy as np
import pandas as pd
from state_space import StatespacewithShift
from state_space import USE_MULTIPROC, PROC_NUM
from multiprocessing import Pool
# from CPD import std_cpd

def Kalman_forecast_process(t, step, x_s, y_base, A, C, b, b_shape_0, w, intercept):
    for s in range(t - step, t):
        x_s = A @ x_s + intercept
        x_s[:3*b_shape_0:3] += b[:, 2] * w[s + 1]
    return (C @ x_s)[0] + y_base[t]

class SMMKalmanFilters:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.y = None
        self.w = None
        self.w_base = None
        self.y_base = None
        self.sample_T = self.config['Training_size']
        self.max_iter = self.config['EM_max_iter']
        self.model = StatespacewithShift(len(self.config['MV_name']), decay=1,
                                      output_Qn=self.config['output_Qn'],
                                      verbose=self.config['verbose'],
                                      eval_step=self.config['evaluation_pred_step'][0])

    def Kalman_forecast(self, steps, cv):
        y = self.y
        w = self.w
        w_base = self.w_base
        y_base = self.y_base
        model = self.model
        x, _ = model.kalman_filter(y, w)
        model.x = x
        b_shape_0 = model.b.shape[0]

        C = np.zeros((1, b_shape_0 * 3 + 1))
        intercept = np.zeros((b_shape_0 * 3 + 1,))
        A = np.zeros((b_shape_0 * 3 + 1, b_shape_0 * 3 + 1))

        for i in range(b_shape_0):
            A[3 * i + 2, 3 * i + 1] = 1
            A[3 * i + 1, 3 * i] = 1
            A[3 * i, 3 * i] = model.b[i][0]
            A[3 * i, 3 * i + 1] = model.b[i][1]
            C[0, 3 * i] = 1
            intercept[3 * i] = model.c[i]
        C[0, -1] = 1
        A[-1, -1] = 1

        data = pd.DataFrame()
        max_lag = max(self.config['Lags'])
        
        for step in steps:
            y_true = []
            y_pred = []
            tmp = np.array(self.data[cv].values)
            for t in range(len(self.data) - max_lag):
                y_true.append(tmp[t + max_lag] + y_base[t])
            
            if USE_MULTIPROC:
                with Pool(processes=PROC_NUM) as pool:
                    y_pred = pool.starmap(Kalman_forecast_process, [(t, step, model.x[t - step], y_base, A, C, model.b, b_shape_0, w, intercept) 
                                                                      for t in range(len(self.data) - max_lag)])
            else:
                for t in range(len(self.data) - max_lag):
                    x_s = model.x[t - step]  # 初始x_s
                    for s in range(t - step, t):
                        x_s = A @ x_s + intercept
                        x_s[:3*b_shape_0:3] += model.b[:, 2] * w[s + 1]
                    y_pred.append((C @ x_s)[0] + y_base[t])
            
            data[f'pred_{cv}_{step}'] = y_pred
            data[f'true_{cv}_{step}'] = y_true

        return data


    def preprocessing(self, data):
        '''outlier detection'''
        mean_values = data.mean()
        for column in data.columns:
            outlier_mask = (data[column] < mean_values[column] - 6 * data[column].std()) | (
                        data[column] > mean_values[column] + 6 * data[column].std())
            data.loc[outlier_mask, column] = mean_values[column]

        # '''Shift data around zero'''
        # for mv in self.config['MV_name']:
        #     data[mv] = data[mv] - np.mean(data[mv])

        '''Change point detection'''
        num_var = len(self.config['MV_name'])
        setting_lag = self.config['Lags']
        max_lag = max(setting_lag)
        for cv_name in self.config['CV_name']:
            y = np.array(data[cv_name].values)
            w = np.zeros((len(y), num_var))
            for i, mv in enumerate(self.config['MV_name']):
                w[:, i] = np.array(data[mv].shift(setting_lag[i]).values)
            y = y[max_lag:]
            w = w[max_lag:, :]
            w_base = np.zeros_like(w)
            y_base = np.zeros_like(y)

        return w, y, w_base, y_base
    
    def Fitting(self, w, y):
        
        # print(max_iter)
        self.model.fit(y[:self.sample_T], w[:self.sample_T, :], self.sample_T, max_iter=self.max_iter)
        return self.model

    def Evaluation(self):
        steps = self.config['evaluation_pred_step']
        #print("step", steps)
        step = steps[0]
        #print("stp:", step)
        for cv in self.config['CV_name']:
            data = self.Kalman_forecast(steps, cv)
            #print("data_fitted", data)
            y_true = np.array(data['true_%s_%i' % (cv, step)].values)
            y_pred = np.array(data['pred_%s_%i' % (cv, step)].values)
            RMSE = np.sqrt(np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / len(y_true))
        return RMSE, y_true, y_pred
    
    def Forecast(self):
        if self.model is None:
            raise ValueError("Model fitting is not done")
        steps = self.config['prediction_steps']
        for cv in self.config['CV_name']:
            data = self.Kalman_forecast(steps, cv)
        return data
    
    def forward(self):
        self.w, self.y, self.w_base, self.y_base = self.preprocessing(self.data)
        # print(self.w.shape)
        # print(len(self.y))
        # print(len(self.data))
        self.model = self.Fitting(self.w, self.y)
        RMSE, y_true, y_pred = self.Evaluation()
        return RMSE, y_pred, y_true
