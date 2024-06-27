import numpy as np
import pandas as pd
import plotly.express as px
from utils import outlier_detect
import plotly.io as pio
import pickle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from CPD import std_cpd
from SSM_ToShimao_20240415.state_space import StatespacewithShift
from scipy.stats import norm
import re
from utils import print_APC_params


if __name__ == '__main__':
    # ---------------------  file and variables -------------------------------

    file_name = 'PHD_20220901_1128_noNA_splitLag.csv' # .csv
    #nooutlierfile_name = file_name[:-4] + 'NoOutlier.csv'
    time = '9to11'
    moel_name = '0610_SSMeanshift'
    # CV_name = ['TIC1141', 'TIC1137', 'TIC1135', 'TIC1102', 'TI1129', 'TI1127', 'TI1125', 'TI1123', 'TIC1001']
    CV_name = ['TIC1141']
    #MV_name = ['FIC1002', 'FIC1105', 'BP1106', 'FIC1107', 'FIC1108', 'FIC1104', 'FIC1305', 'FIC1306', 'FIC1307',
    #           'TIC1156C', 'TI1122', 'TIC1110', 'TI1002', 'FIC1006', 'FIC1109']
    MV_name = ['FIC1002Lag0',
               'FIC1105Lag5', 'FIC1105Lag10', 'FIC1105Lag15', 'FIC1105Lag20',
               'BP1106Lag0', 'BP1106Lag5', 'BP1106Lag10', 'BP1106Lag15',
               'FIC1107Lag0', 'FIC1107Lag5', 'FIC1107Lag10',
               'FIC1108Lag0', 'FIC1108Lag5',
               'FIC1104Lag5', 'FIC1104Lag10', 'FIC1104Lag15', 'FIC1104Lag20',
               'FIC1305Lag5', 'FIC1305Lag10', 'FIC1305Lag15', 'FIC1305Lag20',
               'FIC1306Lag0', 'FIC1306Lag5', 'FIC1306Lag10',
               'FIC1307Lag0', 'FIC1307Lag5',
               'TIC1156CLag10', 'TIC1156CLag15', 'TIC1156CLag20', 'TIC1156CLag25', 'TIC1156CLag30',
               'TI1122Lag5', 'TI1122Lag10', 'TI1122Lag15', 'TI1122Lag20',
               'TIC1110Lag10', 'TIC1110Lag15', 'TIC1110Lag20', 'TIC1110Lag25', 'TIC1110Lag30',
               'TI1002Lag20', 'TI1002Lag25', 'TI1002Lag30', 'TI1002Lag35', 'TI1002Lag40',
               'FIC1006Lag0',
               'FIC1109Lag0']
    sample_T = 80000
    setting_lag = [0] * len(MV_name)
    #setting_lag = [42, 29, 15, 9, 2, 11, 16, 13, 48, 27, 4, 29, 16, 38, 25]  ### TIC1141
    #setting_lag = [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ### TIC1137
    #setting_lag = [0, 3, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ### TI1129
    #setting_lag = [0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0]  ### TI1123
    # ---------------------------- algorithm ---------------------------------
    # Outlier Detection
    data = pd.read_csv(file_name)
    data.columns.values[0] = 'time_index'
    #data.to_csv(nooutlierfile_name, index=False)

    num_var = len(MV_name)
    max_lag = max(np.array(setting_lag))
    data = data[MV_name + CV_name]
    # data = delete_abnormal(data)
    mean_values = data.mean()
    for column in data.columns:
        outlier_mask = (data[column] < mean_values[column] - 6 * data[column].std()) | (
                    data[column] > mean_values[column] + 6 * data[column].std())
        data.loc[outlier_mask, column] = mean_values[column]
    data = data.iloc[0:sample_T, :]
    for mv in MV_name:
        data[mv] = data[mv] - np.mean(data[mv])
    for cv_name in CV_name:
        print(cv_name)
        y = data[cv_name].values
        w = np.zeros((len(y), num_var))
        for i, mv in enumerate(MV_name):
            w[:, i] = data[mv].shift(setting_lag[i]).values
        w = w[max_lag:, :]
        y = y[max_lag:]
        #SSmodel = StatespacewithShift(num_var, decay=0.99995)
        SSmodel = StatespacewithShift.load('model_save/9to11_TIC1141_0610_SSMeanshiftmodel_500_allmv_dv.pkl')
        w_base = np.zeros_like(w)
        y_base = np.zeros_like(y)

        SSmodel.fit(y, w, len(y), 2000)
        with open('model_save/%s_%s_%smodel_500_allmv_dv.pkl' % (time, cv_name, moel_name), 'wb') as file:
            pickle.dump(SSmodel, file)

    # ----------------------------- print params -------------------------------------
    df = pd.DataFrame(columns=['CV_name', 'MV_name', 'tau', 'zeta', 'K','p-value'])
    df['K'].astype(str)
    for cv in CV_name:
        tau, zeta, K, delta = print_APC_params('./model_save/%s_%s_%smodel_500_allmv_dv.pkl' % (time, cv, moel_name))
        p = [0]*len(MV_name)
        for k in range(len(K)):
            p[k] = min(norm.cdf(K[k]/(delta[k]/1.96)), 1- norm.cdf(K[k]/(delta[k]/1.96)))
            K[k] = '%0.4lf±%0.4lf' % (K[k], delta[k])
        df_cv = pd.DataFrame(columns=['CV_name', 'MV_name', 'tau', 'zeta', 'K', 'p-value'])
        df_cv['CV_name'] = [cv] * len(MV_name)
        df_cv['MV_name'] = MV_name
        df_cv['tau'] = tau
        df_cv['zeta'] = zeta
        df_cv['K'].astype(str)
        df_cv['K'] = K
        df_cv['p-value'] = p
        df = pd.concat([df, df_cv], axis=0)
        df.to_csv('./model_save/estimation/csvresult/%s_%sparams.csv' % (cv, moel_name), encoding='utf-8-sig', index=False)

