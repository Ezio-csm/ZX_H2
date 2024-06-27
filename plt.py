import os
print(os.getcwd())
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import plotly.offline as offline

t_decays = [10, 20, 30, 40, 60, 120]
target_name = 'TIC1141'

def plotlyDataPX(data):
    fig = go.Figure()
    fig.update_layout(
        yaxis2=dict(
            title='Relative Error',
            overlaying='y',  # 使第二个y轴覆盖在第一个y轴上
            side='right',  # 将第二个y轴放在右侧
            zeroline=True,  # 显示0线
            zerolinecolor='#FF0000',  # 设置0线颜色
        )
    )

    for t_decay in t_decays:
        name_true = 'true_{}_{}'.format(target_name, t_decay)
        name_pred = 'pred_{}_{}'.format(target_name, t_decay)
        # diff = calc_diff(tmp_true, tmp_pred)
        trues = data[name_true].values
        preds = data[name_pred].values
        rmse = np.sqrt(np.mean((trues[t_decay:] - preds[t_decay:]) ** 2))
        if t_decay == 10:
            fig.add_scatter(x=np.array(range(len(data[name_true].values))), y=data[name_true].values, name='{}_true'.format(t_decay))
        fig.add_scatter(x=np.array(range(len(data[name_pred].values)))[t_decay:], y=data[name_pred].values[t_decay:], name='{}_pred'.format(t_decay))
        # fig.add_scatter(x=x, y=diff, name='{}_diff'.format(t_decay))
        print('RMSE for {}: {}'.format(t_decay, rmse))
    offline.plot(fig, filename='./results/fig_ssm_H2_2.html', auto_open=False)

def calc_diff(truth, pred):
    return [(pred[i] - truth[i]) / truth[i] for i in range(len(truth))]

if __name__ == "__main__":
    data = pd.read_csv('./results/results.csv')

    plotlyDataPX(data)