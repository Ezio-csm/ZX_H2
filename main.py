import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# from pyinstrument import Profiler

config_file_path = './configs/TIC1141.json'

with open(config_file_path, 'r') as file:
    config = json.load(file)

# ---------------------
# profiler = Profiler()
# profiler.start()
# ---------------------

data_path = config['Data_path']
train_size = config['Training_size'] + config['Test_size']
data = pd.read_csv(data_path)
data = data[:train_size]
data['Time'] = range(data.shape[0])
data = data[['Time'] + config['MV_name'] + config['CV_name']]

from SSM import SMMKalmanFilters

model = SMMKalmanFilters(data, config=config)
error, y_pred, y_true = model.forward()
# print(np.sqrt(np.mean((y_true - y_pred) ** 2)))

results = model.Forecast()
if not os.path.exists('./results'):
    os.makedirs('./results')
results.to_csv('./results/results.csv')


# np.save('pred.npy', y_pred)
# np.save('true.npy', y_true)


# ---------------------
# profiler.stop()
# print(profiler.output_text(unicode=True, color=True))
# ---------------------