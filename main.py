import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from pyinstrument import Profiler

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config file")
parser.add_argument("output", type=str, help="output arg")
args = parser.parse_args()

config_file_path = args.config
output_arg = args.output

with open(config_file_path, 'r') as file:
    config = json.load(file)

# ---------------------
profiler = Profiler()
profiler.start()
# ---------------------

data_path = config['Data_path']
data = pd.read_csv(data_path)

if output_arg != None:
    config['CV_name'] = [output_arg]

if config['Test_size'] == -1:
    config['Test_size'] = data.shape[0] - config['Training_size']
spilt_size = config['Training_size'] + config['Test_size']
data = data[:spilt_size]
data['Time'] = range(data.shape[0])
data = data[['Time'] + config['MV_name'] + config['CV_name']]

# if config['Test_size'] == -1:
#     config['Test_size'] = data.shape[0] - config['Training_size']
# train_data = data[:config['Training_size']]
# test_data = data[config['Training_size']:config['Training_size'] + config['Test_size']]

# def norm_data(data):
#     data['Time'] = range(data.shape[0])
#     data = data[['Time'] + config['MV_name'] + config['CV_name']]
# norm_data(train_data)
# norm_data(test_data)

from SSM import SMMKalmanFilters

if __name__ == '__main__':
    if output_arg != None:
        print(f"CV_name changed : {config['CV_name']}")
    model = SMMKalmanFilters(config=config)
    if config['ModelLoaded']:
        with open(config['ModelLoadedPath'], 'rb') as f:
            model.model = pickle.load(f)
    else:
        rmse, y_pred, y_true = model.fit(data)
        print(f"Evaluation RMSE on test {config['evaluation_pred_step']} steps: {rmse}")

    results = model.forecast(data)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if output_arg != None:
        results.to_csv(f'./results/{output_arg}.csv')
    else:
        results.to_csv('./results/results.csv')

    if config['ModelSaved']:
        if not os.path.exists(os.path.dirname(config['ModelSavedPath'])):
            os.makedirs(os.path.dirname(config['ModelSavedPath']))
        with open(config['ModelSavedPath'], 'wb') as f:
            pickle.dump(model.model, f)


# np.save('pred.npy', y_pred)
# np.save('true.npy', y_true)


# ---------------------
profiler.stop()
print(profiler.output_text(unicode=True, color=True))
# ---------------------