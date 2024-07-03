import json
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config file")
args = parser.parse_args()

config_file_path = args.config

with open(config_file_path, 'r') as file:
    config = json.load(file)

data_path = config['Data_path']
data = pd.read_csv(data_path)

if config['Test_size'] == -1:
    config['Test_size'] = data.shape[0] - config['Training_size']
spilt_size = config['Training_size'] + config['Test_size']
data = data[:spilt_size]
data['Time'] = range(data.shape[0])
data = data[['Time'] + config['MV_name'] + config['CV_name']]

def config_setter(lags=None):
    ret = copy.deepcopy(config)
    if(lags):
        ret['Lags'] = lags
    return ret

from SSM import SMMKalmanFilters
from openbox import Optimizer, space as sp
from openbox import ParallelOptimizer

# Define Search Space
space = sp.Space()
sp_lags = []
lowv = config['blkMin']
highv = config['blkMax']
base_lags = config['Lags']
numv = len(base_lags)
for i in range(numv):
    sp_lags.append(sp.Int(f'lag{i}', lowv, highv, default_value=base_lags[i]))
space.add_variables(sp_lags)

def opt_wrapper(config):
    lags = [config[f'lag{i}'] for i in range(numv)]
    model_config = config_setter(lags=lags)
    model = SMMKalmanFilters(config=model_config)
    rmse, _, _ = model.fit(data)
    return {'objectives': [rmse]}

# Run
if __name__ == '__main__':
    opt = ParallelOptimizer(opt_wrapper, space,
                    parallel_strategy='async',
                    batch_size=16,
                    batch_strategy='default',
                    num_objectives=1,
                    num_constraints=0,
                    max_runs=500,
                    surrogate_type='prf',
                    task_id='blk_opt')
    history = opt.run()
    print(history)