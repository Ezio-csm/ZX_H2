import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

config_file_path = './configs/TIC1141.json'

config = config_setter()
data_path = config['Data_path']
train_size = config['Training_size'] + config['Test_size']
data = pd.read_csv(data_path)
data = data[:train_size]
data['Time'] = range(data.shape[0])
data = data[['Time'] + config['MV_name'] + config['CV_name']]

def config_setter(lags=None):
    ret = copy.deepcopy(config)
    if(lags):
        ret['Lags'] = lags
    return ret

from SSM import SMMKalmanFilters
from SimulatedAnnealing import SimulatedAnnealing
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
    model = SMMKalmanFilters(data, config=model_config)
    error, _, _ = model.forward()
    return {'objectives': [error]}

# Run
if __name__ == '__main__':
    opt = ParallelOptimizer(opt_wrapper, space,
                    parallel_strategy='async',
                    batch_size=8,
                    batch_strategy='default',
                    num_objectives=1,
                    num_constraints=0,
                    max_runs=500,
                    surrogate_type='prf',
                    task_id='blk_opt')
    history = opt.run()
    print(history)