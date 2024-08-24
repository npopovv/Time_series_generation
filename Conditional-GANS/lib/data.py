import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)



def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def rolling_window(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)


def get_hf_dataset(assets = 'AAPL', with_vol=True):
    """
    Get high_freq series: volume and close
    """

    df_asset = pd.read_csv('data/dataset.csv', index_col=0, header=[0, 1])\
    .sort_index(axis=1)[assets].fillna(method = 'ffill').astype(float)
    
    price = np.log(df_asset[['close']].values)
    rtn = (price[1:] - price[:-1]).reshape(1, -1, 1)
    vol = np.log(df_asset[['volume']].values[-rtn.shape[1]:]).reshape(1, -1, 1)
    data_raw = np.concatenate([rtn, vol], axis=-1)
      
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed




def get_data(data_type, p, q, **data_params):
    if data_type == 'HIGH_FREQ':
        pipeline, x_real_raw, x_real = get_hf_dataset(**data_params)
    else:
        raise NotImplementedError('Dataset %s not valid' % data_type)
    assert x_real.shape[0] == 1
    x_real = rolling_window(x_real[0], p + q)
    return x_real    
