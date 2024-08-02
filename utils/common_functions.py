import torch
import random
import numpy as np
import os
import pickle
import pandas as pd
from typing import Union


def set_seed(seed):
    """Sets seed for experiments reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def convert_lists_and_tuples_to_string(data):
    """Recursively replaces all lists in a dictionary (including nested dictionaries)
        with a string representation of the list.
    """
    data = data.copy()
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = convert_lists_and_tuples_to_string(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            data[key] = str(value)
        elif value is None:
            data[key] = str(value)
    return data


def write_file(file, path):
    extension = os.path.splitext(path)[1]
    if extension == '.pickle':
        with open(path, 'wb') as f:
            pickle.dump(file, f)


def read_file(path):
    extension = os.path.splitext(path)[1]
    try:
        if extension == '.pickle':
            with open(path, 'rb') as f:
                file = pickle.load(f)
        else:
            print('Unknown extension')
            return None
    except FileNotFoundError:
        print('File not found: ', path)
        return None
    return file


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    elif path_to_file.endswith('parquet'):
        return pd.read_parquet(path_to_file)
    else:
        raise ValueError("Unsupported file format")
