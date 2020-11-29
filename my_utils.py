import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib 
import collections
import pickle, os
from IPython.display import display


def pandas_display(func):
    default_1 = pd.options.display.max_rows
    default_2 = pd.options.display.max_colwidth
    def wrapper(_df, exclude_cols=None, properties=None):
        pd.options.display.max_rows = 1000
        pd.options.display.max_colwidth = None
        func(_df, exclude_cols, properties)
        pd.options.display.max_rows = default_1
        pd.options.display.max_colwidth = default_2
    return wrapper


@pandas_display
def display_df(_df, exclude_cols, properties):
    if exclude_cols:
        cols = np.setdiff1d(_df.columns.values, exclude_cols)
        _df = _df[cols]
    if not properties:
        properties = {
            'text-align': 'left',
            'white-space': 'pre-wrap',
            'word-wrap': 'break-word',
            'width': '230px',
            'max-width': '230px'
        }
    display(_df.style.set_properties(**properties))
    
    
def pickle_save(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)
        print(f'Uploaded as pickle {os.path.realpath(file.name)}')
        
        
def pickle_load(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def reload(library):
    importlib.reload(library)


def display_model_modules(model, sorted=True):
    d = collections.Counter()
    for name, parameter in model.named_parameters():
        d[name] = parameter.numel()
    if sorted:
        mc = d.most_common()
    else:
        mc = d.items()
    df = pd.DataFrame({'layer': [x[0] for x in mc], 'num_parameters': [x[1] for x in mc]})
    display_df(df, exclude_cols=None, properties={
            'text-align': 'left',
            'white-space': 'pre-wrap'
    })


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created {directory}')

