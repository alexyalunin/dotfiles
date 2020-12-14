import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib 
import collections
import pickle, os, time, sys
from IPython.display import display
import traceback
import json
import subprocess


def overwrite_file(new_file, old_file):
    print(f'Overwriting {old_file} with {new_file}')
    bashCmd = ["cp", new_file, old_file]
    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('output:', output)
    print('error:', error)
    return output, error


def print_torch(torch):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print(f'use_cuda: {use_cuda}, n_gpu: {n_gpu}, devices:{[torch.cuda.get_device_name(i) for i in range(n_gpu)]}')
    return use_cuda, device, n_gpu


def print_packages(*args):
    for lib in args:
        print(f'{lib.__name__}: {lib.__version__}')
        print(lib.__file__)


def seed_everything(seed, random, os, np, torch):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
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
    
    
def pickle_dump(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)
        print(f'Saved as pickle {os.path.realpath(file.name)}')
        

def pickle_load(file_name):
    with open(file_name, 'rb') as file:
        res = pickle.load(file)
    return res


def json_dump(obj, file_name):
    assert isinstance(obj, dict), f"{obj} is not a dict"
    with open(file_name, 'w') as file:
        json.dump(obj, file)
        print(f'Saved as json {os.path.realpath(file.name)}')
        

def json_load(file_name):
    with open(file_name, 'r') as file:
        res = json.loads(file.read())
    return res


def reload(library):
    importlib.reload(library)


def display_model_modules(model, sorted=True):
    print(f'Total number of parameters: {model.num_parameters()}')
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


class MyErrorCatcher():
    def __enter__(self, dir='./'):
        self.file = open(dir+'my_output.txt', 'a+')
        self.file.write(time.ctime() + ' STARTED' + '\n')
        
    def __exit__(self, exc_type, exc_value, tb):
        self.file.write(time.ctime() + '\n')
        if exc_type:
            self.file.write(str(exc_type) + '\n')
            self.file.write(str(exc_value) + '\n')
            for x in traceback.format_tb(tb):
                self.file.write(str(x) + '\n')
        self.file.close()
