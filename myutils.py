import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import importlib 
import collections
import pickle, os, time, sys
from IPython.display import display
import traceback
import json
import subprocess
from datetime import datetime, timedelta
import itertools
import copy


def remove_outliers(an_array: np.ndarray, max_deviations=3):
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = an_array[not_outlier]
    return no_outliers


def text_stats(s: pd.Series, remove_out=True):
    lens = s.apply(lambda x: len(x.split(' '))).values
    if remove_out:
        lens = remove_outliers(lens)
    lens = pd.Series(lens)

    print('Mean len: ', lens.mean())
    _ = lens.hist(bins=50)


# decorator
def pandas_display(func):
    default_1 = pd.options.display.max_rows
    default_2 = pd.options.display.max_colwidth
    def wrapper(_df, exclude_cols=None, properties=None, limit_float=True):
        pd.options.display.max_rows = 1000
        pd.options.display.max_colwidth = None
        func(_df, exclude_cols, properties, limit_float)
        pd.options.display.max_rows = default_1
        pd.options.display.max_colwidth = default_2
    return wrapper
    
    
class pandas():
    @staticmethod
    def cols_to_dict(df, key_col, value_col):
        """
            key_col: str or List
            value_col: str or List
        """
        return pd.Series(df[value_col].values, index=df[key_col]).to_dict()
    
    @staticmethod
    def get_indices_where_value(s: pd.Series, v):
        return s[s==v].index.values
    
    @staticmethod
    def get_indices_where_na(s: pd.Series):
        a = pd.isna(s)
        return a[a].index.values
    

def display_series(s):
    print(s.to_string())
    
@pandas_display
def display_df(_df, exclude_cols, properties, limit_float):
    if exclude_cols is not None:
        if not isinstance(exclude_cols, list):
            exclude_cols = list(exclude_cols)
        cols = [x for x in _df.columns.values if x not in set(exclude_cols)]
        _df = _df[cols]

    if not properties:
        properties = {
            'text-align': 'left',
            'white-space': 'pre-wrap',
            'word-wrap': 'break-word',
            'width': '230px',
            'max-width': '230px'
        }
    a = _df.style.set_properties(**properties)
    if limit_float:
        a.precision = 3
    display(a)


class viz():
    @staticmethod
    def corr_df(df):
        plt.figure(figsize=(12,12))
        corr = df.corr()
        cols = [x[:20] for x in corr.columns]
        sns.heatmap(corr, 
                xticklabels=cols,
                yticklabels=cols,
                vmin=-1, vmax=1, cmap='coolwarm',
                annot = True)
    
    @staticmethod
    def plot_countbar(a, print_proportions=True):
        plt.figure(figsize=(8,5))
        sns.set_theme(style="darkgrid")

        df = pd.DataFrame({'values': a})
        unique_sorted = sorted(df['values'].unique())
        ax = sns.countplot(x='values', data=df, order=unique_sorted)

        # df stuff
        res_df = []
        if print_proportions:
            C = collections.Counter(a)
            total_sum = sum(C.values())
            for k,v in C.most_common(): 
                res_df.append({'k':k, 'v':v, '%':round(v/total_sum, 3)*100})
        res_df = pd.DataFrame(res_df)
        res_df = res_df.set_index('k')

        for p, label in zip(ax.patches, res_df.loc[unique_sorted].iterrows()):
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]

    #         label = f'{int(label[1]["v"])}, {label[1]["%"]}%'
            label = f'{int(label[1]["v"])}'
            ax.annotate(label, (x.mean(), y), ha='center', va='bottom')
        plt.show()
    #     display(res_df)
        display(res_df.sort_index())


def concat_list_of_lists(a):
    return list(itertools.chain.from_iterable(a))


def overwrite_file(new_file, old_file):
    print(f'Overwriting {old_file} with {new_file}')
    bashCmd = ["cp", new_file, old_file]
    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('output:', output)
    print('error:', error)
    return output, error


def import_file(file):
    import importlib.util
    spec = importlib.util.spec_from_file_location("", file)
    mylib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mylib)
    return mylib


def print_torch(torch):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', None))
    print(f'use_cuda: {use_cuda}, n_gpu: {n_gpu}, device: {device}, devices:{[torch.cuda.get_device_name(i) for i in range(n_gpu)]}')
    return use_cuda, device, n_gpu


def print_packages(*args,**kwargs):
    for lib in args:
        print(f'{lib.__name__}: {lib.__version__}')
        if 'print_path' in kwargs and kwargs['print_path']:
            print(lib.__file__)


def seed_everything(seed, random=None, os=None, np=None, torch=None):
    if random is not None:
        random.seed(seed)
    if os is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def pickle_save(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)
#         print(f'Saved as pickle {os.path.realpath(file.name)}')
        

def pickle_load(file_name):
    with open(file_name, 'rb') as file:
        res = pickle.load(file)
    return res


def json_save(obj, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(obj, file)
#         print(f'Saved as json {os.path.realpath(file.name)}')
        

def json_load(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        res = json.loads(file.read())
    return res


def reload(library):
    importlib.reload(library)


def display_model_modules(model, sorted=True):
#     print(f'Total number of parameters: {model.num_parameters()}')
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
    """
    with myutils.MyErrorCatcher() as mec:
        for t in tqdm.tqdm(..., total=..., file=mec):
    """
    def __enter__(self, file_path='./log.txt'):
        # self.file = open(file_path, 'a+')
        self.file = open(file_path, 'w')
        self.write('STARTED')
        return self
    
    def write(self, message):
        print(message)
        self.file.write(time.ctime() + ': ' + message + '\n')
        
    def flush(self):
        self.file.flush()
        
    def __exit__(self, exc_type, exc_value, tb):
        self.write('FINISHED')
        if exc_type:
            self.file.write(str(exc_type) + '\n')
            self.file.write(str(exc_value) + '\n')
            for x in traceback.format_tb(tb):
                self.file.write(str(x) + '\n')
        self.file.close()

        
class MeasureTime():
    """
    with myutils.MeasureTime() as _:
        ...
    """
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print(f'Elapsed time: {str(timedelta(seconds=(time.time() - self.t))).split(".")[0]}') 

        
def excel_save(df, path, long_columns=[], n_rows_to_freeze=1, n_cols_to_freeze=0, dropdown_cols=None, alternate_rows_step=None):
    """
        long_columns: List of colomn to be wider
        n_rows_to_freeze: number of rows to be sticky in header
        n_cols_to_freeze: number of columns
        dropdown_cols: Dict[column_name: List[values]]
        alternate_rows_step: step for rows to change color
    """
    if path.split('.')[-1] != 'xlsx':
        path += '.xlsx'
    n_rows = len(df)
    long_columns = set(long_columns)
    sheetname = 'Sheet1'
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    workbook = writer.book
    format = workbook.add_format({'text_wrap': True})
    format.set_align('top')
    df.to_excel(writer, sheet_name=sheetname, index=True)  # send df to writer
    worksheet = writer.sheets[sheetname]  # pull worksheet object

    if alternate_rows_step is not None:
        row_color = workbook.add_format({'text_wrap': True, 'border':1})
        row_color.set_align('top')
        row_color.set_bg_color('#dce6f1')
        for i in range(1, len(df), alternate_rows_step*2):
            for j in range(i, i+alternate_rows_step):
                if j < len(df):
                    worksheet.set_row(j, cell_format=row_color)

    for idx, col in enumerate(df, 1):  # loop through all columns
        series = df[col]
        max_len = max(
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            ) + 1  # adding a little extra space
        l = min(max_len, 20)
        if series.name in long_columns:
            l = min(max_len, 60)
        worksheet.set_column(idx, idx, l, format)

        if dropdown_cols is not None:
            if col in dropdown_cols:
                worksheet.data_validation(0, idx, n_rows, idx, {
                    'validate': 'list',
                    'source': dropdown_cols[col]
                })

    # Add a header format. (bold text on green background)
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BC',
        'border': 1})

    # Write the column headers with the defined format.
    for col_num, value in enumerate(df.columns.values, 1):
        worksheet.write(0, col_num, value, header_format)

    # Make header sticky (https://xlsxwriter.readthedocs.io/worksheet.html)
    worksheet.freeze_panes(n_rows_to_freeze, n_cols_to_freeze)

    writer.save()

        
def excel_load(path, sheet_name=0):
    for engine in ['openpyxl', 'xlrd', 'odf', 'pyxlsb']:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, engine=engine, usecols=lambda x: 'Unnamed' not in str(x))
            df = df.dropna(how='all')
            # print(f'{engine} is ok')
            return df
        except Exception as error:
            pass
            print(f'{engine} returned error: {error}')
    raise Exception('none of the engines worked')

            

def curr_datetime():
    now_datetime = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    return now_datetime

def reverse_dict(D):
    return {v: k for k, v in D.items()}



def check_tensor_to_numpy(x):
    try:
        import torch
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x
        else:
            print("converting x of type:", type(x), "to numpy array")
            return np.array(x)
    except:
        return x

    
def plot_3d(p, threshold=0.01, ax=None):
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    p = check_tensor_to_numpy(p)
    verts, faces, _, _ = measure.marching_cubes(p)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()