from typing import Iterable

import funcy
import numpy as np
import os
import pandas as pd
import re
import time
import pickle
import gzip


def get_date_str(seconds=False):
    if seconds:
        return time.strftime('%Y-%m-%d-%H:%M:%S')
    else:
        return time.strftime('%Y-%m-%d-%H:%M')


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f,  protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load_gzip(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f, encoding='latin-1')

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def update_batch_size(pg):
    new_pars = []
    for p in pg:
        new_p = p.copy()
        if p['size'] > 128:
            new_p['bs'] = min(new_p['bs'], 128)
        new_pars.append(new_p)
    return new_pars

from tqdm import *
from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()
tqdm_nice = tqdm_notebook if in_notebook() else tqdm
