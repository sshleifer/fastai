from typing import Iterable

import funcy
import numpy as np
import os
import pandas as pd
import re
import time
import pickle
import gzip

from torch import nn
from fastai.vision import *
from fastai.imagito.utils import *
from fastai.imagito.classes import ClassUtils
from fastai.datasets import *
from fastai.torch_core import *
from fastai.vision.data import *


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
        if 'size' in p and p['size'] > 128:
            new_p['bs'] = min(new_p['bs'], 128)
        new_pars.append(new_p)
    return new_pars

return_true = lambda x: True
def get_data(size, woof, bs, sample, classes=None, workers=None, shuffle_train=True,
             filter_func=return_true, flip_lr_p=0.5):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    image_list = ImageList.from_folder(path)
    image_list = ClassUtils.filter_classes(image_list, classes, woof)

    return (image_list
            .filter_by_func(filter_func)
            .use_partial_data(sample)
            .split_by_folder(valid='val')

            .label_from_folder().transform(([flip_lr(p=flip_lr_p)], []), size=size)
            .databunch(bs=bs, num_workers=workers, shuffle_train=shuffle_train)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

def assertIsPerc(val):
    assert 0 <= val <= 1

from tqdm import *
from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()
tqdm_nice = tqdm_notebook if in_notebook() else tqdm
