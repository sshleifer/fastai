from typing import Iterable

import funcy
import numpy as np
import os
import pandas as pd
import re
import time
import pickle
import gzip

from fastai.vision import *
from fastai.imagito.utils import *
from fastai.imagito.classes import ClassFolders


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

def filter_classes(image_list, classes=None):
    if classes is None:
        return image_list

    class_names = ClassFolders.from_indices(classes)
    def class_filter(path):
        for class_name in class_names:
            if class_name in str(path):
                return True
        return False

    return image_list.filter_by_func(class_filter)

def get_data(size, woof, bs, sample, classes=None, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    image_list = ImageList.from_folder(path)
    image_list = filter_classes(image_list, classes)

    return (image_list
            .use_partial_data(sample)
            .split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

def assertIsPerc(val):
    assert 0 <= val <= 1

from tqdm import *
from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()
tqdm_nice = tqdm_notebook if in_notebook() else tqdm
