from typing import Iterable

import funcy
import numpy as np
import os
import pandas as pd
import re
import time
import pickle
import gzip

from fastai.core import num_cpus
from fastai.imagito.classes import ClassFolders
from fastai.torch_core import num_distrib
from fastai.vision import ImageList, flip_lr, imagenet_stats


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


def load_distilled_imagelist(save_dir, bs=64, size=32):
    n_gpus = num_distrib() or 1
    workers = min(8, num_cpus() // n_gpus)
    image_list = ImageList.from_folder(save_dir)
    return (image_list
            .split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35, 1))
            .normalize(imagenet_stats)
            )


def make_imagelist(bs, classes, path, sample, size, workers):
    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus() // n_gpus)
    image_list = ImageList.from_folder(path)
    image_list = filter_classes(image_list, classes)
    return (image_list
            .use_partial_data(sample)
            .split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35, 1))
            .normalize(imagenet_stats))


def filter_classes(image_list, classes=None):
    if (classes is None):
        return image_list

    class_names = ClassFolders.from_indices(classes)
    def class_filter(path):
        for class_name in class_names:
            if class_name in str(path):
                return True
        return False

    return image_list.filter_by_func(class_filter)
