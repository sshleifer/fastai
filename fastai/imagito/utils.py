from pathlib import Path
from typing import Iterable

import funcy
import numpy as np
import os
import pandas as pd
import re
import time
import pickle
import gzip


import torch
from torchvision.utils import save_image

from fastai.core import num_cpus
from fastai.imagito.classes import ClassUtils, IMAGENETTE_RENAMER
from fastai.torch_core import num_distrib
from fastai.vision import ImageList, flip_lr, imagenet_stats

from fastai.vision import *
from fastai.imagito.utils import *
from fastai.imagito.classes import ClassUtils


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

return_true = lambda x: True
def get_data(size, woof, bs, sample, classes=None, workers=None, shuffle_train=True, filter_func=return_true):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    image_list = ImageList.from_folder(path)
    image_list = ClassUtils.filter_classes(image_list, classes)

    return (image_list
            .filter_by_func(filter_func)
            .use_partial_data(sample)
            .split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers, shuffle_train=shuffle_train)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

def assertIsPerc(val):
    assert 0 <= val <= 1

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


def save_distilled_images_for_fastai(results_pth, save_dir, model_slug='', map_location=None):
    save_dir = Path(save_dir)
    assert 'train' not in str(save_dir), f'this will add train/ for you, got {save_dir}'
    save_dir.mkdir(exist_ok=True)
    triples_lst = torch.load(results_pth, map_location)
    i = 0
    for a,b, _ in triples_lst:
        for img,label in zip(a,b):
            label_code  = IMAGENETTE_RENAMER[label.item()]
            save_path = save_dir / f'train/{label_code}/{model_slug}_{i}.jpg'
            if save_path.exists():
                raise ValueError(f'{save_path} exists')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(img, save_path)
            i += 1
