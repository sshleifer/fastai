from __future__ import print_function
import os
import shutil
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_url

from fastai.imagito.train_imagito import get_data

S3 = 'https://s3.amazonaws.com/fast-ai-'
S3_IMAGE    = f'{S3}imageclas/'
IMAGENETTE = f'{S3_IMAGE}imagenette'
IMAGENETTE_160 = f'{S3_IMAGE}imagenette-160'
IMAGENETTE_320 = f'{S3_IMAGE}imagenette-320'
IMAGEWOOF = f'{S3_IMAGE}imagewoof'
IMAGEWOOF_160 = f'{S3_IMAGE}imagewoof-160'
IMAGEWOOF_320 = f'{S3_IMAGE}imagewoof-320'


def get_imagenette_train_ds():
    return get_data(size=224, woof=False, bs=64, sample=1., classes=None, workers=None).train_ds

def get_imagewoof_train_ds():
    return get_data(size=224, woof=True, bs=64, sample=1., classes=None, workers=None).train_ds


def get_train_loader(state):
    imagelist = get_data(size=224, woof=False, bs=state.batch_size, sample=1., classes=None, workers=None)
    return imagelist.train_dl.dl

def get_test_loader(state): return get_train_loader(state)
