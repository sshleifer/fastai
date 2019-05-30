import argparse
from pathlib import Path
from fastai.imagito.utils import *
from fastai.vision.models.xresnet2 import xresnet50_2
import torch
import math
from operator import itemgetter

global i

def load_model(model_dir, size, woof, bs, sample):
    model_params = pickle_load(model_dir + '/params.pkl')
    if model_params['size'] != size:
        print('Warning: Model was trained on image size=%d but is being used to predict for size=%d'
              % (model_params['size'], size))

    m = xresnet50_2
    print(model_params)
    data = get_data(size, woof, bs, sample, shuffle_train=False)

    l = Learner(data, m(c_out=10), path=model_dir)
    l.load('final_classif')
    return l, data

def choose_n(t, n):
    """Choose n random elements from tensor t"""
    idx = torch.randperm(t.size(0))[:n]
    return t[idx]

def retain_ones(mask, n):
    """Given a mask, retain n of the 1s and make the rest 0"""
    assert n <= mask.sum()
    m = mask.clone()
    ones_idx = m.nonzero().reshape(1, -1)[0] # get indices of 1s
    idx_to_keep = choose_n(ones_idx, n)
    m[:] = 0
    m[idx_to_keep] = 1
    return m

def do_filter(ll:LabelList, mask):
    """
    Filter images in LabelList where mask == 0
    Mutates `ll`
    """
    global i
    i = 0
    def mask_filter(img):
        global i
        keep = mask[i]
        i += 1
        return keep

    ll.x.filter_by_func(mask_filter)
    i = 0
    ll.y.filter_by_func(mask_filter)
    return ll

def proxy_distr(sample, hardness, total_hard, targets):
    assertIsPerc(sample)
    assertIsPerc(hardness)

    total = len(targets)
    total_easy = total - total_hard

    proxy_size = math.floor(total * sample)
    num_hard = math.floor(proxy_size * hardness)
    num_easy = proxy_size - num_hard

    if num_hard > total_hard:
        # not enough "hard" examples
        print('Desired hardness %f not available, only %f hard out of %f examples. Using hardness=%f instead' %
              (hardness, total_hard, total, float(num_hard) / float(total_hard)))
        num_hard = total_hard

    dataset_hardness = float(total_hard) / float(total)
    if hardness < dataset_hardness:
        print('Warning: Desired hardness=%f less than full dataset hardness=%f' % (hardness, dataset_hardness))
    else:
        print('Sampling using hardness=%f. Original dataset hardness=%f (ignore for loss-based)' % (hardness, dataset_hardness))

    if num_easy > total_easy:
        # not enough "easy" examples
        print('Desired easiness %f not available, only %f easy out of %f examples. Using easiness=%f instead' %
              (1.0 - hardness, total_easy, total, float(num_easy) / float(total_easy)))
        num_hard = total_hard

    return proxy_size, num_hard

def sample_loss_based(ll:LabelList, sample, hardness, preds, targets, loss):
    assertIsPerc(sample)
    assertIsPerc(hardness)

    total = len(targets)
    proxy_size, num_hard = proxy_distr(sample, hardness, math.floor(total * sample * hardness), targets)

    keyList = sorted(enumerate(loss), key=itemgetter(1), reverse=True) # sort loss from highest to lowest
    hard_idx = [x[0] for x in keyList[:num_hard+1]]
    hard_mask = torch.zeros_like(targets)
    hard_mask[hard_idx] = 1
    easy_mask = retain_ones((hard_mask == 0).long(), proxy_size - num_hard)
    mask = easy_mask + hard_mask

    assert ((mask > 1).sum() == 0) # should only contain 1s and 0s
    assert mask.sum() in range(int(proxy_size) - 2, int(proxy_size) + 2) # allow wiggle room for rounding

    return do_filter(ll, mask)

def sample_with_hardness(size, woof, bs, sample, hardness_params):
    hardness = hardness_params['hardness']
    model_dir = hardness_params['hardness_model_dir']
    model, data = load_model(model_dir, size, woof, bs, sample)
    predictions_t, targets_t, loss_t = model.get_preds(ds_type=DatasetType.Train, with_loss=True)
    sample_loss_based(data.train_ds, sample, hardness, predictions_t, targets_t, loss_t)
    return data


import pandas as pd

def get_n_easiest(n):
    df = pd.read_msgpack('pred_df.mp').sort_values('loss', ascending=False)
    return set(df.tail(IMAGENETTE_SIZE - n).paths.values)

IMAGENETTE_SIZE = 12894
def sample_hard_from_disk(size, woof, bs, sample):
    if sample == 1: raise ValueError(f'pointless to sample=100% hardest images')
    easy_tr_paths = get_n_easiest(int(sample * IMAGENETTE_SIZE))
    easy_tr_paths = {os.path.basename(x) for x in easy_tr_paths}
    data = get_data(size, woof, bs, 1., shuffle_train=True,
                    filter_func=lambda x: str(os.path.basename(x)) not in easy_tr_paths)
    print(len(data.train_dl.dataset))
    return data



