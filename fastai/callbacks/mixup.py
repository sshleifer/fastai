"Implements [mixup](https://arxiv.org/abs/1710.09412) training method"
from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback


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
IMAGENETTE_SIZE = 12894


_HOMES = map(Path, ['/home/shleifer/fastai/', '/home/eprokop/fastai/', '/Users/shleifer/fastai-fork/',
                    '/home/paperspace/fastai-fork/'])
for HOME in _HOMES:
    if HOME.exists(): break
PRED_DF_PATH = HOME /'fastai/imagito/pred_df.mp'
WOOF_PRED_DF_PATH = HOME /'fastai/imagito/pred_df_woof.mp'
assert PRED_DF_PATH.exists(), PRED_DF_PATH


def make_hardness_filter_func(hardness_lower_bound, hardness_upper_bound, woof):
    """
    > hardness_bounds = (0., .25)  # top 25% hardest
    > hardness_bounds = (0., 1.) # all
    > hardness_bounds = (.75, 1.)  # top 25 % easiest
    """
    path = PRED_DF_PATH if not woof else WOOF_PRED_DF_PATH
    if (hardness_lower_bound == 0) and (hardness_upper_bound==1.):
        return return_true
    pred_df = pd.read_msgpack(path).sort_values('loss', ascending=False)
    start_idx, end_idx = int(IMAGENETTE_SIZE * hardness_lower_bound), int(
        IMAGENETTE_SIZE * hardness_upper_bound) - 1
    pred_df['path'] = pred_df['paths'].apply(os.path.basename)
    all_paths = set(pred_df['path'].unique())
    paths_to_keep = pred_df.iloc[start_idx: end_idx].path.unique()
    paths_to_toss = all_paths.difference(paths_to_keep)
    filter_func = lambda x: str(os.path.basename(x)) not in paths_to_toss  # ignores val paths
    return filter_func


class CurriculumCallback(LearnerCallback):

    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.original_dl = learn.train_dl

    def on_epoch_begin(self, **kwargs:Any):
        import ipdb; ipdb.set_trace()

    def make_filter_func(self):




class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            new_input = (last_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
        

class MixUpLoss(nn.Module):
    "Adapt the loss function `crit` to go with mixup."
    
    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'): 
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else: 
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output,target[:,0].long()), self.crit(output,target[:,1].long())
            d = (loss1 * target[:,2] + loss2 * (1-target[:,2])).mean()
        else:  d = self.crit(output, target)
        if self.reduction == 'mean': return d.mean()
        elif self.reduction == 'sum':            return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit
