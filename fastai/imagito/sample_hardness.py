from fastai.imagito.utils import *
import pandas as pd
IMAGENETTE_SIZE = 12894
from pathlib import Path


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
    filter_func = lambda x, y: str(os.path.basename(x)) not in paths_to_toss  # ignores val paths
    return filter_func
