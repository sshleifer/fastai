from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice, update_batch_size
from fastai.imagito.send_sms import try_send_sms
from fastai.imagito.grid_const import GRID_18, NEED_TO_RUN_ERIC_BOX


FLIP_GRID = [0., .25, .5]
LR_GRID = [0.0001, 0.001, 0.003, 0.01, 0.05, 0.1]
FINER_LR_GRID =[0.007, 0.015, .02, .03]
LABEL_SMOOTHING_GRID = [True, False]
BIG_GRID = {'lr': LR_GRID + FINER_LR_GRID, 'label_smoothing': LABEL_SMOOTHING_GRID,
            'flip_lr_p': FLIP_GRID}
HUB = 'hardness_upper_bound'
HLB = 'hardness_lower_bound'


to_grid = lambda p: update_batch_size(ParameterGrid(p))
def get_18(sampling_dct):
    pg = []
    for k in GRID_18:
        new = k.copy()
        new.update(sampling_dct)
        pg.append(new)
    return pg


def run_many(pg):
    # try_send_sms(f'Starting {len(pg)} experiments: Params\n {pg}')
    failures = []
    for pars in tqdm_nice(pg):
        try:
            main(**pars)
        except Exception as e:
            failures.append(pars)
            print(e)
    try_send_sms(f'Finished experiments: failures: {failures}')

if __name__ == '__main__':
    run_many(NEED_TO_RUN_ERIC_BOX)
