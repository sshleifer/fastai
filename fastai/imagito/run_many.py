from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice, update_batch_size
from fastai.imagito.send_sms import try_send_sms
from fastai.imagito.grid_const import GRID_18, NEED_TO_RUN_ERIC_BOX_V2


FLIP_GRID = [0., .25, .5]
LR_GRID = [0.0001, 0.001, 0.003, 0.01, 0.05, 0.1]
FINER_LR_GRID =[0.007, 0.015, .02, .03]
LABEL_SMOOTHING_GRID = [True, False]
BIG_GRID = {'lr': LR_GRID + FINER_LR_GRID, 'label_smoothing': LABEL_SMOOTHING_GRID,
            'flip_lr_p': FLIP_GRID}
HUB = 'hardness_upper_bound'
HLB = 'hardness_lower_bound'


def get_18(sampling_dct):

    pg = []
    for k in GRID_18:
        new = k.copy()
        new.update(sampling_dct)
        pg.append(new)
    return pg

BASE = {
    'lr': FINER_LR_GRID,
}
OPTS = ['rms', 'sgd']
MOM_GRID = []


HUB_GRID = [.1, .75, .5, .25]
HLB_GRID = [.9, .75, .5, .25]
SAMPLE_GRID = [1., .7, .5, .25]
a = {HUB: HUB_GRID}
b = {HLB: HLB_GRID}  # easy
c = {'sample': SAMPLE_GRID}
#d = {HUB: [.01], HLB: [.75]}

pgs = []
for extra in [a, b,c]:
    p = BASE.copy()
    p.update(extra)
    pg = update_batch_size(ParameterGrid(p))
    pgs.extend(pg)

def run_many(pg):
    # try_send_sms(f'Starting {len(pg)} experiments: Params\n {pg}')
    failures = []
    for pars in tqdm_nice(pg):
        try:
            main(**pars)
        except Exception as e:
            failures.append(pars)
            print(e, pars)
    try_send_sms(f'Finished experiments: failures: {failures}')

PGS = [pgs]
if __name__ == '__main__':
    run_many(reversed(NEED_TO_RUN_ERIC_BOX_V2[150:]))
    #for pgrid in PGS: run_many(pgrid)
