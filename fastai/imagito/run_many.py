from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice, update_batch_size
from fastai.imagito.send_sms import try_send_sms



FLIP_GRID = [0., .25, .5]
LR_GRID = [0.0001, 0.001, 0.003, 0.01, 0.05, 0.1]
a = {'epochs': [10]}
b = {'epochs': [5]}
c = {'epochs': [1]}
#c = {'sample': [1., .7, .5, .25]}

pgs = []

BASE = {
    'size': [128],
    'bs': [256],
    'flip_lr_p': FLIP_GRID,
}
for extra in [a, b, c]:
    p = BASE.copy()
    p.update(extra)
    pg = update_batch_size(ParameterGrid(p))
    pgs.extend(pg)


BASE2 = {
    'size': [128],
    'bs': [256],
    #'flip_lr_p': FLIP_GRID,
    'lr': LR_GRID,
}

for extra in [b, c]:
    p = BASE2.copy()
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
            print(e)
    try_send_sms(f'Finished experiments: failures: {failures}')

PGS = [pgs]
if __name__ == '__main__':
    for pgrid in PGS:
        run_many(pgrid)
