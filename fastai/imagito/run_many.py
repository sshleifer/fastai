from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice, update_batch_size

pg = update_batch_size(ParameterGrid({
    # 'lr': lr,
    'bs': [256],
    'size': [32, 64, 128, 256],
    'sample': [1., .5, .25],
    'classes': [None, [0,1,2,3,4]],
    'fp16': [True],
    'epochs': [20,]
}))

pg2 = update_batch_size(ParameterGrid({
    # 'lr': lr,
    'bs': [256],
    'size': [32, 64, 128, 256],
    'sample': [1., .5, .25],
    'classes': [[5,6,7,8,9],],
    'fp16': [True],
    'epochs': [10,]
}))

params_jeremy = [{
     'lr': 3e-3,
     'bs': 64,
     'size': 128,
     'fp16': True,
     'epochs': 5
 }, {
     'lr': 1e-2,
     'bs': 64,
     'size': 192,
     'fp16': False,
     'epochs': 5
 }, {
     'lr': 1e-2,
     'bs': 64,
     'size': 256,
     'fp16': False,
     'epochs': 5
 }, {
    'lr': 3e-3,
    'bs': 128,
    'size': 128,
    'fp16': False,
    'epochs': 20
}]


def run_many(pg):

    for pars in tqdm_nice(pg):
        main(**pars)


if __name__ == '__main__':
    run_many(params_jeremy)
    # run_many(pg)
    # run_many(pg2)
