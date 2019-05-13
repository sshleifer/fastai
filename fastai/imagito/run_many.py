from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice

pg = list(ParameterGrid({
        #'lr': lr,
        'size': [32, 64, 128, 256],
        'sample': [1., .5, .25],
        'classes': [None],
        'fp16': [True],
}))


def run_many(pg):
    for pars in tqdm_nice(pg):
        main(**pars)


if __name__ == '__main__':
    run_many(pg)
