from sklearn.model_selection import ParameterGrid
from .train_imagito import main
from .utils import tqdm_nice

pg = list(ParameterGrid({
        #'lr': lr,
        'size': [32, 64, 128, 256],
        'sample': [1., .5, .25],
        'classes': [None],
}))


def run_many(pg):
    for pars in tqdm_nice(pg):
        main(**pars)


if __name__ == '__main__':
    run_many(pg)
