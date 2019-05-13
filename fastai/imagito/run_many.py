from sklearn.model_selection import ParameterGrid
from imagito.train_imagito import main
from imagito.utils import tqdm_nice

pg = list(ParameterGrid({
        #'lr': lr,
        'size': [32, 64, 128, 256],
        'sample': [1., .5],
        'classes': [None],
}))

pg2 = list(ParameterGrid({
        #'lr': lr,
        'size': [32, 64, 128, 256],
        #'mixup': 0.,
        'sample': [1., .5],
        'classes': [None,],
}))

def run_many(pg):
    for pars in tqdm_nice(pg):
        main(**pars)


if __name__ == '__main__':
    run_many(pg)
