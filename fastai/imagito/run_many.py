from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice, update_batch_size


pg = update_batch_size(ParameterGrid({
    # 'lr': lr,
    'bs': [256],
    'size': [128],
    'sample': [1., .5, .25],
    'classes': [None, [0,1,2,3,4]],
    'fp16': [True],
    'epochs': [20,]
}))[11:]

pg2 = update_batch_size(ParameterGrid({
    # 'lr': lr,
    'bs': [256],
    'size': [32, 64, 128, 256],
    'sample': [1.],
    'classes': [[5,6,7,8,9],],
    'fp16': [True],
    'epochs': [20,]
}))
def run_many(pg):
    failures = []
    for pars in tqdm_nice(pg):
        try:
            main(**pars)
        except Exception as e:
            failures.append(pars)
            print(e)
    print(f'failures: {failures}')

if __name__ == '__main__':
    run_many(pg)
    run_many(pg2)
