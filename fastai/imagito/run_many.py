from sklearn.model_selection import ParameterGrid
from fastai.imagito.train_imagito import main
from fastai.imagito.utils import tqdm_nice, update_batch_size
from fastai.imagito.send_sms import try_send_sms


# pg = update_batch_size(ParameterGrid({
#     # 'lr': lr,
#     'bs': [256],
#     'size': [32, 64, 128, 256],
#     'sample': [1., .5, .25],
#     'classes': [None, [0,1,2,3,4]],
#     'fp16': [True],
#     'epochs': [20,]
# }))[11:]

# pg = update_batch_size(ParameterGrid({
#     # 'lr': lr,
#     #'lr': [1e-4, 1e-3, 3e-3, 1e-2, .05, 1e-1],
#     'label_smoothing': [True, False],
#     'size': [128],
#     'bs': [256],
#     'sample': [1., .5, .1],
#     'classes': [None, [0,1,2,3,4], [0,1]],
# }))
#
# pg2 = update_batch_size(ParameterGrid({
#     # 'lr': lr,
#     'arch': ['xresnet34', 'xresnet50', 'presnet34', 'presnet50'],
#     'size': [128],
#     'bs': [256],
#     'sample': [1., .5, .1],
#     'classes': [None, [0,1,2,3,4], [0,1]],
# }))

pg_hardness = update_batch_size(ParameterGrid({
    'lr': [1e-4, 1e-3, 3e-3, 1e-2, .05, 1e-1],
    'label_smoothing': [False],
    'size': [128],
    'bs': [256],
    'sample': [.1, .5, .25],
}))

pg_easy = update_batch_size(ParameterGrid({
    'lr': [1e-4, 1e-3, 3e-3, 1e-2, .05, 1e-1],
    'label_smoothing': [True, False],
    'size': [128],
    'bs': [256],
    'hardness_lower_bound': [.01],
    'hardness_upper_bound': [.25, .5, .7]
}))

pg_32 = update_batch_size(ParameterGrid({
    'lr': [1e-4, 1e-3, 3e-3, 1e-2, .05, 1e-1],
    'label_smoothing': [True, False],
    'size': [32],
    'bs': [512],
    'sample': [.1, .5, .25],
}))

PGS = [pg_hardness, pg_easy, pg_32]
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
    for pgrid in PGS:
        run_many(pgrid)
