HUB = 'hardness_upper_bound'
HLB = 'hardness_lower_bound'
GRID_18 = [
    {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.5},
    {'label_smoothing': True, 'lr': 0.003, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.0001, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.001, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.01, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.05, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.1, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.0},
    {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.25},
    {'label_smoothing': False, 'lr': 0.007, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.015, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.02, 'flip_lr_p': 0.5},
    {'label_smoothing': False, 'lr': 0.03, 'flip_lr_p': 0.5},
    {'label_smoothing': True, 'lr': 0.0001, 'flip_lr_p': 0.5},
    {'label_smoothing': True, 'lr': 0.001, 'flip_lr_p': 0.5},
    {'label_smoothing': True, 'lr': 0.01, 'flip_lr_p': 0.5},
    {'label_smoothing': True, 'lr': 0.1, 'flip_lr_p': 0.5},
    {'label_smoothing': True, 'lr': 0.05, 'flip_lr_p': 0.5}
]

G11 = [{'label_smoothing': True, 'lr': 0.003, 'flip_lr_p': 0.5},
       {'label_smoothing': False, 'lr': 0.0001, 'flip_lr_p': 0.5},
       {'label_smoothing': False, 'lr': 0.001, 'flip_lr_p': 0.5},
       {'label_smoothing': False, 'lr': 0.01, 'flip_lr_p': 0.5},
       {'label_smoothing': False, 'lr': 0.05, 'flip_lr_p': 0.5},
       {'label_smoothing': False, 'lr': 0.1, 'flip_lr_p': 0.5},
       {'label_smoothing': True, 'lr': 0.0001, 'flip_lr_p': 0.5},
       {'label_smoothing': True, 'lr': 0.001, 'flip_lr_p': 0.5},
       {'label_smoothing': True, 'lr': 0.01, 'flip_lr_p': 0.5},
       {'label_smoothing': True, 'lr': 0.1, 'flip_lr_p': 0.5},
       {'label_smoothing': True, 'lr': 0.05, 'flip_lr_p': 0.5}]
halfc = list(range(5))


def parse_strat(strat):
    p = {}
    if 'ep' in strat:
        strat, ep = strat.split('ep')
        p['epochs'] = int(ep)
    if 'Classes' in strat:
        classstr, sample = strat.split('Classes')
        classes = {'2': [0, 1], 'Half ': halfc, 'All ': None, 'Other Half ': list(range(5, 10))
                   }[classstr]
        p['classes'] = classes
        p['sample'] = float(sample.strip('-'))
    if 'hard' in strat:
        assert p == {}
        _, lb, ub = strat.split('-')
        p.update({HLB: float(lb), HUB: float(ub)})
    return p


STRAT2PARAMS_V2 = {
    # '2Classes-0.1': {'classes': [0, 1], 'sample': 0.1},
    # '2Classes-0.5': {'classes': [0, 1], 'sample': 0.5},
    # '2Classes-1.0': {'classes': [0, 1], 'sample': 1.0},
    'All Classes-0.1': {'classes': None, 'sample': 0.1},
    'All Classes-0.25': {'classes': None, 'sample': 0.25},
    'All Classes-0.5': {'classes': None, 'sample': 0.5},
    'All Classes-0.7': {'classes': None, 'sample': 0.7},
    'All Classes-1.0': {'classes': None, 'sample': 1.0},
    'All Classes-1.0-ep1': {'epochs': 1, 'classes': None, 'sample': 1.0},
    'All Classes-1.0-ep10': {'epochs': 10, 'classes': None, 'sample': 1.0},
    'All Classes-1.0-ep5': {'epochs': 5, 'classes': None, 'sample': 1.0},
    'Half Classes-0.1': {'classes': [0, 1, 2, 3, 4], 'sample': 0.1},
    'Half Classes-0.25': {'classes': [0, 1, 2, 3, 4], 'sample': 0.25},
    'Half Classes-0.5': {'classes': [0, 1, 2, 3, 4], 'sample': 0.5},
    'Half Classes-0.7': {'classes': [0, 1, 2, 3, 4], 'sample': 0.7},
    'Half Classes-1.0': {'classes': [0, 1, 2, 3, 4], 'sample': 1.0},
    'Other Half Classes-1.0': {'classes': [5, 6, 7, 8, 9], 'sample': 1.0},
    'distillation': {},
    'hard-0.0-0.1': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.1},
    'hard-0.0-0.25': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.25},
    'hard-0.0-0.5': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.5},
    'hard-0.0-0.75': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.75},
    'hard-0.05-0.5': {'hardness_lower_bound': 0.05, 'hardness_upper_bound': 0.5},
    'hard-0.25-1.0': {'hardness_lower_bound': 0.25, 'hardness_upper_bound': 1.0},
    'hard-0.5-1.0': {'hardness_lower_bound': 0.5, 'hardness_upper_bound': 1.0},
    'hard-0.75-1.0': {'hardness_lower_bound': 0.75, 'hardness_upper_bound': 1.0},
    'hard-0.9-1.0': {'hardness_lower_bound': 0.9, 'hardness_upper_bound': 1.0}
}

STRAT2PARAMS_V3 = {
    'hard-0.0-0.1': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.1},
    'hard-0.0-0.25': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.25},
    'hard-0.0-0.5': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.5},
    'hard-0.0-0.75': {'hardness_lower_bound': 0.0, 'hardness_upper_bound': 0.75},
    'hard-0.05-0.5': {'hardness_lower_bound': 0.05, 'hardness_upper_bound': 0.5},
    'hard-0.25-1.0': {'hardness_lower_bound': 0.25, 'hardness_upper_bound': 1.0},
    'hard-0.5-1.0': {'hardness_lower_bound': 0.5, 'hardness_upper_bound': 1.0},
    'hard-0.75-1.0': {'hardness_lower_bound': 0.75, 'hardness_upper_bound': 1.0},
    'hard-0.9-1.0': {'hardness_lower_bound': 0.9, 'hardness_upper_bound': 1.0},
    'Half Classes-0.1': {'classes': [0, 1, 2, 3, 4], 'sample': 0.1},
    'Half Classes-0.25': {'classes': [0, 1, 2, 3, 4], 'sample': 0.25},
    'Half Classes-0.5': {'classes': [0, 1, 2, 3, 4], 'sample': 0.5},
    'Half Classes-0.7': {'classes': [0, 1, 2, 3, 4], 'sample': 0.7},
    'Half Classes-1.0': {'classes': [0, 1, 2, 3, 4], 'sample': 1.0},
}

arches = {'arch': ['xresnet18', 'xresnet101', 'presnet18']}
from sklearn.model_selection import ParameterGrid
from fastai.imagito.utils import update_batch_size

to_grid = lambda p: update_batch_size(ParameterGrid(p))
NEED_TO_RUN_ERIC_BOX_V2 = []


def listier(j):
    return {k: [v] for k, v in j.items()}



for shtuff in GRID_18:
    for _, j in STRAT2PARAMS_V2.items():
        d = j.copy()
        d.update(shtuff)
        d['woof'] = True
        NEED_TO_RUN_ERIC_BOX_V2.append(d)


NEED_TO_RUN_ERIC_BOX_V3 = []
for shtuff in GRID_18:
    for _, j in STRAT2PARAMS_V3.items():
        d = j.copy()
        d.update(shtuff)
        d['woof'] = True
        NEED_TO_RUN_ERIC_BOX_V3.append(d)

NEED_TO_RUN_ERIC_BOX_V4 = []
for arch in ['xresnet18', 'xresnet101']:
    for _, j in STRAT2PARAMS_V2.items():
        d = j.copy()
        d.update(shtuff)
        d['woof'] = True
        NEED_TO_RUN_ERIC_BOX_V4.append(d)
stem_testers = [16, 48]
stem_grid = [{'stem1': x} for x in stem_testers]  + [{'stem2': x} for x in stem_testers] + [{'stem1': 16, 'stem2': 16}]
opt_grid = [{'opt': 'SGD', 'opt': 'RMS'}]

import funcy

NEW_FUN_NO_WOOF = []
for shtuff in stem_grid + opt_grid:
    for _, j in STRAT2PARAMS_V3.items():
        d = j.copy()
        d.update(shtuff)
        NEW_FUN_NO_WOOF.append(d)

C1, C2, C3 = (NEW_FUN_NO_WOOF[:28], NEW_FUN_NO_WOOF[28:56], NEW_FUN_NO_WOOF[56:])

strat2params = {
    'Half Classes-1.0': {'classes': halfc, 'sample': 1.0},
    'Half Classes-0.5': {'classes': halfc, 'sample': .5},
    'All Classes-0.25': {'classes': None, 'sample': .25},
    'Half Classes-0.25': {'classes': halfc, 'sample': .25},
    'All Classes-0.1': {'classes': None, 'sample': .1},

    'hard-0.0-0.75': {HLB: 0.0, HUB: .75},
    'hard-0.0-0.5': {HLB: 0.0, HUB: .5},
    'hard-0.0-0.25': {HLB: 0.0, HUB: .25},
    'hard-0.05-0.5': {HLB: 0.05, HUB: .5},
    'hard-0.25-1.0': {HLB: .25, HUB: 1.},

    'All Classes-1.0-ep10': {'epochs': 10},
    'All Classes-1.0-ep5': {'epochs': 5},
    'All Classes-1.0-ep1': {'epochs': 1},
}

NEED_TO_RUN_ERIC_BOX = [{'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': True, 'lr': 0.003, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.0001, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.001, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.01, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.05, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.1, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.0, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.25, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.007, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.015, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.02, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.03, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': True, 'lr': 0.0001, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': True, 'lr': 0.001, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': True, 'lr': 0.01, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': True, 'lr': 0.1, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': True, 'lr': 0.05, 'flip_lr_p': 0.5, 'epochs': 10},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': True, 'lr': 0.003, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.0001, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.001, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.01, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.05, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.1, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.0, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.25, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.007, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.015, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.02, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.03, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': True, 'lr': 0.0001, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': True, 'lr': 0.001, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': True, 'lr': 0.01, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': True, 'lr': 0.1, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': True, 'lr': 0.05, 'flip_lr_p': 0.5, 'epochs': 5},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': True, 'lr': 0.003, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.0001, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.001, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.01, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.05, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.1, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.0, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.003, 'flip_lr_p': 0.25, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.007, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.015, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.02, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': False, 'lr': 0.03, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': True, 'lr': 0.0001, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': True, 'lr': 0.001, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': True, 'lr': 0.01, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': True, 'lr': 0.1, 'flip_lr_p': 0.5, 'epochs': 1},
                        {'label_smoothing': True, 'lr': 0.05, 'flip_lr_p': 0.5, 'epochs': 1}]

NEED_TO_RUN = [
    {'label_smoothing': False,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.0,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.25,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.007,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.015,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.02,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.03,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 1.0},
    {'label_smoothing': False,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.0,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.25,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.007,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.015,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.02,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.03,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.5},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.75},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.75},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.75},
    {'label_smoothing': True,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.0,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.25,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.007,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.015,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.02,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.03,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': True,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'classes': [0, 1, 2, 3, 4],
     'sample': 0.25},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.0,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.25,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': False,
     'lr': 0.007,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': False,
     'lr': 0.015,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': False,
     'lr': 0.02,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': False,
     'lr': 0.03,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': True,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': True,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': True,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': True,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': True,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'classes': None,
     'sample': 0.1},
    {'label_smoothing': True,
     'lr': 0.003,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.003,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': False,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': False,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': True,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': True,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': True,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': True,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': True,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.0,
     'hardness_upper_bound': 0.25},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.003,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.0,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.003,
     'flip_lr_p': 0.25,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.007,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.015,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.02,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': False,
     'lr': 0.03,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.0001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.001,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.01,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.1,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5},
    {'label_smoothing': True,
     'lr': 0.05,
     'flip_lr_p': 0.5,
     'hardness_lower_bound': 0.05,
     'hardness_upper_bound': 0.5}]


