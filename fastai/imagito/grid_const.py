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
}
