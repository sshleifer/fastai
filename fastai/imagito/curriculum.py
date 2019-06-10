from fastai.imagito.sample_hardness import make_hardness_filter_func
from fastai.imagito.utils import get_data
import numpy as np
from fastai.basic_train import LearnerCallback, Learner
from fastai.torch_core import batch_to_half
from fastai.basic_data import DeviceDataLoader

def make_easy_to_hard_sched(num_epochs):
    phases = num_epochs //2
    return [(.5 , 1)] * phases + [(0.03, .5)] * phases

EASY_HALF = (0.5, 1)
MIDDLE_HALF = (.25, .75)
HARD_HALF = (0.03, 0.53)
RAMP_SCHED = [
    EASY_HALF,
    EASY_HALF,
    EASY_HALF,
    EASY_HALF,
    MIDDLE_HALF,
    EASY_HALF,
    EASY_HALF,
    HARD_HALF,
    EASY_HALF,
    MIDDLE_HALF,
    EASY_HALF,
    EASY_HALF,
    HARD_HALF,
    HARD_HALF,
    MIDDLE_HALF,
    HARD_HALF,
    EASY_HALF,
    HARD_HALF,
    MIDDLE_HALF,
    HARD_HALF,
]

SCHED_TYPES = ['easy_first', 'hard_first', 'ramp', 'reverse_ramp']


class CurriculumCallback(LearnerCallback):
    sets_dl = True  # FIXME: redundant with method name

    def __init__(self, learn, produce_image_list_fn, num_epochs, woof, sched_type='easy_first'):
        # FIXME: try to get learn.num_epochs
        super().__init__(learn)
        #self.original_dl = learn.data.train_dl
        self.sched_type = sched_type
        if self.sched_type == 'easy_first':
            self.sched = make_easy_to_hard_sched(num_epochs)
        elif self.sched_type == 'hard_first':
            self.sched = list(reversed(make_easy_to_hard_sched(num_epochs)))
        elif sched_type == 'ramp':
            assert len(RAMP_SCHED) == num_epochs
            self.sched = RAMP_SCHED
        elif sched_type == 'ramp-reversed':
            self.sched = list(reversed(RAMP_SCHED))
        else:
            raise ValueError(sched_type)

        self.produce_image_list = produce_image_list_fn
        self.woof = woof
        print(self.sched_type, self.sched)
        #self.woof = self

    def set_dl_on_epoch_begin(self, epoch):
        filter_func = make_hardness_filter_func(*self.sched[epoch], self.woof)
        data = self.produce_image_list(filter_func)
        data.add_tfm(batch_to_half)
        return data.train_dl
        #return self.original_dl.filter_by_func(filter_func)

