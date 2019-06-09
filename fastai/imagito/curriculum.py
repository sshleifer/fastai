from fastai.imagito.sample_hardness import make_hardness_filter_func
from fastai.imagito.utils import get_data
import numpy as np
from fastai.basic_train import LearnerCallback, Learner
from fastai.torch_core import batch_to_half
from fastai.basic_data import DeviceDataLoader

class CurriculumCallback(LearnerCallback):
    sets_dl = True  # FIXME: redundant with method name

    def __init__(self, learn, produce_image_list_fn, num_epochs, woof):
        # FIXME: try to get learn.num_epochs
        super().__init__(learn)
        #self.original_dl = learn.data.train_dl
        self.produce_image_list = produce_image_list_fn
        step = 1/ num_epochs
        upper_bound = np.arange(1., 0, -step)  # start easy
        self.sched = list(zip(upper_bound - step, upper_bound))
        self.woof = woof
        print(self.sched)
        #self.woof = self

    def set_dl_on_epoch_begin(self, epoch):
        filter_func = make_hardness_filter_func(*self.sched[epoch], self.woof)
        data =  self.produce_image_list(filter_func)
        data.add_tfm(batch_to_half)
        return data.train_dl
        #return self.original_dl.filter_by_func(filter_func)


