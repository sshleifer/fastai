import pandas as pd
from pathlib import Path
from utils import *
from sklearn.linear_model import LassoCV
from sklearn.metrics import *
from fastai.imagito.nb_utils import *

all_data_strat = '0-10-1.0'
STRAT = 'sampling_strat'
def run_grouped_regs(max_acc, FEAT = 'z_acc'):
    targ = max_acc[max_acc[STRAT] == all_data_strat].set_index('size')[FEAT]
    def run_reg(grp):
        xydf = grp.set_index('size')[[FEAT]].assign(y=targ)
        fnames, targ_name = [FEAT], 'y'
        clf = LassoCV(cv=3)
        clf.fit(xydf[fnames], xydf[targ_name])
        coefs = zip_to_series(fnames, clf.coef_)
        coefs.loc['r2'] = r2_score(xydf['y'], clf.predict(xydf[fnames]))
        return coefs
    return max_acc.groupby(STRAT).apply(run_reg).pipe(blind_descending_sort).round(3)
