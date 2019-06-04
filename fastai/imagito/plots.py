# ycol = 'KT Pos Pval'

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fastai.imagito.analysis import *
from sklearn.linear_model import *


def get_line(cti, x, y):
    cti['x2'] = cti[x] ** 2
    cti['x3'] = cti[x] ** 3

    clf = LassoCV().fit(cti[[x, 'x2', 'x3']], cti[y])
    xp = np.linspace(0, 1.01)
    line = (xp, clf.predict(np.array([xp, xp ** 2, xp ** 3]).T))
    return line

def drilldown_plot(mg, strat, hue='woof', xcol=ACCURACY, y=TACC, **kwargs):
    pld = mg[mg[STRAT]==strat]

    fg = sns.lmplot(xcol, y, pld, truncate=True, hue=hue, **kwargs)
    for ax in fg.axes.flat:
        ax.set_xlim(pld[xcol].min() - .02, pld[xcol].max() + .02)
        ax.set_ylim(pld[y].min() - .02, pld[y].max() + .02)
        #xcol.set_title(f'Corr: {pld[y2].corr(pld[ACCURACY]):.3f}')
    return fg


M3 = { 'EP', 'Hard Examples', 'Easy Examples', 'Hard Examples (*)'}



def make_scatter(cti, strats, hue=META_STRAT, x ='Relative Cost', y ='r2', yerr=0.1, fit_reg=False):
    """Input data like fastai/imagito/cti.mp"""
    #sns.set_style("whitegrid")
    pld = cti[cti[hue].isin(strats)]
    fg = sns.lmplot(x, y, pld,  # col_wrap=3,
                    palette="Set1",
                    hue=hue,  # aspect=0.4,
                    hue_order=strats,
                    fit_reg=fit_reg,
                    )

    # TODO add the actual confidence values
    fg.ax.errorbar(x=pld[x].values, y=pld[y].values, fmt='none', yerr=yerr, capsize=2.5)

    line_points = get_line(cti, x, y)
    for a in fg.axes.flat:
        a.set_xlim(0, .8)
        a.set_ylim(cti[y].min() - .02, 1.)
        a.plot(*line_points, alpha=0.3, c='black')
