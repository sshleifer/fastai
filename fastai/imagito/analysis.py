from pathlib import Path
from fastai.imagito.nb_utils import *
from fastai.imagito.utils import *
import pandas as pd


STRAT = 'sampling_strat'
Path.ls = property(lambda self: list(self.iterdir()))
ACCURACY = 'accuracy'
ZACC, DATE = 'z_acc', 'date'
DEFAULT_LR = 0.0030

def safe_concat(*args, **kwargs):
    if 'sort' in kwargs:
        return pd.concat(*args, **kwargs)
    else:
        return pd.concat(*args, sort=False, **kwargs)

def tryfloat(x):
    try: return x.astype(float)
    except Exception: return x


DEFAULT_CONFIG_COLS = ['size', 'label_smoothing', 'lr',]
ALL_DATA_STRAT = 'All Classes-1.0'
pd.DataFrame.e19 = property(lambda df: df[df.epoch == 19])
pd.DataFrame.e9 = property(lambda df: df[df.epoch == 9])

def drop_zero_variance_cols(df):
    keep_col_mask = df.apply(lambda x: x.nunique()) > 1
    return df.loc[:, keep_col_mask]

def read_results(experiment_dir):
    metrics = []
    params = {}
    for subdir in tqdm_nice(experiment_dir.ls):

        try:
            par = pd.Series(pickle_load(subdir/'params.pkl'))
            ts = subdir.name
            m = pd.read_csv(subdir/'metrics.csv').assign(date=ts)
            metrics.append(m)
            clas = par['classes']
            par['classes'] = f'{clas[0]}-{clas[-1]}' if isinstance(par['classes'], list) else '0-10'
            params[ts] = par
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f'{type(e)}: {subdir}')
            continue
    metric_df = safe_concat(metrics, sort=False)
    param_df = pd.DataFrame(params).T.rename_axis('date').sort_index()
    return metric_df, param_df



def combine(metric_df, param_df, min_epochs=5):
    keep_dates = metric_df.groupby('date').max().loc[lambda x: x.epoch >= min_epochs-1].index
    print(len(keep_dates))
    metric_df =  metric_df[metric_df.date.isin(keep_dates)]
    print(f'metric_df: {metric_df.shape}')
    changed_params = drop_zero_variance_cols(param_df)
    print(f'n_experiments: {changed_params.shape}')
    df = metric_df.merge(changed_params.reset_index(), how='left')
    return df




def make9_19_data(df, acc_col):
    ep9 = df[df.epoch==9].set_index(DATE)
    ep19 = df[df.epoch==19].set_index(DATE)
    pl_data = ep9[acc_col].to_frame('ep9').assign(ep19=ep19[acc_col])
    return pl_data


### Tables for Milestone
X_COL = "Proxy Boost (SD)"
Y_COL = "Target Boost (SD)"
TIT_COL = 'Changeset'
CAT_NAME = 'Proxy Strategy'
posc, allc = 'Positive Changes','All Changes'

ls_mask = lambda df: df.label_smoothing

def make_cmb(df):
    gb_lst = ['lr', 'size']
    tab_1 = df[(df.epoch==19) & (df[STRAT]=='All Classes-0.5')].loc[ls_mask].groupby(gb_lst)['z_acc_epoch'].median()
    tab_2 = df[(df.epoch==19) & (df[STRAT]=='All Classes-1.0')].loc[ls_mask].groupby(gb_lst)['z_acc_epoch'].median()
    tab_3 = df[(df.epoch==9) & (df[STRAT]=='All Classes-1.0')].loc[ls_mask].groupby(gb_lst)['z_acc_epoch'].median()
    tab_4 = df[(df.epoch==19) & (df[STRAT]=='Half Classes-1.0')].loc[ls_mask].groupby(gb_lst)['z_acc_epoch'].median()
    NAMES = ("50% of Examples", "50% of Epochs", 'Full_Samples', '50% of Classes')
    cmb = tab_1.to_frame(NAMES[0])
    cmb[NAMES[1]] = tab_3
    cmb[NAMES[2]] = tab_2
    cmb[NAMES[3]] = tab_4
    return cmb

def make_cor_tab(df, tab_2):
    _gb = [STRAT, 'lr', 'size']
    best_pars = df[(df.epoch == 19)].groupby(_gb)['z_acc_epoch'].median().unstack(
        level=[1, 2]).idxmax(1)
    pgb = df[(df.epoch == 19)].loc[ls_mask].groupby(_gb)
    all_proxy = pgb['z_acc_epoch'].median().reset_index(level=0)
    all_proxy[Y_COL] = tab_2

    n_experiments = df[(df.epoch == 19)].loc[ls_mask][STRAT].value_counts()

    all_coors = all_proxy.groupby(STRAT).apply(lambda x: x[Y_COL].corr(x['z_acc_epoch']))
    all_pos_coors = all_proxy[all_proxy['z_acc_epoch'] > 0].groupby(STRAT).apply(
        lambda x: x[Y_COL].corr(x['z_acc_epoch']))
    cor_tab = all_coors.to_frame(allc).join(all_pos_coors.to_frame(posc)).round(2).pipe(
        blind_descending_sort)
    cor_tab['Best Params'] = best_pars.dropna().apply(lambda x: f'lr={x[0]}, size={x[1]}')
    cor_tab['N Experiments'] = n_experiments
    return cor_tab

import seaborn as sns
sns.set(color_codes=True)


def make_change_scatters(df):
    cmb = make_cmb(df)
    stk = cmb.stack().reset_index(level=2).rename(columns={'level_2': CAT_NAME, 0: X_COL })
    stk = stk[(stk[CAT_NAME] != 'Full_Samples')]
    stk[Y_COL] = cmb['Full_Samples']
    stk_zoom = stk[stk[X_COL] > 0]

    stk_zoom[TIT_COL] = posc
    stk[TIT_COL] = allc
    pl_data = safe_concat([stk, stk_zoom], sort=False)
    fg = sns.lmplot(data=pl_data, x=X_COL, y=Y_COL, hue=CAT_NAME, legend_out=False,
                    markers=["o", "x", "d"], palette="Set1", ci=70, col=TIT_COL, sharex=False,
                    col_order=[allc, posc])
    fg.axes[0][0].set_ylim(-3,1.5)
    fg.axes[0][1].set_xlim(0,1.5)
    fg.axes[0][1].set_ylim(-3,1.5)
    ### Accompanying table
    c1 = stk.groupby(CAT_NAME).corr()[Y_COL].loc[lambda x: x != 1].round(2).to_frame(allc)
    c2 = stk_zoom.groupby(CAT_NAME).corr()[Y_COL].loc[lambda x: x != 1].round(2).to_frame(posc)
    tab = pd.concat([c1, c2], axis=1)
    tab.index = tab.index.get_level_values(0)
    return fg, tab#.pipe(blind_descending_sort)


