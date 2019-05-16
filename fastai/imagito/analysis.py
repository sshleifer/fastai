import pandas as pd
from pathlib import Path
from utils import *
from sklearn.linear_model import LassoCV
from sklearn.metrics import *
from fastai.imagito.nb_utils import *
from fastai.imagito.utils import *

all_data_strat = '0-10-1.0'
STRAT = 'sampling_strat'
def tryfloat(x):
    try: return x.astype(float)
    except Exception: return x
def run_grouped_regs(max_acc, compare_col ='z_acc', algo_chg_col='size'):
    targ = max_acc[max_acc[STRAT] == all_data_strat].set_index(algo_chg_col)[compare_col]
    def run_reg(grp):
        xydf = grp.set_index(algo_chg_col)[[compare_col]].assign(y=targ)
        fnames, targ_name = [compare_col], 'y'
        clf = LassoCV(cv=3)
        clf.fit(xydf[fnames], xydf[targ_name])
        coefs = zip_to_series(fnames, clf.coef_)
        coefs.loc['r2'] = r2_score(xydf['y'], clf.predict(xydf[fnames]))
        return coefs
    return max_acc.groupby(STRAT).apply(run_reg).pipe(blind_descending_sort).round(3)


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
        except FileNotFoundError:
            print(f'err: {subdir}')
            continue
    metric_df = pd.concat(metrics)
    param_df = pd.DataFrame(params).T.rename_axis('date')
    return metric_df, param_df


def combine(metric_df, param_df):
    keep_dates = metric_df.groupby('date').max().loc[lambda x: x.epoch > 4].index
    print(len(keep_dates))
    metric_df =  metric_df[metric_df.date.isin(keep_dates)]
    print(f'metric_df: {metric_df.shape}')
    changed_params = drop_zero_variance_cols(param_df)
    print(f'{changed_params.shape}')
    df = metric_df.merge(changed_params.reset_index(), how='left')
    return df
