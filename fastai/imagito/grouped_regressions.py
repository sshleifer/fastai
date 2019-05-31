import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from fastai.imagito.analysis import STRAT, DEFAULT_CONFIG_COLS, ALL_DATA_STRAT
from fastai.imagito.nb_utils import zscore, zip_to_series


def make_med_acc_table(df):
    gb_cols = ['date', 'bs', 'label_smoothing', 'classes', 'epochs', 'fp16', 'lr', 'sample', 'size', STRAT]
    max_acc = df.groupby(gb_cols)['accuracy'].max().reset_index()#.pipe(drop_zero_variance_cols)
    # max_acc['z_acc'] = max_acc.groupby(zby)['accuracy'].transform(zscore)
    # med_acc = max_acc.groupby((zby, 'size', 'lr', ))['z_acc'].median().reset_index()
    return max_acc


def regress_aligned_pairs(acc_table, proxy_strat):
    # find all configs that were run for proxy and also run for target.
    # group both into one row per config
    agg_df = lambda df: df.groupby(DEFAULT_CONFIG_COLS)['accuracy'].median()
    y = acc_table[acc_table[STRAT] == ALL_DATA_STRAT].pipe(agg_df).to_frame('y')
    x = acc_table[acc_table[STRAT] == proxy_strat].pipe(agg_df).to_frame('X')
    xy = y.join(x, how='inner').pipe(zscore)
    if xy.shape[0] <= 1 or xy['X'].isnull().all():
        return pd.Series({'N_configs': xy.shape[0], STRAT: proxy_strat})
    clf = LinearRegression().fit(xy[['X']], xy['y'])
    coefs = zip_to_series(['X'], clf.coef_)
    coefs.loc['r2'] = r2_score(xy['y'], clf.predict(xy[['X']]))
    coefs.loc['N_configs'] = xy.shape[0]
    coefs.loc[STRAT] = proxy_strat
    return coefs


def run_grouped_regs(df):
    acc_table = make_med_acc_table(df)
    all_strats = acc_table[STRAT].value_counts().index.drop(ALL_DATA_STRAT)
    reg_results = pd.DataFrame([regress_aligned_pairs(acc_table, strat) for strat in all_strats])
    reg_results = reg_results.round(2).loc[lambda x: x['N_configs'] > 2.]
    reg_results['N_configs'] = reg_results['N_configs'].astype(int)
    return reg_results.rename(columns={'X': 'coeff'})
