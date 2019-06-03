from pathlib import Path
from fastai.imagito.nb_utils import *
from fastai.imagito.utils import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from fastai.imagito.nb_utils import zscore, zip_to_series
from scipy.stats import kendalltau

Z_ACC_EPOCH = 'z_acc_epoch'

STRAT = 'sampling_strat'
Path.ls = property(lambda self: list(self.iterdir()))
ACCURACY = 'accuracy'
ZACC, DATE = 'z_acc', 'date'
NCONFIGS = 'N_configs'
HOSTCOL = 'hostname'
DEFAULT_LR = 0.0030
DEFAULT_CONFIG_COLS = ['size', 'label_smoothing', 'lr', 'flip_lr_p', 'woof', 'arch']
ALL_DATA_STRAT = 'All Classes-1.0'
XR50 = 'xresnet_50'
# New properties for DF
pd.DataFrame.e19 = property(lambda df: df[df.epoch == 19])
pd.DataFrame.e9 = property(lambda df: df[df.epoch == 9])
pd.DataFrame.s128 = property(lambda df: df[df['size'] == 128])
pd.DataFrame.ls_true = property(lambda df: df[df.label_smoothing == True])
pd.DataFrame.full_train = property(lambda df: df[df['epochs'] == 20])
pd.DataFrame.bm_strat = property(lambda df: df[df[STRAT] == ALL_DATA_STRAT])
pd.DataFrame.drop_bm_strat = property(lambda df: df[df[STRAT] != ALL_DATA_STRAT])
pd.DataFrame.ds_woof = property(lambda df: df[df['woof'] == 1])
pd.DataFrame.imagenette = property(lambda df: df[df['woof'] == 0])
pd.DataFrame.n_configs_geq_3 = property(lambda df: df[df[NCONFIGS] >= 3])
pd.DataFrame.n_configs_geq_4 = property(lambda df: df[df[NCONFIGS] >= 4])
pd.DataFrame.n_configs_geq_18 = property(lambda df: df[df[NCONFIGS] >= 18])
pd.DataFrame.n_configs_geq_cut = property(lambda df: df[df[NCONFIGS] >= 25])
pd.DataFrame.just_xr50 = property(lambda df: df[df['arch'] == XR50])

WOOF_SIZE = 12454


def best_epoch(df):
    gb = df.groupby('date')
    exp_df = gb.first().assign(
        accuracy=gb.accuracy.max(), epochs_run=gb.epoch.max() + 1, cost=gb['seconds'].sum())
    return exp_df[exp_df['epochs_run'] == exp_df['epochs']]


pd.DataFrame.exp_df = property(best_epoch)


def safe_concat(*args, **kwargs):
    if 'sort' in kwargs:
        return pd.concat(*args, **kwargs)
    else:
        return pd.concat(*args, sort=False, **kwargs)


def check_killed_early(df, cutoff=1):
    actual_epochs = (df.groupby(DATE).epoch.max() + 1).to_frame('epochs_run').assign(
        intended=df.groupby(DATE).epochs.first())
    msk = actual_epochs['intended'] - actual_epochs['epochs_run']
    print(f'{(msk >= cutoff).sum()}/ {msk.shape[0]} killed early. They are tossed in exp_df')
    killed_early = actual_epochs.loc[msk >= cutoff]
    return killed_early


def tryfloat(x):
    try:
        return x.astype(float)
    except Exception:
        return x


def drop_zero_variance_cols(df):
    keep_col_mask = df.apply(lambda x: x.nunique()) > 1
    return df.loc[:, keep_col_mask]


def read_results(experiment_dir):
    metrics = []
    params = []
    for subdir in tqdm_nice(experiment_dir.ls):
        try:
            par = pd.Series(pickle_load(subdir / 'params.pkl'))
            splat = subdir.name.split('_')
            host = splat[1] if '_' in subdir.name else np.nan
            ts = splat[0]
            d = '2019-05-31-19:11:27'

            m = pd.read_csv(subdir / 'metrics.csv').assign(date=ts, hostname=host)
            metrics.append(m)
            clas = par['classes']
            par['classes'] = f'{clas[0]}-{clas[-1]}' if isinstance(par['classes'], list) else '0-10'
            par['date'], par['hostname'] = ts, host
            params.append(par)
            # if ts == d: import ipdb; ipdb.set_trace()
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f'{type(e)}: {subdir}')
            continue
    metric_df = safe_concat(metrics, sort=False).fillna({'hostname': ''})
    param_df = pd.DataFrame(params).fillna({'hostname': ''})
    return metric_df, param_df


def combine(metric_df, param_df):
    df = metric_df.merge(param_df.reset_index(), how='left',
                         on=['date', 'hostname'])
    return df


def preprocess_and_assign_strat(df):
    DS_PATH = 'ds_path'
    df['_hardness_str'] = df.apply(
        lambda r: f'hard-{r.hardness_lower_bound}-{r.hardness_upper_bound}', 1)
    DEFAULT_HARD_STR = 'hard-0.0-1.0'
    df['classes'] = df['classes'].replace(
        {'0-10': 'All Classes', '0-4': 'Half Classes', '5-9': 'Other Half Classes',
         '0-1': '2Classes'})
    df['seconds'] = df['time'].str.split(':').apply(lambda x: 60 * int(x[0]) + int(x[1]))
    df[STRAT] = df['classes'] + '-' + df['sample'].astype(str)
    df.loc[df[DS_PATH] != 'imagenette', STRAT] = 'distillation'
    df.loc[df['_hardness_str'] != DEFAULT_HARD_STR, STRAT] = df.loc[
        df['_hardness_str'] != DEFAULT_HARD_STR, '_hardness_str']
    emsk = df['epochs'] != 20
    df.loc[emsk, STRAT] = df.loc[emsk, STRAT] + '-ep' + df.loc[emsk, 'epochs'].astype(str)
    df['woof'] = df['woof'].replace({True: 1})
    # Path Hardness bug
    msk = ((df['woof']) & (df['n_train'] == WOOF_SIZE) & (df[STRAT].str.startswith('hard')))
    df.loc[msk, STRAT] = ALL_DATA_STRAT
    return df


def find_overlapping_configs(df, strat, baseline=ALL_DATA_STRAT, config_cols=DEFAULT_CONFIG_COLS):
    gb_size = df.groupby([STRAT] + config_cols).size().unstack(config_cols)
    return gb_size.loc[[strat, baseline]].dropna(axis=1, how='all')


def make_9_19_data_fairer(df, ref_epoch=10, gb_cols=DEFAULT_CONFIG_COLS):
    df = df.bm_strat
    targ = df.e19.groupby(gb_cols)[ACCURACY].mean()
    prox = df[(df['epochs'] == ref_epoch) & (df['epoch'] == ref_epoch - 1)].groupby(gb_cols)[
        ACCURACY].mean()  # .sort_index()
    pl = targ.to_frame('targ').assign(proxy=prox).dropna()
    corr = pl.corr().loc['targ', 'proxy']
    ax = pl.plot.scatter(x='proxy', y='targ')
    ax.set_title(f'corr={corr:.2f}, n={pl.shape[0]}')
    return ax


def make9_19_data(df, acc_col=ACCURACY, ref_epoch=10):
    ep9 = df[df['epoch'] == ref_epoch - 1].set_index(DATE)
    ep19 = df.e19.set_index(DATE)
    pl = ep9.rename(columns={acc_col: 'proxy'}).assign(targ=ep19[acc_col])
    corr = pl.corr().loc['targ', 'proxy']
    ax = pl.plot.scatter(x='proxy', y='targ')
    ax.set_title(f'corr={corr:.2f}, n={pl.shape[0]}')
    return pl


### Tables for Milestone
X_COL = "Proxy Boost (SD)"
Y_COL = "Target Boost (SD)"
TIT_COL = 'Changeset'
CAT_NAME = 'Proxy Strategy'
posc, allc = 'Positive Changes', 'All Changes'


def make_cmb(pdf, gb_lst=DEFAULT_CONFIG_COLS, agg_col=ACCURACY):
    agger = lambda df: df.groupby(gb_lst)[agg_col].median()

    tab_1 = pdf[(pdf.epoch == 19) & (pdf[STRAT] == 'All Classes-0.5')].pipe(agger)
    tab_2 = pdf[(pdf.epoch == 19) & (pdf[STRAT] == ALL_DATA_STRAT)].pipe(agger)
    tab_3 = pdf[(pdf.epoch == 9) & (pdf[STRAT] == ALL_DATA_STRAT)].pipe(agger)
    tab_4 = pdf[(pdf.epoch == 19) & (pdf[STRAT] == 'Half Classes-1.0')].pipe(agger)
    NAMES = ("50% of Examples", "50% of Epochs", 'Full_Samples', '50% of Classes')
    cmb = tab_1.to_frame(NAMES[0])
    cmb[NAMES[1]] = tab_3
    cmb[NAMES[2]] = tab_2
    cmb[NAMES[3]] = tab_4
    return cmb, tab_2


def pacc_at_n(grp, n=2, nruns=1, agg_col=ACCURACY):
    grp = grp.dropna(subset=[agg_col, Y_COL])
    if grp.shape[0] < n:
        return np.nan
    x = grp[agg_col]
    y = grp[Y_COL]
    stats = []
    #print('here')
    for _ in range(nruns):
        yri = y.reset_index().sample(n)
        ilocs = yri.index
        correct = x.reset_index().iloc[ilocs][agg_col].idxmax() == yri[Y_COL].idxmax()
        stats.append(correct)
    return np.mean(stats)


def get_stats(grp, agg_col=ACCURACY):
    """"""
    grp = grp.dropna(subset=[agg_col, Y_COL])
    x = grp[agg_col]
    y = grp[Y_COL]
    tau, pval = kendalltau(x.rank(), y.rank())
    return pval


def make_cor_tab(exp_df, _gb=[STRAT] + DEFAULT_CONFIG_COLS, agg_col=ACCURACY):
    pgb = exp_df.groupby(_gb)

    all_proxy = pgb[agg_col].median().reset_index(level=0)
    all_proxy[Y_COL] = exp_df.bm_strat.groupby(DEFAULT_CONFIG_COLS)[agg_col].median()
    strat_mean = 'strat_mean'
    all_proxy[strat_mean] = all_proxy.groupby(STRAT)[agg_col].transform('median')
    gb_corr = lambda df: df.groupby(STRAT).apply(lambda x: x[Y_COL].corr(x[agg_col]))
    all_coors = gb_corr(all_proxy)
    # print(all_proxy.loc[all_proxy[agg_col] > all_proxy[strat_mean]].shape[0] / all_proxy.shape[0])
    pos_proxy = all_proxy.loc[all_proxy[agg_col] > all_proxy[strat_mean]]
    all_pos_coors = gb_corr(pos_proxy)
    cor_tab = all_coors.to_frame(allc).join(all_pos_coors.to_frame(posc)).round(3)
    cor_tab['KT Pval'] = all_proxy.groupby(STRAT).apply(
        lambda x: get_stats(x, agg_col=agg_col)).round(3)
    cor_tab['KT Pos Pval'] = pos_proxy.groupby(STRAT).apply(
        lambda x: get_stats(x, agg_col=agg_col)).round(3)
    cor_tab['KT Pos Pval'] = pos_proxy.groupby(STRAT).apply(
        lambda x: get_stats(x, agg_col=agg_col)).round(3)
    # cor_tab['Pacc2'] = pos_proxy.groupby(STRAT).apply(
    #     lambda x: pacc_at_n(x, nruns=50, agg_col=agg_col))
    # cor_tab['Pacc2b'] = pos_proxy.groupby(STRAT).apply(
    #     lambda x: pacc_at_n(x, nruns=50, agg_col=agg_col))

    agger = lambda df: df.groupby(_gb)[agg_col].median()
    _res_df = exp_df.pipe(agger).unstack(level=DEFAULT_CONFIG_COLS)
    bm_perf = _res_df.loc[ALL_DATA_STRAT]
    cor_tab['Best on Proxy'], cor_tab['Max Proxy Acc'] = _res_df.idxmax(1), _res_df.max(1).round(3)
    cor_tab['BM Acc for Pars'] = [bm_perf.loc[x] for x in cor_tab['Best on Proxy'].values.tolist()]
    cor_tab['Proxy Truth Rank'] = _res_df.rank(1, ascending=False)[bm_perf.idxmax()]
    cor_tab['Regret'] = bm_perf.max() - cor_tab['BM Acc for Pars']
    tab = cor_tab.join(run_grouped_regs(exp_df))
    tab['Seconds'] = exp_df.s128.just_xr50.groupby(STRAT)['cost'].median()
    return tab

def regress_aligned_pairs(exp_df, proxy_strat):
    # find all configs that were run for proxy and also run for target.
    # group both into one row per config
    agg_df = lambda df: df.groupby(DEFAULT_CONFIG_COLS)['accuracy'].median()
    y = exp_df[exp_df[STRAT] == ALL_DATA_STRAT].pipe(agg_df).to_frame('y')
    x = exp_df[exp_df[STRAT] == proxy_strat].pipe(agg_df).to_frame('X')
    xy = y.join(x, how='inner').pipe(zscore)
    if xy.shape[0] <= 1 or xy['X'].isnull().all():
        return pd.Series({NCONFIGS: xy.shape[0], STRAT: proxy_strat})
    clf = LinearRegression().fit(xy[['X']], xy['y'])
    coefs = zip_to_series(['X'], clf.coef_)
    coefs.loc['r2'] = r2_score(xy['y'], clf.predict(xy[['X']]))
    coefs.loc[NCONFIGS] = xy.shape[0]
    coefs.loc[STRAT] = proxy_strat
    return coefs


def run_grouped_regs(exp_df):
    all_strats = exp_df[STRAT].unique()
    reg_results = pd.DataFrame([regress_aligned_pairs(exp_df, strat) for strat in all_strats])
    reg_results = reg_results.round(2)
    reg_results['N_configs'] = reg_results['N_configs'].astype(int)
    return reg_results.rename(columns={'X': 'coeff'}).set_index(STRAT)


import seaborn as sns

sns.set(color_codes=True)


def make_change_scatters(df):
    "BROKEN"
    cmb, _ = make_cmb(df)
    stk = cmb.stack().reset_index(level=2).rename(columns={'level_2': CAT_NAME, 0: X_COL})
    stk = stk[(stk[CAT_NAME] != 'Full_Samples')]
    stk[Y_COL] = cmb['Full_Samples']
    stk_zoom = stk[stk[X_COL] > 0]

    stk_zoom[TIT_COL] = posc
    stk[TIT_COL] = allc
    pl_data = safe_concat([stk, stk_zoom], sort=False)
    fg = sns.lmplot(data=pl_data, x=X_COL, y=Y_COL, hue=CAT_NAME, legend_out=False,
                    markers=["o", "x", "d"], palette="Set1", ci=70, col=TIT_COL, sharex=False,
                    col_order=[allc, posc])
    fg.axes[0][0].set_ylim(-3, 1.5)
    fg.axes[0][1].set_xlim(0, 1.5)
    fg.axes[0][1].set_ylim(-3, 1.5)
    ### Accompanying table
    c1 = stk.groupby(CAT_NAME).corr()[Y_COL].loc[lambda x: x != 1].round(2).to_frame(allc)
    c2 = stk_zoom.groupby(CAT_NAME).corr()[Y_COL].loc[lambda x: x != 1].round(2).to_frame(posc)
    tab = pd.concat([c1, c2], axis=1)
    tab.index = tab.index.get_level_values(0)
    return fg, tab  # .pipe(blind_descending_sort)
