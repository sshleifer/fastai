from fastai.imagito.analysis import *





def _cull_data_features(df, ref_epoch, acc_col):
    legal_df = df[df['epoch'] <= ref_epoch - 1].set_index(DATE)
    legal_df.groupby(())
    ep19 = df.e19.set_index(DATE)
    #pl = ep9.rename(columns={acc_col: 'proxy'}).assign(targ=ep19[acc_col])
    #return pl

def make_ts_features(df, agg_col=ACCURACY):
    ts = df.full_train.set_index([DATE, HOSTCOL, 'epoch'])[agg_col].unstack().T
    ts_features = pd.DataFrame(
        dict(diffs=ts.diff().stack(level=[DATE, HOSTCOL]),
             diffs12=ts.diff(1).shift(1).stack(level=[DATE, HOSTCOL]),
             pct1=ts.pct_change(1).stack(level=[DATE, HOSTCOL]),
             pct2=ts.pct_change(2).stack(level=[DATE, HOSTCOL]), )
    )
EPOCH = 'epoch'
def rl_experiment(df, order_col=DATE):
    bmdf = df.bm_strat.ds_woof.full_train.s128
    bdate =bmdf[bmdf[ACCURACY] == bmdf[ACCURACY].max()][DATE].iloc[0]
    print(f'correct: {bmdf[ACCURACY].max()},epochs: {bmdf.shape[0]} ')
    dates = bmdf[order_col].unique()
    run = bmdf[bmdf[order_col] < dates[10]]
    for d in dates[10:]:
        #best_d_so_far = run.exp_df[ACCURACY].idxmax()
        best_traj = run.groupby(EPOCH)[ACCURACY].max()
        print(best_traj)
        slc = bmdf[bmdf[order_col] == d]
        current = slc.set_index(EPOCH)[ACCURACY]
        for k,v in  current.items():
            if k < 5: continue
            if (v < (best_traj[k] - .05)):
                print(f'break at {k}')
                break
        #print(f'{run.shape[0]}')
        run = pd.concat([run, slc[slc[EPOCH] <= k]], sort=False)
        print(f'{run.shape[0]}, {run[ACCURACY].max()}')

    return best_traj




