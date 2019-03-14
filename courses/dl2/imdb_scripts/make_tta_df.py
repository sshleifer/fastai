import pandas as pd
from fastai.text import *
from .create_toks import read_texts


def extract_base_part(fname):
    strang = Path(fname).name[:-4]
    if strang.count('_') == 1:
        return strang
    else:
        return '_'.join(strang.split('_')[:-1])


def extract_language(fname):
    strang = Path(fname).name[:-4]
    if strang.count('_') == 1:
        return ''
    else:
        return strang.split('_')[-1]


def make_tta_df(dir_path):
    trn_texts, trn_labels, fnames = read_texts(dir_path / 'test')
    df_tta = pd.DataFrame({'text': trn_texts, 'labels': trn_labels, 'fnames': fnames},
                          # columns=['labels','text']
                          )[['labels', 'text', 'fnames']].sample(frac=1.)
    df_tta['record_id'] = df_tta['fnames'].apply(extract_base_part)
    df_tta['btrans'] = df_tta['fnames'].apply(extract_language)
    return df_tta


def assign_error(df_tta):
    return (df_tta['labels'] - df_tta['yhat']).abs()


def calc_acc(df_tta, yhat_col='yhat', y_col='labels'):
    return (df_tta[yhat_col] > .5) == df_tta[y_col]


def analyze_tta_df(df_tta):
    avg = df_tta.groupby("record_id")[['yhat', 'labels']].mean().reset_index()
    avg['btrans'] = 'avg'
    avg['error'] = assign_error(avg)
    avg['acc'] = calc_acc(avg)
    df_tta['error'] = assign_error(df_tta)
    df_tta['acc'] = calc_acc(df_tta)
    errs = pd.concat([df_tta, avg]).groupby('btrans')[['error', 'acc']].mean()
    return errs, df_tta

def lin_combo_scores(xydf, c1, c2, start=.5):
    res ={}
    for x in np.arange(start, 1.01, .01):
        acc = xydf.assign(yhat=(xydf[c1] * x) + (xydf[c2] * (1 - x))).pipe(calc_acc).mean()
        res[x] = acc
    return pd.Series(res)
