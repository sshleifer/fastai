from __future__ import division

from typing import Iterable

import funcy
import numpy as np
import os
import pandas as pd
import re
import time
#from eli5.sklearn import PermutationImportance
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import *

#from util.pd_utils import select_subset, to_iterable, is_iterable
import logging
logger = logging.getLogger(__name__)
from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()
tqdm_nice = tqdm_notebook if in_notebook() else tqdm
#TODO use this throughout!


def get_date_str(seconds=True):
    if seconds:
        return time.strftime('%Y-%m-%d-%H%M:%S')
    else:
        return time.strftime('%Y-%m-%d-%H:%M')


def tqdm_chunks(collection, chunk_size, enum=False):
    """Call funcy.chunks and return the resulting generator wrapped in a progress bar."""
    tqdm_nice = tqdm_notebook if in_notebook() else tqdm
    chunks = funcy.chunks(chunk_size, collection)
    if enum:
        chunks = enumerate(chunks)
    return tqdm_nice(chunks, total=int(np.ceil(len(collection) / chunk_size)))


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, Iterable): return list(o)
    return [o]



def get_membership_mask(candidates, collection_they_might_be_in) -> list:
    """Return a boolean list where entry i indicates whether candidates[i] is in the second arg."""
    return np.array([x in collection_they_might_be_in for x in candidates])


def drop_all_zero(tr_mat, te_mat):
    mask = np.squeeze(tr_mat.sum(0) > 0)
    return tr_mat[:, np.array(mask)[0]], te_mat[:, np.array(mask)[0]]


def denumerate(lst): return dict(enumerate(lst))
def lmap(fn, coll): return list(map(fn, coll))
def dhead(d, n=5): return funcy.project(d, funcy.take(n, d.keys()))

def sample_row(df): return df.sample(1).iloc[0]
def lzip(*args): return list(zip(*args))
def lmap(*args): return list(map(*args))
def keys(dct): return list(dct.keys())
def vals(dct): return list(dct.values())

# os utils
def zip_to_series(a, b):
    '''Check that two iterables are of equal length, and then make a {a:b} Series'''
    if len(a) != len(b):
        raise ValueError('zip_to_series got iterables of uneven '
                         'length: {}, {}'.format(len(a), len(b)))
    return pd.Series(dict(zip(a, b)))

def parse_date_from_path(path, pattern=r'\d+', date_format=None):
    '''Extract date string from path and return as pandas.Timestamp

    Args:
        path: (str) path that contains date str in arbitrary format
        pattern: (str) default r'[digits]+'; regex pattern to find date string in path string
        date_format: (str) default None; date format passed to pd.to_datetime. if None,
                     try multiple formats, and raise ValueError if it can't find a match
    '''
    basename = os.path.basename(path)
    date_str_list = re.findall(pattern, basename)
    # having "-" as separator allows pd.to_datetime to cover most of the edge cases
    sep = '-' if len(date_str_list) > 2 else ''
    date_str = sep.join(date_str_list)
    return pd.to_datetime(date_str, format=date_format)


# numerical utils (no pandas)
def tlog(x, t, base=10):
    '''Symmetric truncated log transform centered around 0.

    A monotonic transformation of the input variable x that is centered around 0.
    (i.e., positive and negative values are treated equally.)
    The transformation is fully invertible for t > 1 and is partially invertible for t = 1.

    |x| >= t -> np.log
    |x| < t -> linearly rescaled to be between (-np.log(t), np.log(t))

    Args:
        x: np.ndarray | pd.Series
        t: float
            A positive real number that's larger or equal to 1.
            Allowing for a number between (0, 1) would make the transformation non-monotonic.
        base: positive float or int
            The base for the log transformation.

    Returns:
        np.ndarray | pd.Series after log transform
    '''
    if t < 1.0:
        raise ValueError(u't has to be larger or equal to 1. Received: "{}"'.format(t))

    y = x.copy().astype('float64')  # cast to float in case input is defined on integers.

    indexes_to_log = np.abs(y) >= t

    indexes_to_rescale = ~indexes_to_log
    y[indexes_to_rescale] = x[indexes_to_rescale] * np.log(t) / float(t)

    values_to_log = x[indexes_to_log]
    result = np.sign(values_to_log) * np.log(np.abs(values_to_log))
    y[indexes_to_log] = result

    transformed_y = y / np.log(base)  # np.log promises to return a float
    return transformed_y


def exp_decay(X, rate):
    '''Compute exponential decay exp(-1 * rate * X)

    Args:
        X: ndarray-like object
            An array of distances
        rate: float
            The rate parameter of the exponential decay
    Returns:
        array of the decay values
    '''
    decay = np.exp(-1 * rate * X)
    return decay


def is_nan(n):
    '''np.isnan would break if something is not a float'''
    if isinstance(n, float):
        return np.isnan(n)
    else:
        return False


def percentile_clip(X, lower_pctl=1, upper_pctl=99):
    '''Clip series or array with upper and lower percentile'''
    min_val, max_val = np.nanpercentile(X, [lower_pctl, upper_pctl], axis=0)
    return np.clip(X, min_val, max_val)


def optimize_dtypes(df, col_subset=None, min_resolution=2e-6, include_index=True):
    '''Returns dataframe with data types optimized for memory consumption

    Args:
        df: (pd.DataFrame) dataframe to optimize
        col_subset: (list) columns to optimize over, defaults to all
        min_resolution: (float) minimum required resolution for floats (default gives float32)
        include_index: (bool) include index in optimization
    '''
    if include_index:
        index_name = df.index.name
        cols = df.columns
        df = df.reset_index()
        cols_with_index = df.columns
        index_cols = [c for c in cols_with_index if c not in cols]

    if col_subset is None:
        col_subset = df.columns

    # collect data type info
    int_types = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64]
    dtype_info = [(dinfo.dtype, dinfo.min, dinfo.max, dinfo.bits)
                  for dinfo in [np.iinfo(dtype) for dtype in int_types]]
    iinfo_df = pd.DataFrame(dtype_info, columns=['dtype', 'min_val', 'max_val', 'bits'])
    float_types = [np.float16, np.float32, np.float64]
    float_info = [(dinfo.dtype, dinfo.min, dinfo.max, dinfo.resolution)
                  for dinfo in [np.finfo(dtype) for dtype in float_types]]
    finfo_df = pd.DataFrame(float_info, columns=['dtype', 'min_val', 'max_val', 'resolution'])

    # collect int INSPECT_COLS to convert
    int_cols = df[col_subset].select_dtypes(include=[np.integer]).columns

    # collect int INSPECT_COLS to convert
    float_cols_all = df[col_subset].select_dtypes(include=[np.floating]).columns
    # dont upconvert float INSPECT_COLS that are already above min_resolution
    float_cols = []
    for c in float_cols_all:
        idx = (finfo_df['dtype'] == df[c].dtypes).argmax()
        if finfo_df.loc[idx, 'resolution'] < min_resolution:
            float_cols.append(c)

    for c in int_cols:
        min_val = df[c].min()
        max_val = df[c].max()
        poss_dtypes = iinfo_df[(iinfo_df.min_val <= min_val) & (iinfo_df.max_val >= max_val)]
        if poss_dtypes.shape[0] > 0:
            best_idx = poss_dtypes['bits'].idxmin()
            rec_dtype = poss_dtypes.loc[best_idx, 'dtype']
            df[c] = df[c].astype(rec_dtype)

    for c in float_cols:
        min_val = df[c].replace([np.inf, -np.inf], np.nan).dropna().min()
        max_val = df[c].replace([np.inf, -np.inf], np.nan).dropna().max()
        poss_dtypes = finfo_df[(finfo_df.min_val <= min_val) &
                               (finfo_df.max_val >= max_val) &
                               (finfo_df.resolution <= min_resolution)]
        if poss_dtypes.shape[0] > 0:
            best_idx = poss_dtypes['resolution'].idxmax()
            rec_dtype = poss_dtypes.loc[best_idx, 'dtype']
            df[c] = df[c].astype(rec_dtype)

    if include_index:
        df = df.set_index(index_cols)
        df.index.name = index_name

    return df


def select_subset(X=None, feature_names=None, patterns=None):
    '''Extracts a subset of X based on which of the columns match one of the patterns

    Args:
        X: (ndarray, DataFrame, or None); the data set to be filtered.
            None if just filtering a list of names
        feature_names: (list str); if X is not a pd.DataFrame, provide feature_names here
        patterns: (list, string, None) a set of regex patterns or just substrings;
                  if None, returns original df; raises error if none match
    Returns:
        ndarray or DataFrame depending on input type of X
    '''
    import re

    # Enforce input constraints
    if isinstance(X, pd.DataFrame) and feature_names is not None:
        raise ValueError('feature_names should be None when X is a pd.DataFrame')

    if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame) and feature_names is None:
        raise ValueError('feature_names must be provided when X is not a pd.DataFrame')

    if isinstance(X, pd.DataFrame) and isinstance(X.columns, pd.MultiIndex):
        raise NotImplementedError('does not work on multi-level columns at the moment')

    if patterns is not None:
        feature_names = feature_names if not isinstance(X, pd.DataFrame) else X.columns
        matching_features = np.array([any([re.match(pat, feat) or (pat in feat)
                                           for pat in to_iterable(patterns)])
                                      for feat in feature_names])

        if X is None:
            return [fn for (fn, mask) in zip(feature_names, matching_features) if mask]
        elif isinstance(X, np.ndarray):
            sub_X = X[:, matching_features]
        elif isinstance(X, pd.DataFrame):
            sub_X = X.loc[:, matching_features]

        if sub_X.size == 0:
            raise ValueError('no columns matched any of the patterns: {}'.format(patterns))
        else:
            return sub_X
    else:
        return X

def is_iterable(obj):
    '''Tests whether an object is iterable. Returns False for strings'''
    return hasattr(obj, '__iter__')

def to_list(item):
    '''Wrap object in list; to safely pass (possibly single) items to functions expecting lists'''
    return to_iterable(item, force_list=True)


def zscore(ser):
    '''get zscore of a pd.Series. returns pd.Series with same index.'''
    return (ser - ser.mean()) / ser.std()


def to_iterable(symbols, force_list=False):
    '''Transform strings or iterables of strings into something that can be safely iterated
    Args:
        symbols: input string or iterable to convert
        force_list: (bool) convert iterable to list
    '''
    if is_iterable(symbols) and force_list:
        return [symbol for symbol in symbols]
    elif is_iterable(symbols) or symbols is None:
        return symbols
    return [symbols]


def subset(df, patterns):
    '''Return select_subset(df, patterns=to_list(patterns))'''
    if isinstance(patterns, str):
        patterns = [patterns]
    return select_subset(df, patterns=patterns)


def drop_subset(df, pattern):
    '''drop subset of columns matching pattern'''
    columns_to_drop = df.pipe(subset, pattern).columns
    return df.drop(columns_to_drop, 1)


def blind_descending_sort(data):
    '''Sorts dataframe by first column, without knowing its name.
    Ideal after multi_unstack, and other notebook stuff.
    Args:
        data: (pd.Series or pd.DataFrame)
    '''
    if isinstance(data, pd.Series):
        return data.sort_values(ascending=False)
    return data.sort_values(data.columns[0], ascending=False)


DEFAULT_FLATTEN_SEP = '_'


def flatten(names, sep=DEFAULT_FLATTEN_SEP):
    """turn iterable of strings into _ separated string, or return itself if string is passed."""
    return sep.join(map(str, names)) if is_iterable(names) else names


def flatten_cols(arg_df, sep=DEFAULT_FLATTEN_SEP):
    """Turn multiindex into single index. Does not mutate."""
    df = arg_df.copy()
    df.columns = df.columns.map(lambda x: flatten(x, sep=sep))
    return df


def add_oof_yhat_column(clf, xydf, feature_names, y_col, n_folds=3, new_col='oof_yhat'):
    assert new_col not in xydf.columns
    xydf['_fold'] = np.random.randint(0, n_folds, xydf.shape[0])
    xydf[new_col] = np.nan
    for fold in tqdm_notebook(range(n_folds)):
        train_mask = xydf['_fold'] != fold
        clf.fit(xydf.loc[train_mask, feature_names], xydf.loc[train_mask, y_col])
        if hasattr(clf, 'predict_proba'):
            preds = clf.predict_proba(xydf.loc[~train_mask, feature_names])[:, 1]
        else:
            preds = clf.predict(xydf.loc[~train_mask, feature_names])
        xydf.loc[~train_mask, new_col] = preds
    return xydf  # .drop('fold', 1)


def test_low_var(df):
    desc = df.describe().T
    low_var = desc['std'].loc[lambda x: x == 0]
    if low_var.shape[0] >= 0:
        raise AssertionError('features {} have 0 variance'.format(low_var.index))


def permutation_importance(my_model, val_X, val_y, ret_df=False):
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
    if ret_df:
        return eli5.explain_weights_df(perm, feature_names=val_X.columns.tolist())
    else:
        return eli5.show_weights(perm, feature_names=val_X.columns.tolist())


def predict_proba_fn_binary(m, x):
    return m.predict_proba(x)[:, 1]


def predict_proba(m, x):
    return m.predict_proba(x)


def shape_assert(a, b):
    message = f'a.shape = {a.shape[0]} but b.shape = {b.shape[0]}'
    assert a.shape[0] == b.shape[0], message


def cross_val_predict_proba_df(m, X, y, n_splits=5, binary=True, sample_weight=None,
                               stratified=False):
    shape_assert(X, y)
    predict_proba_fn = predict_proba_fn_binary if binary else predict_proba
    if binary:
        preds = pd.Series({k: np.nan for k in X.index})
    else:
        n_classes = len(np.unique(y))
        preds = pd.DataFrame({k: [np.nan] * n_classes for k in X.index}).T
    if stratified:
        gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    else:
        gen = KFold(n_splits=n_splits, shuffle=False).split(X)

    for train, test in gen:
        if sample_weight is None:
            m.fit(X.iloc[train, :], y.iloc[train])
        else:
            m.fit(X.iloc[train, :], y.iloc[train], sample_weight=sample_weight.iloc[train])
        preds.iloc[test] = predict_proba_fn(m, X.iloc[test, :])
    return preds


def cross_val_predict_proba_fastai(train, n_splits=5, binary=True, sample_weight=None,
                               stratified=True):
    shape_assert(X, y)
    predict_proba_fn = predict_proba_fn_binary if binary else predict_proba
    if binary:
        preds = pd.Series({k: np.nan for k in X.index})
    else:
        n_classes = len(np.unique(y))
        preds = pd.DataFrame({k: [np.nan] * n_classes for k in X.index}).T
    if stratified:
        gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    else:
        gen = KFold(n_splits=n_splits, shuffle=False).split(X)

    for train, test in gen:
        if sample_weight is None:
            m.fit(X.iloc[train, :], y.iloc[train])
        else:
            m.fit(X.iloc[train, :], y.iloc[train], sample_weight=sample_weight.iloc[train])
        preds.iloc[test] = predict_proba_fn(m, X.iloc[test, :])
    return preds


def avg_predict_proba(clfs, X_test):
    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(X_test)
        else:
            preds += clf.predict_proba(X_test)
    preds = preds / len(clfs)
    return preds


def cross_val_predict_proba(m, X, y, predict_proba_fn=lambda m, x: m.predict_proba(x),
                            n_splits=5, combine=np.vstack, binary=True):
    cv = KFold(n_splits=n_splits, shuffle=False)
    preds = []
    for train, test in cv.split(X):
        m.fit(X[train, :], y[train])
        preds.append(predict_proba_fn(m, X[test, :]))
    preds = combine(preds)  # cro
    return preds[:, 1] if binary else preds