from fastai.text import *
import html
import fire
import numpy as np
import re
import pandas as pd

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')


def make_dir_structure_under(path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def copy_subset_of_files(src_path: Path, dest_path, n=500,
                         dirs=('test', 'train')) -> None:
    """More making small IMDB"""
    if isinstance(src_path, str):
        src_path = Path(src_path)
    if isinstance(dest_path, str):
        dest_path = Path(dest_path)
    for sd in dirs:
        sdir = src_path / sd
        paths = list(sdir.glob('*/*.txt'))
        if sd == 'train':
            paths = [x for x in paths if 'unsup' not in str(x)]
        assert len(paths) > 0
        small_paths = np.random.choice(paths, size=n, replace=False)
        for sp in small_paths:
            full_dest_path = dest_path / sp.relative_to(src_path)
            make_dir_structure_under(full_dest_path)
            shutil.copy(sp, full_dest_path)
    return
    # should be under train
    sdir = src_path / 'unsup'
    if not sdir.exists():
        return
    paths = list(sdir.glob('*.txt'))
    small_paths = np.random.choice(paths, size=n, replace=False)
    for sp in small_paths:
        dest_path = dest_path / sp.relative_to(src_path)
        make_dir_structure_under(dest_path)
        shutil.copy(sp, dest_path)




CLASSES = ['neg', 'pos', 'unsup']
def read_texts(path, classes=CLASSES):
    texts,labels,fnames = [],[],[]
    for idx,label in enumerate(classes):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
            fnames.append(fname)
    return np.array(texts),np.array(labels), np.array(fnames)


np.random.seed(42)
def shuffle(lst1, lst2):
    trn_idx = np.random.permutation(len(lst1))
    return lst1[trn_idx], lst2[trn_idx]

def make_csv_from_dir(imdb_dir, dest_path):
    """
    Args:
        imdb_dir like imdb/train"""
    trn_texts, trn_labels, fnames = read_texts(imdb_dir)
    df_trn = pd.DataFrame({'text': trn_texts, 'labels': trn_labels},
                 columns=['labels','text']).sample(frac=1.)
    df_trn[df_trn['labels'] != 2].to_csv(dest_path, header=False, index=False)
    print(f'saved {df_trn.shape[0]} rows to {dest_path}')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls, lang='en'):
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls+1} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer(lang=lang).proc_all_mp(partition_by_cores(texts), lang=lang)
    return tok, list(labels)


def get_all(df, n_lbls, lang='en'):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls, lang=lang)
        tok += tok_
        labels += labels_
    return tok, labels


def create_toks(dir_path, chunksize=24000, n_lbls=1, lang='en', backwards=False):
    print(f'dir_path {dir_path} chunksize {chunksize} n_lbls {n_lbls} lang {lang}')
    try:
        spacy.load(lang)
    except OSError:
        # TODO handle tokenization of Chinese, Japanese, Korean
        print(f'spacy tokenization model is not installed for {lang}.')
        lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
        print(f'Command: python -m spacy download {lang}')
        sys.exit(1)
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    df_trn = pd.read_csv(dir_path / 'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(dir_path / 'val.csv', header=None, chunksize=chunksize)

    tmp_path = dir_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)
    tok_trn, trn_labels = get_all(df_trn, n_lbls, lang=lang)
    tok_val, val_labels = get_all(df_val, n_lbls, lang=lang)

    np.save(tmp_path / 'tok_trn.npy', tok_trn)
    np.save(tmp_path / 'tok_val.npy', tok_val)
    np.save(tmp_path / 'lbl_trn.npy', trn_labels)
    np.save(tmp_path / 'lbl_val.npy', val_labels)

    trn_joined = [' '.join(o) for o in tok_trn]
    open(tmp_path / 'joined.txt', 'w', encoding='utf-8').writelines(trn_joined)


if __name__ == '__main__': fire.Fire(create_toks)
