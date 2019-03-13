#from fastai.text import *
import numpy as np
from pathlib import Path
from collections import Counter
import collections
import fire
import pickle


def tok2id(dir_path, max_vocab=30000, min_freq=1, itos_path=None):
    """Makes trn_ids.npy, val_ids.npy and itos.pkl"""
    print(f'dir_path {dir_path} max_vocab {max_vocab} min_freq {min_freq}')
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    tmp_path = p / 'tmp'
    assert tmp_path.exists(), f'Error: {tmp_path} does not exist.'

    trn_tok = np.load(tmp_path / 'tok_trn.npy')
    val_tok = np.load(tmp_path / 'tok_val.npy')

    if itos_path is None:
        freq = Counter(p for o in trn_tok for p in o)
        print(freq.most_common(25))
        itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
        itos.insert(0, '_pad_')
        itos.insert(0, '_unk_')
    else:
        itos = pickle.load(Path(itos_path).open('rb'))
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print(len(itos))


    trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
    trn_lm_rev = np.array([[stoi[o] for o in reversed(p)] for p in trn_tok])
    val_lm = np.array([[stoi[o] for o in p] for p in val_tok])
    val_lm_rev = np.array([[stoi[o] for o in reversed(p)] for p in val_tok])

    np.save(tmp_path / 'trn_ids.npy', trn_lm)
    np.save(tmp_path / 'trn_ids_bwd.npy', trn_lm_rev)
    np.save(tmp_path / 'val_ids.npy', val_lm)
    np.save(tmp_path / 'val_ids_bwd.npy', val_lm_rev)
    pickle.dump(itos, open(tmp_path / 'itos.pkl', 'wb'))

if __name__ == '__main__': fire.Fire(tok2id)
