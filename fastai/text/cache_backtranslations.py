

import textblob
import funcy
import glob
from tqdm import tqdm_notebook
from fastai.text import *
from fastai.core import save_texts
import pickle
from pathlib import Path
from multiprocessing import Pool

def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_dir_structure_under(path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

def back_translate(x):
    es =  textblob.TextBlob(x).translate(to="es")
    en =  textblob.TextBlob(str(es)).translate(to="en")
    return str(en)





def save_backtranslations(txt_files):
    """Write back translations to disk."""
    for i, pth in enumerate(txt_files):
        save_path = pth.replace('imdb', 'imdb_es')
        if Path(save_path).exists():
            continue
        text = open_text(pth)
        back = back_translate(text)

        make_dir_structure_under(save_path)
        save_texts(save_path, [back])


def map_it():
    path = untar_data(URLs.IMDB)
    txt_files = glob.glob(f'{path}/*/*/*.txt')
    pool = Pool(8)
    chunks = funcy.chunks(1000, txt_files)
    pool.map(save_backtranslations, list(chunks))


if __name__ =='__main__':
    map_it()
