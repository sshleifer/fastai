from fastai.text import *
import html
import fire

from imdb_scripts.create_toks import make_csv_from_dir, copy_subset_of_files, create_toks
from imdb_scripts.predict_with_classifier import run_tta_experiment
from imdb_scripts.train_clas import train_clas
from imdb_scripts.eval_clas import eval_clas
from imdb_scripts.tok2id import tok2id
from imdb_scripts.finetune_lm import train_lm
from sklearn.utils import shuffle

target_language = 'es'
WT103_PATH = Path('/home/paperspace/fastai-fork/courses/dl2/wt103/')
assert WT103_PATH.exists()
ORIG_SMALL_DATA_DIR = Path('/home/paperspace/text-augmentation/imdb_1k3k/')


def gen_eda_dataset(src_path, **eda_kwargs):
    from eda import eda
    paths = list((src_path / 'neg/').glob('*')) + list((src_path / 'pos/').glob('*'))
    for p in paths: assert 'EDA' not in str(p)
    for path in tqdm_notebook(paths):
        txt = path.open().read()
        new_sentences = eda(txt, **eda_kwargs)
        for i, new_sent in enumerate(new_sentences):
            new_name = f'{path.name[:-4]}_EDA_{i}.txt'
            new_path = path.parent / new_name
            with new_path.open('w') as f:
                f.write(new_sent)


def make_small_ds(src_path, dest_path, n_train, n_test=3000):

    if dest_path is None:
        dest_path = Path(f'/home/paperspace/imdb_{int(n_train/1000)}k_{int(n_test/1000)}k/')
        if dest_path.exists(): shutil.rmtree(dest_path)
        dest_path.mkdir(exist_ok=True)
        print(dest_path)

    copy_subset_of_files(src_path, dest_path, dirs=('train',), n=n_train)
    copy_subset_of_files(src_path, dest_path, dirs=('test',), n=n_test)
    prepare_tokens_and_labels(dest_path)
    return dest_path

#big_data_dir = Path('/home/paperspace/text-augmentation/imdb')

def run_experiment(target_language, n_to_copy=None, second_lang=False,
        orig_small_data_dir=ORIG_SMALL_DATA_DIR, classif_cl=20, lm_cl=4,

                   do_vat=False, **classif_kwargs):
    experiment_dir = Path(f'/home/paperspace/text-augmentation/imdb_small_aug_{target_language}')
    if experiment_dir.exists() and not second_lang:
        shutil.rmtree(experiment_dir)
    if not second_lang:
        shutil.copytree(orig_small_data_dir, experiment_dir)

    add_aug_files(target_language, experiment_dir, n_to_copy=n_to_copy)
    prepare_tokens_and_labels(experiment_dir)
    # Finetune LM
    train_lm(experiment_dir, WT103_PATH, early_stopping=True, cl=lm_cl)
    # Train Classifier
    learn = train_clas(experiment_dir, 0, cl=classif_cl, do_vat=do_vat, **classif_kwargs)

    return learn.sched.rec_metrics
    # eval_clas(experiment_dir, val_dir=Path('/home/paperspace/baseline_data/tmp/'))  # CudaError

import time


def run_eda_experiment(experiment_dir, wt_103_path=WT103_PATH, from_scratch=False, **classif_kwargs):
    """Experiment on smaller version of IMBD with different augmentation strategies"""
    results = {}
    start = time.time()
    if not from_scratch:
        train_lm(experiment_dir, wt_103_path, early_stopping=True, cl=10)
    # Train Classifier
    learn = train_clas(experiment_dir, 0, from_scratch=from_scratch, **classif_kwargs)
    estime = time.time() - start
    results.update({'metrics': learn.sched.rec_metrics, 'time': estime})
    return results

def run_n_experiment(src_path, target_language='es', n_train=2000, n_to_copy=None, eval_tta=False,
                     do_baseline=True, tta_langs=('et',)):
    """Experiment on smaller version of IMBD with different augmentation strategies"""
    reference_path = make_small_ds(src_path, None, n_train=n_train)
    experiment_dir = Path(f'/home/paperspace/text-augmentation/imdb_small_aug_{target_language}')
    results = {}
    start = time.time()
    es_metrics = run_experiment(
        target_language, orig_small_data_dir=reference_path, lm_cl=10,
        n_to_copy=n_to_copy,
    )
    estime = time.time() - start
    results.update({'btrans': es_metrics, 'btrans_time': estime})
    if eval_tta:
        for tta_lang in tta_langs:
            add_aug_files(tta_lang, experiment_dir, subdir='test')
        start = time.time()
        err_tab, tta_df = run_tta_experiment(experiment_dir / 'models' / 'fwd_clas_1.h5',
                                             experiment_dir / 'tmp' / 'itos.pkl',
                                             experiment_dir)

        results.update({'tta_err_tab': err_tab, 'tta_df': tta_df.drop('fnames', 1), 'tta_time': time.time() - start})
    if not do_baseline:
        return results

    start = time.time()
    baseline_metrics = run_experiment(target_language, orig_small_data_dir=reference_path,
                                      n_to_copy=0)
    base_time = time.time() - start
    results.update({'baseline': baseline_metrics, 'baseline_time': base_time, })
    return results


def add_aug_files(target_language, small_data_dir, n_to_copy=None, subdir='train'):
    aug_dir = Path(f'/home/paperspace/text-augmentation/imdb_{target_language}/')
    glb = (small_data_dir / subdir).glob('*/*.txt')
    selected_train = shuffle(list(glb))
    if n_to_copy is not None:
        selected_train = selected_train[:n_to_copy]
    assert aug_dir.exists()
    for p in selected_train:
        ext = p.relative_to(small_data_dir)
        p2 = aug_dir.joinpath(ext)
        if p2.exists():
            dest_path = str(p)[:-4] + f'_{target_language}.txt'
            shutil.copy(p2, dest_path)

def make_val_csv(small_data_dir, crosswalk_path=None):
    val_csv_path = small_data_dir/'val.csv'
    make_csv_from_dir(small_data_dir /'test', val_csv_path,
                      crosswalk_file=crosswalk_path)

def make_train_csv(small_data_dir):
    val_csv_path = small_data_dir/'train.csv'
    make_csv_from_dir(small_data_dir/'train', val_csv_path)

def prepare_tokens_and_labels(small_data_dir, max_vocab=60000):
    train_csv_path = small_data_dir/'train.csv'
    make_csv_from_dir(small_data_dir/'train', train_csv_path)
    trdf = pd.read_csv(small_data_dir/'train.csv', header=None)
    print(f'train_df.shape: {trdf.shape}')
    # Copy small val data
    make_csv_from_dir(small_data_dir/'test', small_data_dir/'val.csv')
    create_toks(small_data_dir)
    tok2id(small_data_dir, max_vocab=max_vocab)  # usually don't use this much



if __name__ == '__main__': fire.Fire(run_experiment)


