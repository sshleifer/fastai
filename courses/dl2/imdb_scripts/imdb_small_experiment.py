from fastai.text import *
import html
import fire

from imdb_scripts.create_toks import make_csv_from_dir, copy_subset_of_files, create_toks

from imdb_scripts.train_clas import train_clas
from imdb_scripts.eval_clas import eval_clas
from imdb_scripts.tok2id import tok2id
from imdb_scripts.finetune_lm import train_lm
from sklearn.utils import shuffle

target_language = 'es'
WT103_PATH = Path('/home/paperspace/fastai-fork/courses/dl2/wt103/')
assert WT103_PATH.exists()
orig_small_data_dir = Path('/home/paperspace/text-augmentation/imdb_1k3k/')


#big_data_dir = Path('/home/paperspace/text-augmentation/imdb')

def run_experiment(target_language, n_to_copy=None, second_lang=False):
    small_data_dir = Path(f'/home/paperspace/text-augmentation/imdb_small_aug_{target_language}')
    if small_data_dir.exists() and not second_lang:
        shutil.rmtree(small_data_dir)
        shutil.copytree(orig_small_data_dir, small_data_dir)

    add_aug_files(target_language, small_data_dir, n_to_copy=n_to_copy)
    prepare_tokens_and_labels(small_data_dir)
    # Finetune LM
    train_lm(small_data_dir, WT103_PATH, early_stopping=True, cl=10)
    # Train Classifier
    learn = train_clas(small_data_dir, 0, bs=64, cl=20)
    return learn.sched.rec_metrics
    # eval_clas(small_data_dir, val_dir=Path('/home/paperspace/baseline_data/tmp/'))  # CudaError


def add_aug_files(target_language, small_data_dir, n_to_copy=None):
    aug_dir = Path(f'/home/paperspace/text-augmentation/imdb_{target_language}/')
    selected_train = shuffle(list((small_data_dir / 'train/').glob('*/*.txt')))
    if n_to_copy is not None:
        selected_train = selected_train[:n_to_copy]
    assert aug_dir.exists()
    for p in selected_train:
        ext = p.relative_to(small_data_dir)
        p2 = aug_dir.joinpath(ext)
        if p2.exists():
            dest_path = str(p)[:-4] + f'_{target_language}.txt'
            shutil.copy(p2, dest_path)


def prepare_tokens_and_labels(small_data_dir):
    train_csv_path = small_data_dir/'train.csv'
    make_csv_from_dir(small_data_dir/'train', train_csv_path)
    trdf = pd.read_csv(small_data_dir/'train.csv', header=None)
    print(f'train_df.shape: {trdf.shape}')
    # Copy small val data
    make_csv_from_dir(small_data_dir/'test', small_data_dir/'val.csv')
    create_toks(small_data_dir)
    tok2id(small_data_dir, max_vocab=60000)  # usually don't use this much



if __name__ == '__main__': fire.Fire(run_experiment)


