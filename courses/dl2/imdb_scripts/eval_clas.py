import fire
from fastai.text import *
from fastai.lm_rnn import *
from sklearn.metrics import confusion_matrix
import time
def eval_clas(model_dir_path, final_clas_file=None, load_encoder=True,
              lm_file=None, val_dir=None, cuda_id=0,
              lm_id='', clas_id=None, bs=64, backwards=False, save_hard=False,
              use_sampler=True,
              save_path='preds.npy',
              bpe=False):
    start= time.time()
    print(f'model_dir_path {model_dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
          f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else ''
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    model_dir_path = Path(model_dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    if final_clas_file is None:
        final_clas_file = f'{PRE}{clas_id}clas_1'
    if lm_file is None:
        lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = model_dir_path / 'models' / f'{lm_file}.h5'
    if load_encoder:
        assert lm_path.exists(), f'Error: {lm_path} does not exist.'
    if val_dir is None:
        val_dir = model_dir_path / 'tmp'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(val_dir / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(val_dir / f'val_{IDS}.npy')
    val_lbls = np.load(val_dir / 'lbl_val.npy').flatten()
    val_lbls = val_lbls.flatten()
    val_lbls -= val_lbls.min()
    c=int(val_lbls.max())+1

    val_ds = TextDataset(val_sent, val_lbls)
    if use_sampler:
        val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
        val_lbls_sampled = val_lbls[list(val_samp)]
    else:
        val_samp = None
        val_lbls_sampled = val_lbls

    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    # maybe if we change this first arg to val_dir we will have fewer issues.
    md = ModelData(model_dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        # Use the model's itos
        itos = pickle.load(open(model_dir_path / 'tmp' / 'itos.pkl', 'rb'))
        vs = len(itos)

    m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                layers=[em_sz*3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    if load_encoder:
        learn.load_encoder(lm_file)
    learn.load(final_clas_file)
    preds = learn.predict()
    if save_hard:
        pass
        #for x,y in val_dl:

    predictions = np.argmax(preds, axis=1)
    acc = (val_lbls_sampled == predictions).mean()


    print('Accuracy =', acc, 'Confusion Matrix =')
    print(confusion_matrix(val_lbls_sampled, predictions))

    ### Test- Compare Unsorted
    unsorted_preds = val_samp.unsort(preds)
    unsorted_class_preds = np.argmax(unsorted_preds, 1)
    unsorted_acc = (val_lbls == unsorted_class_preds).mean()
    print('Accuracy =', unsorted_acc, 'Confusion Matrix =')
    print(confusion_matrix(val_lbls, unsorted_class_preds))

    if save_path is not None:
        pred_save_path = model_dir_path / 'tmp' / save_path
        np.save(pred_save_path, unsorted_preds)
        np.save(model_dir_path/ 'tmp' / 'eval_labels.npy', val_lbls)

    print(f'Time: {time.time()- start}')

if __name__ == '__main__': fire.Fire(eval_clas)

