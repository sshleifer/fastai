from fastai.callbacks import SaveModelCallback, CSVLogger
from fastai.callbacks.mem import PeakMemMetric
from fastai.text import *

#from .paths import PATH_LM

# from .preprocessing import USER_NAME, DR_NAME
# from .utils import CTX_COL, TARGET_COL, IS_VALID, to_arr
# from durbango.send_sms import send_sms
# from durbango import pickle_load
# from .paths import HOME

clas_config = {
    'emb_sz': 400,
    'n_hid': 1150,
    'n_layers': 3,
    'pad_token': 1,
    'qrnn': False,
    'bidir': False,
    'output_p': 0.4,
    'hidden_p': 0.3,
    'input_p': 0.4,
    'embed_p': 0.05,
    'weight_p': 0.5
}
MOMS = (0.8, 0.7)

def train_ulmfit(edf, MODEL_NAME, start_lr=1e-1, keep_going=True, bs=256, max_len=256, first_epoch_sched=(1,1),
                 save_preds=False, test_data=None,
                 **learner_kwargs):
    """This code needs to be synced with my fastai-for max_len branch"""
    vocab = pickle_load(HOME / 'lm_v6_vocab.pkl')
    tl = (TextList.from_df(edf, PATH_LM, cols=[CTX_COL], vocab=vocab)
      .split_from_df(IS_VALID).label_from_df(cols=[TARGET_COL]))

    if test_data is not None:
        td = TextList.from_df(test_data, PATH_LM, cols=[CTX_COL], vocab=vocab)
        tl = tl.add_test(td)

    data_clas = tl.databunch(bs=bs, max_len=max_len)
    learn = text_classifier_learner(data_clas, AWD_LSTM, config=clas_config, drop_mult=0.5,
                                    pretrained=False, clip=None, **learner_kwargs)
    learn.load_encoder('lm_fastai_v6_final_enc')
    callbacks = [SaveModelCallback(learn, name=MODEL_NAME),
                 CSVLogger(learn, filename=f'{MODEL_NAME}_history', append=True),
                 PeakMemMetric(learn)]
    lr = start_lr
    run_train_schedule(learn, callbacks, lr, keep_going, first_epoch_sched=first_epoch_sched)
    if not save_preds: return
    Path('ulm_preds').mkdir(exist_ok=True)
    ## Save Val Preds
    probas, _ = learn.get_preds(ordered=True)
    max_proba, pred = probas.max(1)
    val_df = edf.loc[edf[IS_VALID]]
    val_df['proba'] = to_arr(max_proba)
    val_df['pred'] = to_arr(pred)
    val_df.to_msgpack(f'ulm_preds/{MODEL_NAME}_preds.mp')
    if test_data is not None:
        probas, _ = learn.get_preds(DatasetType.Test, ordered=True)
        max_proba, pred = probas.max(1)
        test_data['proba'] = to_arr(max_proba)
        test_data['pred'] = to_arr(pred)
        test_data.to_msgpack(f'ulm_preds/{MODEL_NAME}_test_preds.mp')
    send_sms(f'Done with {MODEL_NAME}')


def run_train_schedule(learn, callbacks, lr, keep_going, first_epoch_sched=(1,1)):
    for n in first_epoch_sched:
        learn.fit_one_cycle(n, lr, moms=MOMS, callbacks=callbacks)
    learn.freeze_to(-2)
    lr /= 2
    learn.fit_one_cycle(1, slice(lr / (2.6 ** 4), lr), moms=MOMS, callbacks=callbacks)
    learn.freeze_to(-3)
    lr /= 2
    learn.fit_one_cycle(1, slice(lr / (2.6 ** 4), lr), moms=MOMS, callbacks=callbacks)
    learn.unfreeze()
    lr /= 5
    learn.fit_one_cycle(2, slice(lr / (2.6 ** 4), lr), moms=MOMS, callbacks=callbacks)
    if keep_going:
        learn.fit_one_cycle(20, slice(lr / (2.6 ** 4), lr), moms=MOMS, callbacks=callbacks)

#from durbango.nb_utils import
from durbango import lmap
def parse_time_to_seconds(timestr):
    m,s = lmap(int, timestr.split(':'))
    return m*60 + s


def path_to_df(path, do_max_len_split=True, return_raw=False):
    df = pd.read_csv(path)
    keep_mask = df['time'].apply(lambda x: x[0].isdigit())
    df = df.loc[keep_mask]
    df['seconds'] = df['time'].apply(parse_time_to_seconds)
    df = df.drop('time', 1)
    df = df.astype(float).reset_index(drop=True).drop('epoch',1)
    if return_raw: return df
    tot = df['seconds'].sum()
    maxes = df.max()
    mins = df.min()
    max_row = df['accuracy'].idxmax()
    maxes['best_epoch'] = max_row + 1
    time_to_max = df.loc[:max_row]['seconds'].sum()
    stem = path.stem
    if do_max_len_split:
        splat = stem.split('_')
        _, nturns, _,_, max_len, *shit = splat
        maxes['max_len'] = int(max_len)
        maxes['n_turns'] = int(nturns)

    maxes['stem'] = stem
    maxes['time_to_max'] = (time_to_max /60)
    maxes['total_time'] = (tot/60)
    maxes['valid_loss'], maxes['train_loss'] = mins['valid_loss'], mins['train_loss']
    return maxes
