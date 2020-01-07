from durbango import tqdm_nice
from transformers import *
from fastai.text import *
import funcy
from functools import partial
from fastai.callbacks import *
from fastai.callbacks.mem import PeakMemMetric
import time
from fastai.basic_train import Learner
import transformers
from durbango import pickle_save

MODEL_CLASSES = {
    'bert-base-uncased': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet-base-cased': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm-mlm-enfr-1024': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta-base': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert-base-uncased': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
}

class TransformersBaseTokenizer:
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type='bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def add_special_cases(self, *args, **kwargs): pass

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str, fixup_func=funcy.identity) -> List[str]:
        """Limits the maximum sequence length and add the special tokens."""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[
                     :self.max_seq_len - 2]
        else:
            tokens = self._pretrained_tokenizer.tokenize(fixup_func(t))[:self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]


class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos=[])
        self.tokenizer = tokenizer

    def numericalize(self, t: Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        assert not isinstance(t, str)
        return self.tokenizer.convert_tokens_to_ids(t)
        # return self.tokenizer.encode(t)

    def textify(self, nums: Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."

        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(
            nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)


class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model

    def forward(self, input_ids):
        # Return only the logits from the transfomer
        logits = self.transformer(input_ids)[0]
        return logits


def choose_best_lr(learner: Learner, div_by=10., max_lr=1e-4, min_lr=1e-5):
    learner.lr_find()
    learner.recorder.plot()
    plt.show()

    lrs = learner.recorder.lrs
    losses = [x.item() for x in learner.recorder.losses]
    best_idx = np.argmin(losses)
    if best_idx == 0: print('lowest lr was the best')
    lr = max(min_lr, min(lrs[best_idx] / div_by, max_lr))
    print(f'chose lr: {lr:.2E}')
    return lr


"""
transformer_tokenizer = tokenizer_class.from_pretrained(model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer=transformer_tokenizer,
                                                       model_type=model_type)
fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])
"""

recorder_attrs_to_save = ['lrs', 'metrics', 'moms']


def run_experiment(sched, databunch, exp_name='dbert_baseline', fp_16=True, discrim_lr=False,
                   moms=(0.9, 0.999), clip=1., min_lr=0., orig_max_lr=1., one_cycle=True, num_labels=2,
                   reduce_on_plateau=False, wt_name='distilbert-base-uncased'):
    learner, num_groups = get_distilbert_learner(databunch, exp_name, wt_name, betas=moms, num_labels=num_labels)
    recorder_hist = defaultdict(list)
    if fp_16:
        learner = learner.to_fp16()

    lrs = []
    callbacks = [
                 CSVLogger(learner, filename='metrics', append=True),
                 PeakMemMetric(learner),
                 EarlyStoppingCallback(learner, monitor='accuracy', min_delta=-0.02, patience=5),
                 ]


    metadata = {}
    if reduce_on_plateau: callbacks.append(ReduceLROnPlateauCallback(learner, min_lr=min_lr))
    if clip is not None: callbacks.append(GradientClipping(learner, clip=clip))
    t0 = time.time()
    for freeze, cyc_len in sched:
        try:
            if freeze is None: learner.unfreeze()
            else: learner.freeze_to(freeze)

            if one_cycle:
                max_lr = choose_best_lr(learner, min_lr=min_lr, max_lr=orig_max_lr)
            else:
                max_lr = orig_max_lr
            if discrim_lr: max_lr = slice(max_lr * 0.95 ** num_groups, max_lr)
            lrs.append(max_lr)
            if one_cycle:
                learner.fit_one_cycle(cyc_len, max_lr=max_lr, moms=moms, callbacks=callbacks)
            else:
                learner.fit(cyc_len, lr=max_lr, callbacks=callbacks)
            losses = [x.item() for x in learner.recorder.losses]
            recorder_hist['losses'].append(losses)
            for attr in recorder_attrs_to_save:
                val = getattr(learner.recorder, attr, [np.nan])
                recorder_hist[attr].append(val)

            metadata = dict(
                lrs=lrs, sched=sched, desc='scheduled_v0', bs=databunch.batch_size, mins=(time.time() - t0) / 60,
                weight_name=wt_name, fp_16=fp_16, discrim_lr=discrim_lr,
                recorder_hist=dict(recorder_hist),
            )
            pickle_save(metadata, learner.path / 'metadata.pkl')
        except KeyboardInterrupt:
            break
    return learner, metadata


re1 = re.compile(r'  +')
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_distilbert_learner(databunch, exp_name, wt_name, num_labels=5, **optim_kwargs):
    adam_w_func = partial(transformers.AdamW, correct_bias=False, **optim_kwargs)
    transformer_model = DistilBertForSequenceClassification.from_pretrained(wt_name, num_labels=num_labels)
    custom_transformer_model = CustomTransformerModel(transformer_model=transformer_model)
    log_dir = Path(f'logs/{exp_name}')
    log_dir.mkdir(exist_ok=True)
    if (log_dir/'metrics.csv').exists():
        raise ValueError(f"{str(log_dir/'metrics.csv')} exists. Need new exp_name")
    learner = Learner(databunch,
                      custom_transformer_model,
                      path=log_dir,
                      opt_func=adam_w_func,
                      metrics=[accuracy])
    list_layers = [learner.model.transformer.distilbert.embeddings,
                   # can we consolidate this if we dont want finegrained control?
                   learner.model.transformer.distilbert.transformer.layer[0],
                   learner.model.transformer.distilbert.transformer.layer[1],
                   learner.model.transformer.distilbert.transformer.layer[2],
                   learner.model.transformer.distilbert.transformer.layer[3],
                   learner.model.transformer.distilbert.transformer.layer[4],
                   learner.model.transformer.distilbert.transformer.layer[5],
                   learner.model.transformer.pre_classifier]
    num_groups = len(list_layers)
    learner = learner.split(list_layers)
    return learner, num_groups
import funcy
def lflat(lsts): return list(funcy.flatten(lsts))

def make_lm_examples(text_arr, model_name = 'distilbert-base-uncased', block_size =512):
    """Note that we are losing the last truncated example here for the sake of simplicity (no padding)
    """
    transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _base_tokenizer = TransformersBaseTokenizer(
        pretrained_tokenizer=transformer_tokenizer, model_type=model_name)
    toks = [_base_tokenizer.tokenizer(p) for p in tqdm_nice(text_arr)]
    ids = [transformer_tokenizer.convert_tokens_to_ids(p) for p in tqdm_nice(toks)]
    tokenized_text = lflat(ids)

    examples = []
    for i in range(0, len(tokenized_text) - block_size + 1, block_size):
        examples.append(tokenized_text[i : i + block_size])
    return examples

