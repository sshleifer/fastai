from fastai.script import *
from fastai.vision import *
from fastai.vision.models.xresnet2 import xresnet50_2
from fastai.vision.models.xresnet import xresnet50
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.imagito.utils import *
from fastai.imagito.classes import ClassUtils
from fastai.imagito.sample_hardness import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def params_to_dict(gpu, woof, lr, size, alpha, mom, eps, epochs, bs, mixup, opt,
                   arch, dump, sample, classes=None):
    return {
        'gpu': gpu,
        'woof': woof,
        'lr': lr,
        'size': size,
        'alpha': alpha,
        'mom': mom,
        'eps': eps,
        'epochs': epochs,
        'bs': bs,
        'mixup': mixup,
        'opt': opt,
        'arch': arch,
        'dump': dump,
        'sample': sample,
        'classes': classes
    }


@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
        lr: Param("Learning rate", float)=3e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        alpha: Param("Alpha", float)=0.99,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-6,
        epochs: Param("Number of epochs", int)=20,
        bs: Param("Batch size", int)=256,
        mixup: Param("Mixup", float)=0.,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50, presnet34, presnet50, None)", str)=None,
        dump: Param("Print model; don't train", int)=0,
        fp16=True,
        sample: Param("Percentage of dataset to sample, ex: 0.1", float)=1.0,
        classes: Param("Comma-separated list of class indices to filter by, ex: 0,5,9", str)=None,
        label_smoothing=False,
        hardness_lower_bound=0., hardness_upper_bound=1.,
        save=False,
        ):
    "Distributed training of Imagenette."
    params_dict = locals()
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    if classes is not None and isinstance(classes, str): classes = [int(i) for i in classes.split(',')]

    filter_func = make_hardness_filter_func(hardness_lower_bound, hardness_upper_bound)
    if (hardness_lower_bound, hardness_upper_bound) != (0., 1.): assert sample == 1.
    data = get_data(size, woof, bs, sample, classes, filter_func=filter_func)
    params_dict['n_train'] = len(data.train_dl.dataset)
    bs_rat = bs/256
    if gpu is not None: bs_rat *= num_distrib()
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    lr *= bs_rat


    m = xresnet50_2 if arch is None else globals()[arch]
    # NOTE(SS): globals()[arch] raised KeyError

    # save params to file like experiments/2019-05-12_22:10/params.pkl
    now = get_date_str(seconds=True)
    Path('experiments').mkdir(exist_ok=True)
    model_dir = Path(f'experiments/{now}')
    model_dir.mkdir(exist_ok=False)
    pickle_save(params_dict, model_dir/'params.pkl')
    n_classes = len(classes) if classes is not None else 10
    loss_func = LabelSmoothingCrossEntropy() if label_smoothing else nn.CrossEntropyLoss()
    learn = Learner(data, m(c_out=n_classes), wd=1e-2, opt_func=opt_func,
                    path=model_dir,
                    metrics=[accuracy],
                    bn_wd=False, true_wd=True,
                    loss_func=loss_func)


    if dump: print(learn.model); exit()
    if mixup: learn = learn.mixup(alpha=mixup)
    if fp16: learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    # save results to a file like 2019-05-12_22:10/metrics.csv
    # (CSVLogger model_path/filename + .csv)
    csv_logger = CSVLogger(learn, filename='metrics')
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3,
                        callbacks=[csv_logger])
    if save:
        learn.save('final_classif')
    learn.destroy()

