from fastai.script import *
from fastai.vision import *
from fastai.vision.models import xresnet50
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.imagito.utils import *

from fastai.imagito.classes import ClassFolders

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def filter_classes(image_list, classes=None):
    if (classes is None):
        return image_list

    class_names = ClassFolders.from_indices(classes)
    def class_filter(path):
        for class_name in class_names:
            if class_name in str(path):
                return True
        return False

    return image_list.filter_by_func(class_filter)

def get_data(size, woof, bs, sample, classes=None, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    image_list = ImageList.from_folder(path)
    image_list = filter_classes(image_list, classes)

    return (image_list
            .use_partial_data(sample)
            .split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

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

def results_path(filename):
    return './results/' + filename


#from fastprogress


@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        alpha: Param("Alpha", float)=0.99,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-6,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=256,
        mixup: Param("Mixup", float)=0.,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50, presnet34, presnet50)", str)='xresnet50',
        dump: Param("Print model; don't train", int)=0,
        fp16=False,
        sample: Param("Percentage of dataset to sample, ex: 0.1", float)=1.0,
        classes: Param("Comma-separated list of class indices to filter by, ex: 0,5,9", str)=None
        ):
    "Distributed training of Imagenette."
    params_dict = locals()
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)

    if classes is not None: classes = [int(i) for i in classes.split(',')]

    data = get_data(size, woof, bs, sample, classes)
    bs_rat = bs/256
    if gpu is not None: bs_rat *= num_distrib()
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    lr *= bs_rat

    m = xresnet50# globals()[arch]
    # save params to file like experiments/2019-05-12_22:10/params.pkl
    now = get_date_str(seconds=False)
    Path('experiments').mkdir(exist_ok=True)
    model_dir = Path(f'experiments/{now}')
    model_dir.mkdir(exist_ok=False)
    pickle_save(params_dict, model_dir/'params.pkl')

    learn = Learner(data, m(c_out=10), wd=1e-2, opt_func=opt_func,
                    path=model_dir,
                    metrics=[accuracy, top_k_accuracy],
                    bn_wd=False, true_wd=True,
                    loss_func=LabelSmoothingCrossEntropy())


    if dump: print(learn.model); exit()
    if mixup: learn = learn.mixup(alpha=mixup)
    if fp16: learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    # save results to a file like 2019-05-12_22:10/metrics.csv
    csv_logger = CSVLogger(learn, filename=model_dir/'metrics')

    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3,
                        callbacks=[csv_logger])

