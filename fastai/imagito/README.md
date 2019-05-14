# imagenettissimo
Little baby datasets

## Setup

Fastai is already installed on the CS231n VM.

Update to the latest version (need to do so to get vision.models.xresnet2):
```
conda install -c fastai fastai
```

Sanity check: train on Imagenette 128px for 1 epoch:
```
python train_imagenette.py --epochs 1 --bs 128 --lr 3e-3 --mixup 0 --size 128
```

## Run


```
python fastai/imagito/train_imagito.py [args..]
```
any kwarg to the `main` func can be passed through the command line

or edit the param grid in 

## Dev on GCloud

ex. rsync with local filesystem (from `fastai/fastai/imagito`):
```
rsync -avhr ./ eprokop@34.83.17.89:/home/eprokop/fastai/imagito
```
