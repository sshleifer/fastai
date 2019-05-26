import argparse
from pathlib import Path
from fastai.imagito.utils import *
from fastai.vision.models.xresnet2 import xresnet50_2
import torch


def load_model(model_dir):
    model_params = pickle_load(model_dir + '/params.pkl')
    m = xresnet50_2
    print(model_params)
    data = get_data(model_params['size'],
             model_params['woof'],
             model_params['bs'],
             model_params['sample'])

    l = Learner(data, m(c_out=10), path=model_dir)
    l.load('final_classif')
    return l

def predict_and_save(model, model_dir, ds_type):
    predictions, targets, loss = model.get_preds(ds_type=ds_type, with_loss=True)

    prefix = 'train' if ds_type is DatasetType.Train else 'val'

    torch.save(predictions, model_dir + '/' + prefix + '_predictions.pt')
    torch.save(targets, model_dir + '/' + prefix + '_targets.pt')
    torch.save(loss, model_dir + '/' + prefix + '_loss.pt')

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('saved_model', help='directory containing saved model')
    args = parser.parse_args()

    model_dir = args.saved_model
    model = load_model(model_dir)

    predict_and_save(model, model_dir, DatasetType.Train)
    predict_and_save(model, model_dir, DatasetType.Valid)

if __name__ == '__main__':
    main()
