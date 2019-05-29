import argparse
import torch
from fastai.imagito.utils import *
from fastai.vision import *
from fastai.imagito.classes import ClassFolders

def grade_examples(model_dir, ds_type):
    prefix = 'train' if ds_type is DatasetType.Train else 'val'
    preds = torch.load(model_dir + '/' + prefix + '_predictions.pt')
    targets = torch.load(model_dir + '/' + prefix + '_targets.pt')


    # loss = torch.load(model_dir + '/' + prefix + '_loss.pt')

    # contains 0 if that example is 'hard', 1 if the example if 'easy'
    grades = torch.argmax(preds, dim=1) == targets

    # TODO examples are out of order compared with how they are saved on disk


def main():
    parser =  argparse.ArgumentParser(description='')
    parser.add_argument('results_dir', help='directory containing saved results')
    args = parser.parse_args()

    model_dir = args.results_dir
    # grade_examples(model_dir, DatasetType.Train)
    grade_examples(model_dir, DatasetType.Valid)

if __name__ == '__main__':
    main()