from pathlib import Path

import torch

from torchvision.utils import save_image



def save_distilled_images_for_fastai(results_pth, save_dir, map_location=None):
    save_dir = Path(save_dir)
    assert 'train' not in str(save_dir), f'this will add train/ for you, got {save_dir}'
    save_dir.mkdir(exist_ok=True)
    triples_lst = torch.load(results_pth, map_location)
    i = 0
    for a,b, _ in triples_lst:
        for img,label in zip(a,b):
            save_path = save_dir / f'train/{label}/{i}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(img, save_path)
            i += 1

