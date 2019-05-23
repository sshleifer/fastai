from __future__ import print_function
import os

from torchvision.datasets.folder import ImageFolder


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = self._verify_split(split)
        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        # wnid_to_classes = self._load_meta_file()[0]
        #
        #
        #
        #
        # idcs = [idx for _, idx in self.imgs]
        # self.wnids = self.classes
        # self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        # self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        # self.class_to_idx = {cls: idx
        #                      for clss, idx in zip(self.classes, idcs)
        #                      for cls in clss}



    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val'

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)



