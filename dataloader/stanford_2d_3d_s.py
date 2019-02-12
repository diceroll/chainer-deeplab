import json
import random
from pathlib import Path

import chainer
import numpy as np
from chainer.datasets import split_dataset_random
from PIL import Image

from dataloader.augmentation import *


class Stanford2D3DS(chainer.dataset.DatasetMixin):

    def __init__(self, dir_path, task, area=None, train=True, n_data=None):
        '''
        Stanford 2D-3D-Semantics Dataset (2D-3D-S)
        http://buildingparser.stanford.edu/dataset.html

        Args:
            dir_path(type: str): Directory path that data exists
            task(type: str): 'depth' is depth estimation task.
                             'semantic' is semantic segmentation task.
                             'normal' is surface normal estimation task
            area(type: str): Area used for train_data(valid_data, test_data)
            notes:
                Area candidate: '1', '2', '3', '4', '5a', '5b', '6'
                If area is None, all of area will be used.

        Example:
            dataset = Stanford2D3DS(area='1 2 3 4 5a 5b')
        '''

        self.dir_path = Path(dir_path)
        if not self.dir_path.exists():
            raise ValueError

        if task not in ['depth', 'semantic', 'normal']:
            raise ValueError

        if task == 'semantic':
            self._read_semantic_labels()

        if area is None:
            area = '1 2 3 4 5a 5b 6'

        self.task = task
        self.train = train
        self.crop = None
        self.augmentations_before_crop = {}
        self.augmentations_after_crop = {}
        self.data = []
        self.label = []

        for a in area.split():
            p = self.dir_path / 'area_{}/data/rgb/'.format(a)
            self.data += list(p.glob('*.png'))

        if n_data is not None:
            random.shuffle(self.data)
            self.data = self.data[:n_data]

        for img_path in self.data:
            self.label.append(img_path.parent.parent / self.task / '{}{}.png'.format(img_path.name[:-7], self.task))

    def __len__(self):
        return len(self.data)

    def set_augmentations(self, crop=None, augmentations={}):
        self.crop = crop

        for process in augmentations:
            if process in ['cutout', 'random_erasing']:
                self.augmentations_after_crop[process] = augmentations[process]
            else:
                self.augmentations_before_crop[process] = augmentations[process]

    def get_example(self, i):
        img_path = self.data[i]
        img = np.array(Image.open(img_path), np.float32)

        label_path = self.label[i]
        if self.task == 'normal' and not label_path.exists():
            label_path = label_path.parent / '{}s.png'.format(label_path.name[:-4])
        label = np.array(Image.open(label_path))
        label = self._preprocess(label, self.task)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if self.train:
            for process in self.augmentations_before_crop:
                if np.random.rand() < self.augmentations_before_crop[process]:
                    if process == 'random_rotate':
                        img, label = eval(process)(img, label, self.task)
                    else:
                        img, label = eval(process)(img, label)

        if self.crop is not None:
            rnd1 = np.random.randint(img.shape[0] - self.crop)
            rnd2 = np.random.randint(img.shape[1] - self.crop)
            img = img[rnd1:rnd1 + self.crop, rnd2:rnd2 + self.crop, :]
            label = label[rnd1:rnd1 + self.crop, rnd2:rnd2 + self.crop, :]

        if self.train:
            for process in self.augmentations_after_crop:
                if np.random.rand() < self.augmentations_after_crop[process]:
                    img, label = eval(process)(img, label)

        img = img.transpose(2, 0, 1) / 255 - 0.5
        label = label.transpose(2, 0, 1)

        if self.task == 'semantic':
            label = label[0]

        return img, label

    def _preprocess(self, label, task):
        if task == 'depth':
            label[label == 65535] = -999
            label = np.float32(label) / 512
            label[label < 0] = -1

        elif task == 'semantic':
            label = (label * np.array((256**2, 256, 1)).reshape((1, 1, 3))).sum(axis=2)
            label[label > len(self.semantic_labels)] = 0
            label = np.frompyfunc(lambda x: self.semantic_labels[x], 1, 1)(label)
            label = np.int32(label)

        elif task == 'normal':
            label = (np.float32(label) - 128) / 128

        return label

    def _read_semantic_labels(self):
        self.label_dict = {
            '<UNK>': -1,
            'ceiling': 0,
            'floor': 1,
            'wall': 2,
            'column': 3,
            'beam': 4,
            'window': 5,
            'door': 6,
            'table': 7,
            'chair': 8,
            'bookcase': 9,
            'sofa': 10,
            'board': 11,
            'clutter': 12
        }
        with open(self.dir_path / 'semantic_labels.json') as f:
            semantic_labels = json.load(f)
        self.semantic_labels = [self.label_dict[key.split('_')[0]] for key in semantic_labels]


if __name__ == '__main__':
    dir_path = '/mnt/HDD1/Dataset/'

    dataset = Stanford2D3DS(dir_path, 'depth', area='1 2', train=True)
    dataset.set_augmentations(crop=513, augmentations={'mirror': 0.5, 'flip': 0.5})
    img, label = dataset.get_example(0)
    print(img.shape, img.dtype, label.shape, label.dtype)

    dataset = Stanford2D3DS(dir_path, 'semantic', area='3', train=False)
    dataset.set_augmentations(crop=513)
    img, label = dataset.get_example(0)
    print(img.shape, img.dtype, label.shape, label.dtype)

    dataset = Stanford2D3DS(dir_path, 'normal', area='5a 5b 6', train=False, n_data=100)
    dataset.set_augmentations(crop=513, augmentations={'gamma_correction': 0.8})
    img, label = dataset.get_example(0)
    print(img.shape, img.dtype, label.shape, label.dtype)
