import argparse
import random
import sys
from pathlib import Path

import chainer
import cupy
import numpy as np
from chainer import iterators, serializers

from dataloader.stanford_2d_3d_s import Stanford2D3DS
from models.deeplab import DeepLab
from models.modified_classifier import ModifiedClassifier
from modified_evaluator import ModifiedEvaluator


def main():
    parser = argparse.ArgumentParser(description='training mnist')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--load_model', '-lm', type=str, default=None,
                        help='Path of the model object to load')

    args = parser.parse_args()

    backbone = 'mobilenet'
    model = ModifiedClassifier(DeepLab(n_class=13, task='semantic', backbone=backbone))

    if args.load_model is not None:
        serializers.load_npz(args.load_model, model)
    else:
        print('You need to specify path of the model object')
        sys.exit()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dir_path = '/mnt/HDD1/Dataset'
    test_data = Stanford2D3DS(dir_path, 'semantic', area='5a', train=False, n_data=100)
    test_iter = iterators.MultiprocessIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    label_list = list(test_data.label_dict.keys())[1:]
    evaluator = ModifiedEvaluator(test_iter, model, label_names=label_list, device=args.gpu)
    observation = evaluator()

    for i in ['main/loss', 'main/acc', 'main/mean_class_acc', 'main/miou']:
        print(i, observation[i])


if __name__ == '__main__':
    main()
