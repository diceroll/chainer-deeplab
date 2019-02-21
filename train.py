import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import chainer
import chainer.functions as F
import cupy
import numpy as np
from chainer import iterators, optimizers, serializers
from chainer.training import StandardUpdater, Trainer, extensions

from dataloader.stanford_2d_3d_s import Stanford2D3DS
from models.deeplab import DeepLab
from models.modified_classifier import ModifiedClassifier
from modified_evaluator import ModifiedEvaluator


def main():
    parser = argparse.ArgumentParser(description='training mnist')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--report_trigger', '-rt', type=str, default='1e',
                        help='Interval for reporting(Ex.100i, default:1e)')
    parser.add_argument('--save_trigger', '-st', type=str, default='1e',
                        help='Interval for saving the model(Ex.100i, default:1e)')
    parser.add_argument('--load_model', '-lm', type=str, default=None,
                        help='Path of the model object to load')
    parser.add_argument('--load_optimizer', '-lo', type=str, default=None,
                        help='Path of the optimizer object to load')
    args = parser.parse_args()

    start_time = datetime.now()
    save_dir = Path('output/{}'.format(start_time.strftime('%Y%m%d_%H%M')))

    random.seed(args.seed)
    np.random.seed(args.seed)
    cupy.random.seed(args.seed)

    backbone = 'mobilenet'
    model = ModifiedClassifier(DeepLab(n_class=13, task='semantic', backbone=backbone), lossfun=F.softmax_cross_entropy)
    if args.load_model is not None:
        serializers.load_npz(args.load_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    if args.load_optimizer is not None:
        serializers.load_npz(args.load_optimizer, optimizer)

    dir_path = './dataset/2D-3D-S/'
    augmentations = {'mirror': 0.5, 'flip': 0.5}
    train_data = Stanford2D3DS(dir_path, 'semantic', area='1 2 3 4', train=True)
    train_data.set_augmentations(crop=513, augmentations=augmentations)
    valid_data = Stanford2D3DS(dir_path, 'semantic', area='6', train=False, n_data=100)
    valid_data.set_augmentations(crop=513)

    train_iter = iterators.MultiprocessIterator(train_data, args.batchsize, n_processes=1)
    valid_iter = iterators.MultiprocessIterator(valid_data, args.batchsize, repeat=False, shuffle=False, n_processes=1)

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=save_dir)

    label_list = list(valid_data.label_dict.keys())[1:]
    report_trigger = (int(args.report_trigger[:-1]), 'iteration' if args.report_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.LogReport(trigger=report_trigger))
    trainer.extend(ModifiedEvaluator(valid_iter, model, label_names=label_list,
                                     device=args.gpu), name='val', trigger=report_trigger)

    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/acc', 'val/main/loss',
                                           'val/main/acc', 'val/main/mean_class_acc', 'val/main/miou',
                                           'elapsed_time']), trigger=report_trigger)

    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key=report_trigger[1],
                                         marker='.', file_name='loss.png', trigger=report_trigger))
    trainer.extend(extensions.PlotReport(['main/acc', 'val/main/acc'], x_key=report_trigger[1],
                                         marker='.', file_name='accuracy.png', trigger=report_trigger))
    class_accuracy_report = ['val/main/mean_class_acc']
    class_accuracy_report.extend(['val/main/class_acc/{}'.format(label) for label in label_list])
    class_iou_report = ['val/main/miou']
    class_iou_report.extend(['val/main/iou/{}'.format(label) for label in label_list])
    trainer.extend(extensions.PlotReport(class_accuracy_report, x_key=report_trigger[1],
                                         marker='.', file_name='class_accuracy.png', trigger=report_trigger))
    trainer.extend(extensions.PlotReport(class_iou_report, x_key=report_trigger[1],
                                         marker='.', file_name='class_iou.png', trigger=report_trigger))

    save_trigger = (int(args.save_trigger[:-1]), 'iteration' if args.save_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.snapshot_object(model, filename='model_{0}-{{.updater.{0}}}.npz'
                                              .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.snapshot_object(optimizer, filename='optimizer_{0}-{{.updater.{0}}}.npz'
                                              .format(save_trigger[1])), trigger=save_trigger)

    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    (save_dir / 'training_details').mkdir()

    # Write parameters text
    with open(save_dir / 'training_details/train_params.txt', 'w') as f:
        f.write('model: {}(backbone: {})\n'.format(model.predictor.__class__.__name__, backbone))
        f.write('n_epoch: {}\n'.format(args.epoch))
        f.write('batch_size: {}\n'.format(args.batchsize))
        f.write('n_data_train: {}\n'.format(len(train_data)))
        f.write('n_data_val: {}\n'.format(len(valid_data)))
        f.write('seed: {}\n'.format(args.seed))
        if len(augmentations) > 0:
            f.write('[augmentation]\n')
            for process in augmentations:
                f.write('  {}: {}\n'.format(process, augmentations[process]))

    trainer.run()


if __name__ == '__main__':
    main()
