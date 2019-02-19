import copy

import numpy as np
from chainer import configuration, cuda, function, link
from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.training import extension, extensions
from chainercv.evaluations import eval_semantic_segmentation


class ModifiedEvaluator(extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, iterator, target, label_names=None, device=None):
        super(ModifiedEvaluator, self).__init__(iterator, target, device=device)
        self.label_names = label_names

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()
        pred_labels = []
        gt_labels = []

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)
                if eval_func.predictions is not None:
                    pred_labels.extend(cuda.to_cpu(eval_func.predictions))
                    gt_labels.extend(cuda.to_cpu(eval_func.gt))

            summary.add(observation)

        observation = summary.compute_mean()

        if self.label_names is not None and len(pred_labels) > 0:
            pred_labels = np.array(pred_labels)
            gt_labels = np.array(gt_labels)
            result = eval_semantic_segmentation(pred_labels, gt_labels)
            report = {'miou': result['miou'],
                      'pixel_acc': result['pixel_accuracy'],
                      'mean_class_acc': result['mean_class_accuracy']}
            for l, label_name in enumerate(self.label_names):
                try:
                    report['iou/{:s}'.format(label_name)] = result['iou'][l]
                    report['class_acc/{:s}'.format(label_name)] = result['class_accuracy'][l]
                except IndexError:
                    report['iou/{:s}'.format(label_name)] = np.nan
                    report['class_acc/{:s}'.format(label_name)] = np.nan

            with reporter_module.report_scope(observation):
                reporter_module.report(report, eval_func)

        return observation
