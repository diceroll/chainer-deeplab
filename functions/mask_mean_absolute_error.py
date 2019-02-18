import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class MaskMeanAbsoluteError(function_node.FunctionNode):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        if self.ignore_label is not None:
            self.t_valid = x1 != self.ignore_label
            diff *= self.t_valid.ravel()
        return numpy.array(abs(diff).sum() / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        if self.ignore_label is not None:
            self.t_valid = x1 != self.ignore_label
            diff *= self.t_valid.ravel()
        return abs(diff).sum() / diff.dtype.type(diff.size),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        coeff = gy * gy.array.dtype.type(1. / self.diff.size)
        coeff = chainer.functions.broadcast_to(coeff, self.diff.shape)
        gx0 = coeff * backend.get_array_module(gy.array).sign(self.diff)
        if self.ignore_label is not None:
            gx0 *= self.t_valid
        return gx0, -gx0


def mask_mean_absolute_error(x0, x1, ignore_label=None):
    return MaskMeanAbsoluteError(ignore_label=ignore_label).apply((x0, x1))[0]
