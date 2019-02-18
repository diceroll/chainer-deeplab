import numpy

from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class MaskMeanSquaredError(function_node.FunctionNode):

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
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        if self.ignore_label is not None:
            self.t_valid = inputs[1] != self.ignore_label
            diff *= self.t_valid.ravel()
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        if self.ignore_label is not None:
            self.t_valid = inputs[1] != self.ignore_label
            diff *= self.t_valid.ravel()
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, indexes, gy):
        x0, x1 = self.get_retained_inputs()
        ret = []
        diff = x0 - x1
        if self.ignore_label is not None:
            diff *= self.t_valid
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx0 = gy0 * diff * (2. / diff.size)
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret


def mask_mean_squared_error(x0, x1, ignore_label=None):
    return MaskMeanSquaredError(ignore_label=ignore_label).apply((x0, x1))[0]
