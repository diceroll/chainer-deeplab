import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class MaskHuberLoss(function_node.FunctionNode):

    def __init__(self, delta=1.0, ignore_label=None):
        self.delta = delta
        self.ignore_label = ignore_label
        self.valid_size = None

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 't'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        x0, x1 = inputs
        diff = (inputs[0] - inputs[1]).ravel()
        if self.ignore_label is not None:
            self.t_valid = x1 != self.ignore_label
            diff *= self.t_valid.ravel()
            self.valid_size = self.t_valid.sum(dtype=diff.dtype)
        else:
            self.valid_size = diff.dtype.type(diff.size)
        delta = diff.dtype.type(self.delta)

        xp.abs(diff, out=diff)
        y = xp.square(diff)
        diff -= delta
        xp.maximum(diff, 0, dtype=diff.dtype, out=diff)
        xp.square(diff, out=diff)
        y -= diff
        y *= 0.5

        return abs(y).sum() / self.valid_size,

    def backward(self, indexes, grad_outputs):
        x0, x1 = self.get_retained_inputs()
        gy, = grad_outputs
        diff = x0 - x1
        if self.ignore_label is not None:
            diff *= self.t_valid
        # `functions.clip` only accepts float value.
        delta = float(self.delta)
        gx = chainer.functions.clip(diff, -delta, delta)
        gx = chainer.functions.broadcast_to(gy, gx.shape) * gx / self.valid_size

        return gx, -gx


def mask_huber_loss(x, t, delta=1.0, ignore_label=None):
    return MaskHuberLoss(delta=delta, ignore_label=ignore_label).apply((x, t))[0]
