import six
from chainer.backends import cuda
from chainer.function import Function
from chainer.utils import type_check


class TopKAccuracy(Function):

    def __init__(self, k=1):
        self.k = k

    def check_type_forward(self, in_types):
        type_check.argname(in_types, ('x', 't'))
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i'
        )

        t_ndim = type_check.eval(t_type.ndim)
        type_check.expect(
            x_type.ndim >= t_type.ndim,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
        )
        for i in six.moves.range(t_ndim + 1, type_check.eval(x_type.ndim)):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        argsorted_pred = xp.argsort(y)[:, -self.k:]
        return xp.asarray(xp.any(argsorted_pred.T == t, axis=0).mean(dtype=xp.float32)),


def top_k_accuracy(y, t, k=1):
    return TopKAccuracy(k=k)(y, t)
