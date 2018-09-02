from chainer.backends import cuda
from chainer.function import Function


class TopKAccuracy(Function):

    def __init__(self, k=1):
        self.k = k

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        argsorted_pred = xp.argsort(y)[:, -self.k:]
        return xp.asarray(xp.any(argsorted_pred.T == t, axis=0).mean(dtype=xp.float32)),


def top_k_accuracy(y, t, k=1):
    return TopKAccuracy(k=k)(y, t)

