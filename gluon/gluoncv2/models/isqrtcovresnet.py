"""
    iSQRT-COV-ResNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root
    Normalization,' https://arxiv.org/abs/1712.01034.
"""

__all__ = ['iSQRTCOVResNet', 'isqrtcovresnet18', 'isqrtcovresnet34', 'isqrtcovresnet50', 'isqrtcovresnet50b',
           'isqrtcovresnet101', 'isqrtcovresnet101b']

import os
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block
from .resnet import ResUnit, ResInitBlock


class CovPool(mx.autograd.Function):
    """
    Covariance pooling function.
    """
    def forward(self, x):
        batch, channels, height, width = x.shape
        n = height * width
        xn = x.reshape(batch, channels, n)
        identity_bar = ((1.0 / n) * mx.nd.eye(n, ctx=xn.context, dtype=xn.dtype)).expand_dims(axis=0).repeat(
            repeats=batch, axis=0)
        ones_bar = mx.nd.full(shape=(batch, n, n), val=(-1.0 / n / n), ctx=xn.context, dtype=xn.dtype)
        i_bar = identity_bar + ones_bar
        sigma = mx.nd.batch_dot(mx.nd.batch_dot(xn, i_bar), xn.transpose(axes=(0, 2, 1)))
        self.save_for_backward(x, i_bar)
        return sigma

    def backward(self, grad_sigma):
        x, i_bar = self.saved_tensors
        batch, channels, height, width = x.shape
        n = height * width
        xn = x.reshape(batch, channels, n)
        grad_x = grad_sigma + grad_sigma.transpose(axes=(0, 2, 1))
        grad_x = mx.nd.batch_dot(mx.nd.batch_dot(grad_x, xn), i_bar)
        grad_x = grad_x.reshape(batch, channels, height, width)
        return grad_x


class NewtonSchulzSqrt(mx.autograd.Function):
    """
    Newton-Schulz iterative matrix square root function.

    Parameters:
    ----------
    n : int
        Number of iterations (n > 1).
    """
    def __init__(self, n):
        super(NewtonSchulzSqrt, self).__init__()
        assert (n > 1)
        self.n = n

    def forward(self, x):
        n = self.n
        batch, cols, rows = x.shape
        assert (cols == rows)
        m = cols
        identity = mx.nd.eye(m, ctx=x.context, dtype=x.dtype).expand_dims(axis=0).repeat(repeats=batch, axis=0)
        x_trace = (x * identity).sum(axis=(1, 2), keepdims=True)
        a = x / x_trace
        i3 = 3.0 * identity
        yi = mx.nd.zeros(shape=(batch, n - 1, m, m), ctx=x.context, dtype=x.dtype)
        zi = mx.nd.zeros(shape=(batch, n - 1, m, m), ctx=x.context, dtype=x.dtype)
        b2 = 0.5 * (i3 - a)
        yi[:, 0, :, :] = mx.nd.batch_dot(a, b2)
        zi[:, 0, :, :] = b2
        for i in range(1, n - 1):
            b2 = 0.5 * (i3 - mx.nd.batch_dot(zi[:, i - 1, :, :], yi[:, i - 1, :, :]))
            yi[:, i, :, :] = mx.nd.batch_dot(yi[:, i - 1, :, :], b2)
            zi[:, i, :, :] = mx.nd.batch_dot(b2, zi[:, i - 1, :, :])
        b2 = 0.5 * (i3 - mx.nd.batch_dot(zi[:, n - 2, :, :], yi[:, n - 2, :, :]))
        yn = mx.nd.batch_dot(yi[:, n - 2, :, :], b2)
        x_trace_sqrt = x_trace.sqrt()
        c = yn * x_trace_sqrt
        self.save_for_backward(x, x_trace, a, yi, zi, yn, x_trace_sqrt)
        return c

    def backward(self, grad_c):
        x, x_trace, a, yi, zi, yn, x_trace_sqrt = self.saved_tensors
        n = self.n
        batch, m, _ = x.shape
        identity0 = mx.nd.eye(m, ctx=x.context, dtype=x.dtype)
        identity = identity0.expand_dims(axis=0).repeat(repeats=batch, axis=0)
        i3 = 3.0 * identity

        grad_yn = grad_c * x_trace_sqrt
        b = i3 - mx.nd.batch_dot(yi[:, n - 2, :, :], zi[:, n - 2, :, :])
        grad_yi = 0.5 * (mx.nd.batch_dot(grad_yn, b) - mx.nd.batch_dot(mx.nd.batch_dot(
            zi[:, n - 2, :, :], yi[:, n - 2, :, :]), grad_yn))
        grad_zi = -0.5 * mx.nd.batch_dot(mx.nd.batch_dot(yi[:, n - 2, :, :], grad_yn), yi[:, n - 2, :, :])
        for i in range(n - 3, -1, -1):
            b = i3 - mx.nd.batch_dot(yi[:, i, :, :], zi[:, i, :, :])
            ziyi = mx.nd.batch_dot(zi[:, i, :, :], yi[:, i, :, :])
            grad_yi_m1 = 0.5 * (mx.nd.batch_dot(grad_yi, b) - mx.nd.batch_dot(mx.nd.batch_dot(
                zi[:, i, :, :], grad_zi), zi[:, i, :, :]) - mx.nd.batch_dot(ziyi, grad_yi))
            grad_zi_m1 = 0.5 * (mx.nd.batch_dot(b, grad_zi) - mx.nd.batch_dot(mx.nd.batch_dot(
                yi[:, i, :, :], grad_yi), yi[:, i, :, :]) - mx.nd.batch_dot(grad_zi, ziyi))
            grad_yi = grad_yi_m1
            grad_zi = grad_zi_m1

        grad_a = 0.5 * (mx.nd.batch_dot(grad_yi, i3 - a) - grad_zi - mx.nd.batch_dot(a, grad_yi))

        x_trace_sqr = x_trace * x_trace
        grad_atx_trace = (mx.nd.batch_dot(grad_a.transpose(axes=(0, 2, 1)), x) * identity).sum(
            axis=(1, 2), keepdims=True)
        grad_cty_trace = (mx.nd.batch_dot(grad_c.transpose(axes=(0, 2, 1)), yn) * identity).sum(
            axis=(1, 2), keepdims=True)
        grad_x_extra = (0.5 * grad_cty_trace / x_trace_sqrt - grad_atx_trace / x_trace_sqr).tile(
            reps=(1, m, m)) * identity

        grad_x = grad_a / x_trace + grad_x_extra
        return grad_x


class Triuvec(mx.autograd.Function):
    """
    Extract upper triangular part of matrix into vector form.
    """
    def forward(self, x):
        batch, cols, rows = x.shape
        assert (cols == rows)
        n = cols
        import numpy as np
        triuvec_inds = np.triu(np.ones(n)).reshape(-1).nonzero()[0]
        x_vec = x.reshape(batch, -1)
        y = x_vec[:, triuvec_inds]
        self.save_for_backward(x, triuvec_inds)
        return y

    def backward(self, grad_y):
        x, triuvec_inds = self.saved_tensors
        batch, n, _ = x.shape
        grad_x = mx.nd.zeros_like(x).reshape(batch, -1)
        grad_x[:, triuvec_inds] = grad_y
        grad_x = grad_x.reshape(batch, n, n)
        return grad_x


class iSQRTCOVPool(HybridBlock):
    """
    iSQRT-COV pooling layer.

    Parameters:
    ----------
    num_iter : int, default 5
        Number of iterations (num_iter > 1).
    """
    def __init__(self,
                 num_iter=5,
                 **kwargs):
        super(iSQRTCOVPool, self).__init__(**kwargs)
        with self.name_scope():
            self.cov_pool = CovPool()
            self.sqrt = NewtonSchulzSqrt(num_iter)
            self.triuvec = Triuvec()

    def hybrid_forward(self, F, x):
        x = self.cov_pool(x)
        x = self.sqrt(x)
        x = self.triuvec(x)
        return x


class iSQRTCOVResNet(HybridBlock):
    """
    iSQRT-COV-ResNet model from 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix
    Square Root Normalization,' https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 bottleneck,
                 conv1_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(iSQRTCOVResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i not in [0, len(channels) - 1]) else 1
                        stage.add(ResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels
            self.features.add(iSQRTCOVPool())

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            in_units = in_channels * (in_channels + 1) // 2
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_units))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_isqrtcovresnet(blocks,
                       conv1_stride=True,
                       model_name=None,
                       pretrained=False,
                       ctx=cpu(),
                       root=os.path.join("~", ".mxnet", "models"),
                       **kwargs):
    """
    Create iSQRT-COV-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported iSQRT-COV-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    final_block_channels = 256

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = iSQRTCOVResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def isqrtcovresnet18(**kwargs):
    """
    iSQRT-COV-ResNet-18 model from 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix
    Square Root Normalization,' https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=18, model_name="isqrtcovresnet18", **kwargs)


def isqrtcovresnet34(**kwargs):
    """
    iSQRT-COV-ResNet-34 model from 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix
    Square Root Normalization,' https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=34, model_name="isqrtcovresnet34", **kwargs)


def isqrtcovresnet50(**kwargs):
    """
    iSQRT-COV-ResNet-50 model from 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix
    Square Root Normalization,' https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=50, model_name="isqrtcovresnet50", **kwargs)


def isqrtcovresnet50b(**kwargs):
    """
    iSQRT-COV-ResNet-50 model with stride at the second convolution in bottleneck block from 'Towards Faster Training
    of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization,'
    https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=50, conv1_stride=False, model_name="isqrtcovresnet50b", **kwargs)


def isqrtcovresnet101(**kwargs):
    """
    iSQRT-COV-ResNet-101 model from 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix
    Square Root Normalization,' https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=101, model_name="isqrtcovresnet101", **kwargs)


def isqrtcovresnet101b(**kwargs):
    """
    iSQRT-COV-ResNet-101 model with stride at the second convolution in bottleneck block from 'Towards Faster Training
    of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization,'
    https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=101, conv1_stride=False, model_name="isqrtcovresnet101b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        isqrtcovresnet18,
        isqrtcovresnet34,
        isqrtcovresnet50,
        isqrtcovresnet50b,
        isqrtcovresnet101,
        isqrtcovresnet101b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != isqrtcovresnet18 or weight_count == 44205096)
        assert (model != isqrtcovresnet34 or weight_count == 54313256)
        assert (model != isqrtcovresnet50 or weight_count == 56929832)
        assert (model != isqrtcovresnet50b or weight_count == 56929832)
        assert (model != isqrtcovresnet101 or weight_count == 75921960)
        assert (model != isqrtcovresnet101b or weight_count == 75921960)

        x = mx.nd.random.randn(14, 3, 224, 224, ctx=ctx)
        # y = net(x)
        x.attach_grad()
        with mx.autograd.record():
            y = net(x)
            y.backward()
            # print(x.grad)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
