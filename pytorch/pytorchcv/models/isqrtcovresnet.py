"""
    iSQRT-COV-ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root
    Normalization,' https://arxiv.org/abs/1712.01034.
"""

__all__ = ['iSQRTCOVResNet', 'isqrtcovresnet18', 'isqrtcovresnet34', 'isqrtcovresnet50', 'isqrtcovresnet50b',
           'isqrtcovresnet101', 'isqrtcovresnet101b']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block
from .resnet import ResUnit, ResInitBlock


class CovPool(torch.autograd.Function):
    """
    Covariance pooling function.
    """
    @staticmethod
    def forward(ctx, x):
        batch, channels, height, width = x.size()
        n = height * width
        xn = x.reshape(batch, channels, n)
        identity_bar = ((1.0 / n) * torch.eye(n, dtype=xn.dtype, device=xn.device)).unsqueeze(dim=0).repeat(batch, 1, 1)
        ones_bar = torch.full((batch, n, n), fill_value=(-1.0 / n / n), dtype=xn.dtype, device=xn.device)
        i_bar = identity_bar + ones_bar
        sigma = xn.bmm(i_bar).bmm(xn.transpose(1, 2))
        ctx.save_for_backward(x, i_bar)
        return sigma

    @staticmethod
    def backward(ctx, grad_sigma):
        x, i_bar = ctx.saved_tensors
        batch, channels, height, width = x.size()
        n = height * width
        xn = x.reshape(batch, channels, n)
        grad_x = grad_sigma + grad_sigma.transpose(1, 2)
        grad_x = grad_x.bmm(xn).bmm(i_bar)
        grad_x = grad_x.reshape(batch, channels, height, width)
        return grad_x


class NewtonSchulzSqrt(torch.autograd.Function):
    """
    Newton-Schulz iterative matrix square root function.

    Parameters:
    ----------
    x : Tensor
        Input tensor (batch * cols * rows).
    n : int
        Number of iterations (n > 1).
    """
    @staticmethod
    def forward(ctx, x, n):
        assert (n > 1)
        batch, cols, rows = x.size()
        assert (cols == rows)
        m = cols
        identity = torch.eye(m, dtype=x.dtype, device=x.device).unsqueeze(dim=0).repeat(batch, 1, 1)
        x_trace = (x * identity).sum(dim=(1, 2), keepdim=True)
        a = x / x_trace
        i3 = 3.0 * identity
        yi = torch.zeros(batch, n - 1, m, m, dtype=x.dtype, device=x.device)
        zi = torch.zeros(batch, n - 1, m, m, dtype=x.dtype, device=x.device)
        b2 = 0.5 * (i3 - a)
        yi[:, 0, :, :] = a.bmm(b2)
        zi[:, 0, :, :] = b2
        for i in range(1, n - 1):
            b2 = 0.5 * (i3 - zi[:, i - 1, :, :].bmm(yi[:, i - 1, :, :]))
            yi[:, i, :, :] = yi[:, i - 1, :, :].bmm(b2)
            zi[:, i, :, :] = b2.bmm(zi[:, i - 1, :, :])
        b2 = 0.5 * (i3 - zi[:, n - 2, :, :].bmm(yi[:, n - 2, :, :]))
        yn = yi[:, n - 2, :, :].bmm(b2)
        x_trace_sqrt = torch.sqrt(x_trace)
        c = yn * x_trace_sqrt
        ctx.save_for_backward(x, x_trace, a, yi, zi, yn, x_trace_sqrt)
        ctx.n = n
        return c

    @staticmethod
    def backward(ctx, grad_c):
        x, x_trace, a, yi, zi, yn, x_trace_sqrt = ctx.saved_tensors
        n = ctx.n
        batch, m, _ = x.size()
        identity0 = torch.eye(m, dtype=x.dtype, device=x.device)
        identity = identity0.unsqueeze(dim=0).repeat(batch, 1, 1)
        i3 = 3.0 * identity

        grad_yn = grad_c * x_trace_sqrt
        b = i3 - yi[:, n - 2, :, :].bmm(zi[:, n - 2, :, :])
        grad_yi = 0.5 * (grad_yn.bmm(b) - zi[:, n - 2, :, :].bmm(yi[:, n - 2, :, :]).bmm(grad_yn))
        grad_zi = -0.5 * yi[:, n - 2, :, :].bmm(grad_yn).bmm(yi[:, n - 2, :, :])
        for i in range(n - 3, -1, -1):
            b = i3 - yi[:, i, :, :].bmm(zi[:, i, :, :])
            ziyi = zi[:, i, :, :].bmm(yi[:, i, :, :])
            grad_yi_m1 = 0.5 * (grad_yi.bmm(b) - zi[:, i, :, :].bmm(grad_zi).bmm(zi[:, i, :, :]) - ziyi.bmm(grad_yi))
            grad_zi_m1 = 0.5 * (b.bmm(grad_zi) - yi[:, i, :, :].bmm(grad_yi).bmm(yi[:, i, :, :]) - grad_zi.bmm(ziyi))
            grad_yi = grad_yi_m1
            grad_zi = grad_zi_m1

        grad_a = 0.5 * (grad_yi.bmm(i3 - a) - grad_zi - a.bmm(grad_yi))

        x_trace_sqr = x_trace * x_trace
        grad_atx_trace = (grad_a.transpose(1, 2).bmm(x) * identity).sum(dim=(1, 2), keepdim=True)
        grad_cty_trace = (grad_c.transpose(1, 2).bmm(yn) * identity).sum(dim=(1, 2), keepdim=True)
        grad_x_extra = (0.5 * grad_cty_trace / x_trace_sqrt - grad_atx_trace / x_trace_sqr).repeat(1, m, m) * identity

        grad_x = grad_a / x_trace + grad_x_extra
        return grad_x, None


class Triuvec(torch.autograd.Function):
    """
    Extract upper triangular part of matrix into vector form.
    """
    @staticmethod
    def forward(ctx, x):
        batch, cols, rows = x.size()
        assert (cols == rows)
        n = cols
        triuvec_inds = torch.ones(n, n).triu().view(n * n).nonzero()
        # assert (len(triuvec_inds) == n * (n + 1) // 2)
        x_vec = x.reshape(batch, -1)
        y = x_vec[:, triuvec_inds]
        ctx.save_for_backward(x, triuvec_inds)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, triuvec_inds = ctx.saved_tensors
        batch, n, _ = x.size()
        grad_x = torch.zeros_like(x).view(batch, -1)
        grad_x[:, triuvec_inds] = grad_y
        grad_x = grad_x.view(batch, n, n)
        return grad_x


class iSQRTCOVPool(nn.Module):
    """
    iSQRT-COV pooling layer.

    Parameters:
    ----------
    num_iter : int, default 5
        Number of iterations (num_iter > 1).
    """
    def __init__(self,
                 num_iter=5):
        super(iSQRTCOVPool, self).__init__()
        self.num_iter = num_iter
        self.cov_pool = CovPool.apply
        self.sqrt = NewtonSchulzSqrt.apply
        self.triuvec = Triuvec.apply

    def forward(self, x):
        x = self.cov_pool(x)
        x = self.sqrt(x, self.num_iter)
        x = self.triuvec(x)
        return x


class iSQRTCOVResNet(nn.Module):
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
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(iSQRTCOVResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i not in [0, len(channels) - 1]) else 1
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels))
        in_channels = final_block_channels
        self.features.add_module("final_pool", iSQRTCOVPool())

        in_features = in_channels * (in_channels + 1) // 2
        self.output = nn.Linear(
            in_features=in_features,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_isqrtcovresnet(blocks,
                       conv1_stride=True,
                       model_name=None,
                       pretrained=False,
                       root=os.path.join("~", ".torch", "models"),
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
    root : str, default '~/.torch/models'
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
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def isqrtcovresnet18(**kwargs):
    """
    iSQRT-COV-ResNet-18 model from 'Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix
    Square Root Normalization,' https://arxiv.org/abs/1712.01034.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_isqrtcovresnet(blocks=101, conv1_stride=False, model_name="isqrtcovresnet101b", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

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

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != isqrtcovresnet18 or weight_count == 44205096)
        assert (model != isqrtcovresnet34 or weight_count == 54313256)
        assert (model != isqrtcovresnet50 or weight_count == 56929832)
        assert (model != isqrtcovresnet50b or weight_count == 56929832)
        assert (model != isqrtcovresnet101 or weight_count == 75921960)
        assert (model != isqrtcovresnet101b or weight_count == 75921960)

        x = torch.randn(14, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 1000))


if __name__ == "__main__":
    _test()
