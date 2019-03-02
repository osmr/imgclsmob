"""
    iSQRT-COV-ResNet, implemented in PyTorch.
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
    def backward(ctx, grad_output):
        x, i_bar = ctx.saved_tensors
        batch, channels, height, width = x.size()
        n = height * width
        xn = x.reshape(batch, channels, n)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(xn).bmm(i_bar)
        grad_input = grad_input.reshape(batch, channels, height, width)
        return grad_input


class NewtonSchulzSqrt(torch.autograd.Function):
    """
    Newton-Schulz iterative matrix square root function.
    """
    @staticmethod
    def forward(ctx, x, num_iter):
        assert (num_iter > 1)
        batch, cols, rows = x.size()
        assert (cols == rows)
        n = cols
        identity = torch.eye(n, dtype=x.dtype, device=x.device).unsqueeze(dim=0).repeat(batch, 1, 1)
        x_trace = (x * identity).sum(dim=(1, 2), keepdim=True)
        a = x / x_trace
        i3 = 3.0 * identity
        y = torch.zeros(batch, num_iter, n, n, dtype=x.dtype, device=x.device)
        z = torch.zeros(batch, num_iter, n, n, dtype=x.dtype, device=x.device)
        b = 0.5 * (i3 - a)
        y[:, 0, :, :] = a.bmm(b)
        z[:, 0, :, :] = b
        for i in range(1, num_iter - 1):
            b = 0.5 * (i3 - z[:, i - 1, :, :].bmm(y[:, i - 1, :, :]))
            y[:, i, :, :] = y[:, i - 1, :, :].bmm(b)
            z[:, i, :, :] = b.bmm(z[:, i - 1, :, :])
        yzy = 0.5 * y[:, num_iter - 2, :, :].bmm(i3 - z[:, num_iter - 2, :, :].bmm(y[:, num_iter - 2, :, :]))
        c = yzy * torch.sqrt(x_trace).view(batch, 1, 1).expand_as(x)
        ctx.save_for_backward(x, a, yzy, x_trace, y, z)
        ctx.num_iter = num_iter
        return c

    @staticmethod
    def backward(ctx, grad_output):
        x, A, ZY, normA, Y, Z = ctx.saved_tensors
        num_iter = ctx.num_iter
        batch_size = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA.squeeze()))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batch_size, 1, 1).type(dtype)
        if num_iter < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, num_iter - 2, :, :].bmm(Z[:, num_iter - 2, :, :])) -
                          Z[:, num_iter - 2, :, :].bmm(Y[:, num_iter - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, num_iter - 2, :, :].bmm(der_postCom).bmm(Y[:, num_iter - 2, :, :])
            for i in range(num_iter - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA)
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batch_size):
            grad_input[i, :, :] += (der_postComAux[i] - grad_aux[i] / (normA.squeeze()[i] * normA.squeeze()[i])) *\
                                   torch.ones(dim, device=x.device).diag().type(dtype)
        return grad_input, None


class Triuvec(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        x = input
        batch_size = x.data.shape[0]
        channels = x.data.shape[1]
        x = x.reshape(batch_size, channels * channels)
        identity = torch.ones(channels, channels).triu().reshape(channels * channels)
        index = identity.nonzero()
        y = torch.zeros(batch_size, channels * (channels + 1) // 2, device=x.device).type(x.dtype)
        y = x[:, index]
        ctx.save_for_backward(input, index)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, index = ctx.saved_tensors
        x = input
        batch_size = x.data.shape[0]
        channels = x.data.shape[1]
        grad_input = torch.zeros(batch_size, channels * channels, device=x.device, requires_grad=False).type(x.dtype)
        grad_input[:, index] = grad_output
        grad_input = grad_input.reshape(batch_size, channels, channels)
        return grad_input


class iSQRTCOVPool(nn.Module):
    """
    iSQRT-COV pooling layer.
    """
    def __init__(self):
        super(iSQRTCOVPool, self).__init__()
        self.cov_pool = CovPool.apply
        self.sqrt = NewtonSchulzSqrt.apply
        self.triu_vec = Triuvec.apply

    def forward(self, x):
        x = self.cov_pool(x)
        x = self.sqrt(x, 5)
        x = self.triu_vec(x)
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
                stride = 2 if (j == 0) and (i not in [0, len(channels_per_stage) - 1]) else 1
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
                       root=os.path.join('~', '.torch', 'models'),
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
    from torch.autograd import Variable

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

        x = Variable(torch.randn(14, 3, 224, 224))
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 1000))


if __name__ == "__main__":
    _test()
