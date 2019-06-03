"""
    NIN for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Network In Network,' https://arxiv.org/abs/1312.4400.
"""

__all__ = ['CIFARNIN', 'nin_cifar10', 'nin_cifar100', 'nin_svhn']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class NINConv(HybridBlock):
    """
    NIN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 **kwargs):
        super(NINConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=True,
                in_channels=in_channels)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class CIFARNIN(HybridBlock):
    """
    NIN model for CIFAR from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_kernel_sizes : list of int
        Convolution window sizes for the first units in each stage.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 first_kernel_sizes,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARNIN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        if (j == 0) and (i != 0):
                            if i == 1:
                                stage.add(nn.MaxPool2D(
                                    pool_size=3,
                                    strides=2,
                                    padding=1))
                            else:
                                stage.add(nn.AvgPool2D(
                                    pool_size=3,
                                    strides=2,
                                    padding=1))
                            stage.add(nn.Dropout(rate=0.5))
                        kernel_size = first_kernel_sizes[i] if j == 0 else 1
                        padding = (kernel_size - 1) // 2
                        stage.add(NINConv(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding))
                        in_channels = out_channels
                self.features.add(stage)

            self.output = nn.HybridSequential(prefix="")
            self.output.add(NINConv(
                in_channels=in_channels,
                out_channels=classes,
                kernel_size=1))
            self.output.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_nin_cifar(classes,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create NIN model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    channels = [[192, 160, 96], [192, 192, 192], [192, 192]]
    first_kernel_sizes = [5, 5, 3]

    net = CIFARNIN(
        channels=channels,
        first_kernel_sizes=first_kernel_sizes,
        classes=classes,
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


def nin_cifar10(classes=10, **kwargs):
    """
    NIN model for CIFAR-10 from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar(classes=classes, model_name="nin_cifar10", **kwargs)


def nin_cifar100(classes=100, **kwargs):
    """
    NIN model for CIFAR-100 from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar(classes=classes, model_name="nin_cifar100", **kwargs)


def nin_svhn(classes=10, **kwargs):
    """
    NIN model for SVHN from 'Network In Network,' https://arxiv.org/abs/1312.4400.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nin_cifar(classes=classes, model_name="nin_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (nin_cifar10, 10),
        (nin_cifar100, 100),
        (nin_svhn, 10),
    ]

    for model, classes in models:

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
        assert (model != nin_cifar10 or weight_count == 966986)
        assert (model != nin_cifar100 or weight_count == 984356)
        assert (model != nin_svhn or weight_count == 966986)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
