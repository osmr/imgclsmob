"""
    X-DenseNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.
"""

__all__ = ['XDenseNet', 'xdensenet121_2', 'xdensenet161_2', 'xdensenet169_2', 'xdensenet201_2', 'pre_xconv3x3_block',
           'XDenseUnit']

import os
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .preresnet import PreResInitBlock, PreResActivation
from .densenet import TransitionBlock


@mx.init.register
class XMaskInit(mx.init.Initializer):
    """
    Returns an initializer performing "X-Net" initialization for masks.

    Parameters:
    ----------
    expand_ratio : int
        Ratio of expansion.
    """
    def __init__(self,
                 expand_ratio,
                 **kwargs):
        super(XMaskInit, self).__init__(**kwargs)
        assert (expand_ratio > 0)
        self.expand_ratio = expand_ratio

    def _init_weight(self, _, arr):
        shape = arr.shape
        expand_size = max(shape[1] // self.expand_ratio, 1)
        shape1_arange = mx.nd.arange(shape[1], ctx=arr.context)
        arr[:] = 0
        for i in range(shape[0]):
            jj = mx.nd.random.shuffle(shape1_arange)[:expand_size]
            arr[i, jj, :, :] = 1


class XConv2D(nn.Conv2D):
    """
    X-Convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    groups : int, default 1
        Number of groups.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups=1,
                 expand_ratio=2,
                 **kwargs):
        super(XConv2D, self).__init__(
            in_channels=in_channels,
            channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            **kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        grouped_in_channels = in_channels // groups
        self.mask = self.params.get(
            name="mask",
            grad_req="null",
            shape=(out_channels, grouped_in_channels, kernel_size[0], kernel_size[1]),
            init=XMaskInit(expand_ratio=expand_ratio),
            differentiable=False)

    def hybrid_forward(self, F, x, weight, bias=None, mask=None):
        masked_weight = weight * mask
        return super(XConv2D, self).hybrid_forward(F, x, weight=masked_weight, bias=bias)


class PreXConvBlock(HybridBlock):
    """
    X-Convolution block with Batch normalization and ReLU pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 bn_use_global_stats=False,
                 return_preact=False,
                 activate=True,
                 expand_ratio=2,
                 **kwargs):
        super(PreXConvBlock, self).__init__(**kwargs)
        self.return_preact = return_preact
        self.activate = activate

        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = nn.Activation("relu")
            self.conv = XConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                expand_ratio=expand_ratio)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_xconv1x1_block(in_channels,
                       out_channels,
                       strides=1,
                       use_bias=False,
                       bn_use_global_stats=False,
                       return_preact=False,
                       activate=True,
                       expand_ratio=2):
    """
    1x1 version of the pre-activated x-convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    return PreXConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate,
        expand_ratio=expand_ratio)


def pre_xconv3x3_block(in_channels,
                       out_channels,
                       strides=1,
                       padding=1,
                       dilation=1,
                       groups=1,
                       bn_use_global_stats=False,
                       return_preact=False,
                       activate=True,
                       expand_ratio=2):
    """
    3x3 version of the pre-activated x-convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    return PreXConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate,
        expand_ratio=expand_ratio)


class XDenseUnit(HybridBlock):
    """
    X-DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 dropout_rate,
                 expand_ratio,
                 **kwargs):
        super(XDenseUnit, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        with self.name_scope():
            self.conv1 = pre_xconv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                expand_ratio=expand_ratio)
            self.conv2 = pre_xconv3x3_block(
                in_channels=mid_channels,
                out_channels=inc_channels,
                bn_use_global_stats=bn_use_global_stats,
                expand_ratio=expand_ratio)
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.concat(identity, x, dim=1)
        return x


class XDenseNet(HybridBlock):
    """
    X-DenseNet model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int, default 2
        Ratio of expansion.
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
                 bn_use_global_stats=False,
                 dropout_rate=0.0,
                 expand_ratio=2,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(XDenseNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(PreResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    if i != 0:
                        stage.add(TransitionBlock(
                            in_channels=in_channels,
                            out_channels=(in_channels // 2),
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = in_channels // 2
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(XDenseUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bn_use_global_stats=bn_use_global_stats,
                            dropout_rate=dropout_rate,
                            expand_ratio=expand_ratio))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PreResActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_xdensenet(blocks,
                  expand_ratio=2,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create X-DenseNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    expand_ratio : int, default 2
        Ratio of expansion.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    if blocks == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif blocks == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif blocks == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif blocks == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported X-DenseNet version with number of layers {}".format(blocks))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = XDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        expand_ratio=expand_ratio,
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


def xdensenet121_2(**kwargs):
    """
    X-DenseNet-121-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=121, model_name="xdensenet121_2", **kwargs)


def xdensenet161_2(**kwargs):
    """
    X-DenseNet-161-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=161, model_name="xdensenet161_2", **kwargs)


def xdensenet169_2(**kwargs):
    """
    X-DenseNet-169-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=169, model_name="xdensenet169_2", **kwargs)


def xdensenet201_2(**kwargs):
    """
    X-DenseNet-201-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=201, model_name="xdensenet201_2", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        xdensenet121_2,
        xdensenet161_2,
        xdensenet169_2,
        xdensenet201_2,
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
        assert (model != xdensenet121_2 or weight_count == 7978856)
        assert (model != xdensenet161_2 or weight_count == 28681000)
        assert (model != xdensenet169_2 or weight_count == 14149480)
        assert (model != xdensenet201_2 or weight_count == 20013928)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
