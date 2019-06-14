"""
    IBN-DenseNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.
"""

__all__ = ['IBNDenseNet', 'ibn_densenet121', 'ibn_densenet161', 'ibn_densenet169', 'ibn_densenet201']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import pre_conv3x3_block, IBN
from .preresnet import PreResInitBlock, PreResActivation
from .densenet import TransitionBlock


class IBNPreConvBlock(HybridBlock):
    """
    IBN-Net specific convolution block with BN/IBN normalization and ReLU pre-activation.

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
    use_ibn : bool, default False
        Whether use Instance-Batch Normalization.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_ibn=False,
                 bn_use_global_stats=False,
                 return_preact=False,
                 **kwargs):
        super(IBNPreConvBlock, self).__init__(**kwargs)
        self.use_ibn = use_ibn
        self.return_preact = return_preact

        with self.name_scope():
            if self.use_ibn:
                self.ibn = IBN(
                    channels=in_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    first_fraction=0.6,
                    inst_first=False)
            else:
                self.bn = nn.BatchNorm(
                    in_channels=in_channels,
                    use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        if self.use_ibn:
            x = self.ibn(x)
        else:
            x = self.bn(x)
        x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def ibn_pre_conv1x1_block(in_channels,
                          out_channels,
                          strides=1,
                          use_ibn=False,
                          bn_use_global_stats=False,
                          return_preact=False):
    """
    1x1 version of the IBN-Net specific pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_ibn : bool, default False
        Whether use Instance-Batch Normalization.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return IBNPreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        use_ibn=use_ibn,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact)


class IBNDenseUnit(HybridBlock):
    """
    IBN-DenseNet unit.

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
    conv1_ibn : bool
        Whether to use IBN normalization in the first convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 dropout_rate,
                 conv1_ibn,
                 **kwargs):
        super(IBNDenseUnit, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        with self.name_scope():
            self.conv1 = ibn_pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_ibn=conv1_ibn,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=inc_channels,
                bn_use_global_stats=bn_use_global_stats)
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


class IBNDenseNet(HybridBlock):
    """
    IBN-DenseNet model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 dropout_rate=0.0,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(IBNDenseNet, self).__init__(**kwargs)
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
                        conv1_ibn = (i < 3) and (j % 3 == 0)
                        stage.add(IBNDenseUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bn_use_global_stats=bn_use_global_stats,
                            dropout_rate=dropout_rate,
                            conv1_ibn=conv1_ibn))
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


def get_ibndensenet(num_layers,
                    model_name=None,
                    pretrained=False,
                    ctx=cpu(),
                    root=os.path.join("~", ".mxnet", "models"),
                    **kwargs):
    """
    Create IBN-DenseNet model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    if num_layers == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif num_layers == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif num_layers == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif num_layers == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported IBN-DenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = IBNDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def ibn_densenet121(**kwargs):
    """
    IBN-DenseNet-121 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=121, model_name="ibn_densenet121", **kwargs)


def ibn_densenet161(**kwargs):
    """
    IBN-DenseNet-161 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=161, model_name="ibn_densenet161", **kwargs)


def ibn_densenet169(**kwargs):
    """
    IBN-DenseNet-169 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=169, model_name="ibn_densenet169", **kwargs)


def ibn_densenet201(**kwargs):
    """
    IBN-DenseNet-201 model from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_ibndensenet(num_layers=201, model_name="ibn_densenet201", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        ibn_densenet121,
        ibn_densenet161,
        ibn_densenet169,
        ibn_densenet201,
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
        assert (model != ibn_densenet121 or weight_count == 7978856)
        assert (model != ibn_densenet161 or weight_count == 28681000)
        assert (model != ibn_densenet169 or weight_count == 14149480)
        assert (model != ibn_densenet201 or weight_count == 20013928)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
