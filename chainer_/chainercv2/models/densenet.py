"""
    DenseNet, implemented in Chainer.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class DenseConv(Chain):
    """
    DenseNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad):
        super(DenseConv, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(size=in_channels)
            self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True)

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def dense_conv1x1(in_channels,
                  out_channels):
    """
    1x1 version of the DenseNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DenseConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=1,
        pad=0)


def dense_conv3x3(in_channels,
                  out_channels):
    """
    3x3 version of the DenseNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DenseConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=1,
        pad=1)


class DenseUnit(Chain):
    """
    DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : bool
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate):
        super(DenseUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        with self.init_scope():
            self.conv1 = dense_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = dense_conv3x3(
                in_channels=mid_channels,
                out_channels=inc_channels)
            if self.use_dropout:
                self.dropout = partial(
                    F.dropout,
                    ratio=dropout_rate)

    def __call__(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.concat((identity, x), axis=1)
        return x


class TransitionBlock(Chain):
    """
    DenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(TransitionBlock, self).__init__()
        with self.init_scope():
            self.conv = dense_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)
            self.pool = partial(
                F.average_pooling_2d,
                ksize=2,
                stride=2,
                pad=0)

    def __call__(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseInitBlock(Chain):
    """
    DenseNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DenseInitBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=7,
                stride=2,
                pad=3,
                nobias=True)
            self.bn = L.BatchNormalization(size=out_channels)
            self.activ = F.relu
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class PostActivation(Chain):
    """
    DenseNet final block, which performs the same function of postactivation as in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PostActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(size=in_channels)
            self.activ = F.relu

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class DenseNet(Chain):
    """
    DenseNet model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 dropout_rate=0.0,
                 in_channels=3,
                 classes=1000):
        super(DenseNet, self).__init__()

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", DenseInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        if i != 0:
                            setattr(stage, "trans{}".format(i + 1), TransitionBlock(
                                in_channels=in_channels,
                                out_channels=(in_channels // 2)))
                            in_channels = in_channels // 2
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), DenseUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                dropout_rate=dropout_rate))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'post_activ', PostActivation(
                    in_channels=in_channels))
                setattr(self.features, 'final_pool', partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, 'flatten', partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, 'fc', L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_densenet(num_layers,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join('~', '.chainer', 'models'),
                 **kwargs):
    """
    Create DenseNet model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
        raise ValueError("Unsupported DenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [growth_rate] * yi,
                                   [xi[-1][-1] // 2])[1:]],
                      layers,
                      [[init_block_channels * 2]])[1:]

    net = DenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def densenet121(**kwargs):
    """
    DenseNet-121 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet(num_layers=121, model_name="densenet121", **kwargs)


def densenet161(**kwargs):
    """
    DenseNet-161 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet(num_layers=161, model_name="densenet161", **kwargs)


def densenet169(**kwargs):
    """
    DenseNet-169 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet(num_layers=169, model_name="densenet169", **kwargs)


def densenet201(**kwargs):
    """
    DenseNet-201 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_densenet(num_layers=201, model_name="densenet201", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = True

    models = [
        densenet121,
        densenet161,
        densenet169,
        densenet201,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != densenet121 or weight_count == 7978856)
        assert (model != densenet161 or weight_count == 28681000)
        assert (model != densenet169 or weight_count == 14149480)
        assert (model != densenet201 or weight_count == 20013928)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
