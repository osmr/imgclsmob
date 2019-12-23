"""
    ShaResNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.
"""

__all__ = ['ShaResNet', 'sharesnet18', 'sharesnet34', 'sharesnet50', 'sharesnet50b', 'sharesnet101', 'sharesnet101b',
           'sharesnet152', 'sharesnet152b']

import os
from inspect import isfunction
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import ReLU6, conv1x1_block, conv3x3_block, SimpleSequential
from .resnet import ResInitBlock


class ShaConvBlock(Chain):
    """
    Shared convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.activate
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    shared_conv : Chain, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 activation=(lambda: F.relu),
                 activate=True,
                 shared_conv=None):
        super(ShaConvBlock, self).__init__()
        self.activate = activate

        with self.init_scope():
            if shared_conv is None:
                self.conv = L.Convolution2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    ksize=ksize,
                    stride=stride,
                    pad=pad,
                    nobias=(not use_bias),
                    dilate=dilate,
                    groups=groups)
            else:
                self.conv = shared_conv
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            if self.activate:
                assert (activation is not None)
                if isfunction(activation):
                    self.activ = activation()
                elif isinstance(activation, str):
                    if activation == "relu":
                        self.activ = F.relu
                    elif activation == "relu6":
                        self.activ = ReLU6()
                    else:
                        raise NotImplementedError()
                else:
                    self.activ = activation

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def sha_conv3x3_block(in_channels,
                      out_channels,
                      stride=1,
                      pad=1,
                      dilate=1,
                      groups=1,
                      use_bias=False,
                      activation=(lambda: F.relu),
                      activate=True,
                      shared_conv=None):
    """
    3x3 version of the shared convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.activate
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    shared_conv : Chain, default None
        Shared convolution layer.
    """
    return ShaConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        activate=activate,
        shared_conv=shared_conv)


class ShaResBlock(Chain):
    """
    Simple ShaResNet block for residual path in ShaResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    shared_conv : Chain, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shared_conv=None):
        super(ShaResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            self.conv2 = sha_conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                activation=None,
                activate=False,
                shared_conv=shared_conv)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ShaResBottleneck(Chain):
    """
    ShaResNet bottleneck block for residual path in ShaResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    shared_conv : Chain, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_stride=False,
                 bottleneck_factor=4,
                 shared_conv=None):
        super(ShaResBottleneck, self).__init__()
        assert (conv1_stride or not ((stride > 1) and (shared_conv is not None)))
        mid_channels = out_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=(stride if conv1_stride else 1))
            self.conv2 = sha_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=(1 if conv1_stride else stride),
                shared_conv=shared_conv)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ShaResUnit(Chain):
    """
    ShaResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    shared_conv : Chain, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 conv1_stride,
                 shared_conv=None):
        super(ShaResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            if bottleneck:
                self.body = ShaResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv1_stride=conv1_stride,
                    shared_conv=shared_conv)
            else:
                self.body = ShaResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    shared_conv=shared_conv)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=None)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ShaResNet(Chain):
    """
    ShaResNet model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
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
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(ShaResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", ResInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    shared_conv = None
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            unit = ShaResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride,
                                shared_conv=shared_conv)
                            if (shared_conv is None) and not (bottleneck and not conv1_stride and stride > 1):
                                shared_conv = unit.body.conv2.conv
                            setattr(stage, "unit{}".format(j + 1), unit)
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_sharesnet(blocks,
                  conv1_stride=True,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create ShaResNet model with specific parameters.

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
    root : str, default '~/.chainer/models'
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
        raise ValueError("Unsupported ShaResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ShaResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def sharesnet18(**kwargs):
    """
    ShaResNet-18 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=18, model_name="sharesnet18", **kwargs)


def sharesnet34(**kwargs):
    """
    ShaResNet-34 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=34, model_name="sharesnet34", **kwargs)


def sharesnet50(**kwargs):
    """
    ShaResNet-50 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=50, model_name="sharesnet50", **kwargs)


def sharesnet50b(**kwargs):
    """
    ShaResNet-50b model with stride at the second convolution in bottleneck block from 'ShaResNet: reducing residual
    network parameter number by sharing weights,' https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=50, conv1_stride=False, model_name="sharesnet50b", **kwargs)


def sharesnet101(**kwargs):
    """
    ShaResNet-101 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=101, model_name="sharesnet101", **kwargs)


def sharesnet101b(**kwargs):
    """
    ShaResNet-101b model with stride at the second convolution in bottleneck block from 'ShaResNet: reducing residual
    network parameter number by sharing weights,' https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=101, conv1_stride=False, model_name="sharesnet101b", **kwargs)


def sharesnet152(**kwargs):
    """
    ShaResNet-152 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=152, model_name="sharesnet152", **kwargs)


def sharesnet152b(**kwargs):
    """
    ShaResNet-152b model with stride at the second convolution in bottleneck block from 'ShaResNet: reducing residual
    network parameter number by sharing weights,' https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=152, conv1_stride=False, model_name="sharesnet152b", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        sharesnet18,
        sharesnet34,
        sharesnet50,
        sharesnet50b,
        sharesnet101,
        sharesnet101b,
        sharesnet152,
        sharesnet152b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != sharesnet18 or weight_count == 8556072)
        # assert (model != sharesnet34 or weight_count == 13613864)
        # assert (model != sharesnet50 or weight_count == 17373224)
        # assert (model != sharesnet50b or weight_count == 20469800)
        # assert (model != sharesnet101 or weight_count == 26338344)
        # assert (model != sharesnet101b or weight_count == 29434920)
        # assert (model != sharesnet152 or weight_count == 33724456)
        # assert (model != sharesnet152b or weight_count == 36821032)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
