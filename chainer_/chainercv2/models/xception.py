"""
    Xception for ImageNet-1K, implemented in Chainer.
    Original paper: 'Xception: Deep Learning with Depthwise Separable Convolutions,' https://arxiv.org/abs/1610.02357.
"""

__all__ = ['Xception', 'xception']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, SimpleSequential


class DwsConv(Chain):
    """
    Depthwise separable convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 pad=0):
        super(DwsConv, self).__init__()
        with self.init_scope():
            self.dw_conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                groups=in_channels)
            self.pw_conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=1,
                nobias=True)

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DwsConvBlock(Chain):
    """
    Depthwise separable convolution block with batchnorm and ReLU pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad,
                 activate):
        super(DwsConvBlock, self).__init__()
        self.activate = activate

        with self.init_scope():
            if self.activate:
                self.activ = F.relu
            self.conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=kernel_size,
                stride=stride,
                pad=pad)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)

    def __call__(self, x):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def dws_conv3x3_block(in_channels,
                      out_channels,
                      activate):
    """
    3x3 version of the depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool
        Whether activate the convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        pad=1,
        activate=activate)


class XceptionUnit(Chain):
    """
    Xception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the downsample polling.
    reps : int
        Number of repetitions.
    start_with_relu : bool, default True
        Whether start with ReLU activation.
    grow_first : bool, default True
        Whether start from growing.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 reps,
                 start_with_relu=True,
                 grow_first=True):
        super(XceptionUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=None)

            self.body = SimpleSequential()
            with self.body.init_scope():
                for i in range(reps):
                    if (grow_first and (i == 0)) or ((not grow_first) and (i == reps - 1)):
                        in_channels_i = in_channels
                        out_channels_i = out_channels
                    else:
                        if grow_first:
                            in_channels_i = out_channels
                            out_channels_i = out_channels
                        else:
                            in_channels_i = in_channels
                            out_channels_i = in_channels
                    activate = start_with_relu if (i == 0) else True
                    setattr(self.body, "block{}".format(i + 1), dws_conv3x3_block(
                        in_channels=in_channels_i,
                        out_channels=out_channels_i,
                        activate=activate))
                if stride != 1:
                    setattr(self.body, "pool", partial(
                        F.max_pooling_2d,
                        ksize=3,
                        stride=stride,
                        pad=1,
                        cover_all=False))

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = F.identity(x)
        x = self.body(x)
        x = x + identity
        return x


class XceptionInitBlock(Chain):
    """
    Xception specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(XceptionInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=32,
                stride=2,
                pad=0)
            self.conv2 = conv3x3_block(
                in_channels=32,
                out_channels=64,
                stride=1,
                pad=0)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class XceptionFinalBlock(Chain):
    """
    Xception specific final block.
    """
    def __init__(self):
        super(XceptionFinalBlock, self).__init__()
        with self.init_scope():
            self.conv1 = dws_conv3x3_block(
                in_channels=1024,
                out_channels=1536,
                activate=False)
            self.conv2 = dws_conv3x3_block(
                in_channels=1536,
                out_channels=2048,
                activate=True)
            self.activ = F.relu
            self.pool = partial(
                F.average_pooling_2d,
                ksize=10,
                stride=1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class Xception(Chain):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 **kwargs):
        super(Xception, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", XceptionInitBlock(
                    in_channels=in_channels))
                in_channels = 64
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), XceptionUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=(2 if (j == 0) else 1),
                                reps=(2 if (j == 0) else 3),
                                start_with_relu=((i != 0) or (j != 0)),
                                grow_first=((i != len(channels) - 1) or (j != len(channels_per_stage) - 1))))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", XceptionFinalBlock())

            in_channels = 2048
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


def get_xception(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".chainer", "models"),
                 **kwargs):
    """
    Create Xception model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    channels = [[128], [256], [728] * 9, [1024]]

    net = Xception(
        channels=channels,
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


def xception(**kwargs):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xception(model_name="xception", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        xception,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xception or weight_count == 22855952)

        x = np.zeros((1, 3, 299, 299), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
