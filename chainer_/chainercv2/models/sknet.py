"""
    SKNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.
"""

__all__ = ['SKNet', 'sknet50', 'sknet101', 'sknet152']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent, SimpleSequential
from .resnet import ResInitBlock


class SKConvBlock(Chain):
    """
    SKNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    groups : int, default 32
        Number of groups in branches.
    num_branches : int, default 2
        Number of branches (`M` parameter in the paper).
    reduction : int, default 16
        Reduction value for intermediate channels (`r` parameter in the paper).
    min_channels : int, default 32
        Minimal number of intermediate channels (`L` parameter in the paper).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=32,
                 num_branches=2,
                 reduction=16,
                 min_channels=32):
        super(SKConvBlock, self).__init__()
        self.num_branches = num_branches
        self.out_channels = out_channels
        mid_channels = max(in_channels // reduction, min_channels)

        with self.init_scope():
            self.branches = Concurrent(stack=True)
            with self.branches.init_scope():
                for i in range(num_branches):
                    dilate = 1 + i
                    setattr(self.branches, "branch{}".format(i + 2), conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        pad=dilate,
                        dilate=dilate,
                        groups=groups))
            self.fc1 = conv1x1_block(
                in_channels=out_channels,
                out_channels=mid_channels)
            self.fc2 = conv1x1(
                in_channels=mid_channels,
                out_channels=(out_channels * num_branches))
            self.softmax = partial(
                F.softmax,
                axis=1)

    def __call__(self, x):
        y = self.branches(x)

        u = F.sum(y, axis=1)
        s = F.average_pooling_2d(u, ksize=u.shape[2:])
        z = self.fc1(s)
        w = self.fc2(z)

        batch = w.shape[0]
        w = F.reshape(w, shape=(batch, self.num_branches, self.out_channels))
        w = self.softmax(w)
        w = F.expand_dims(F.expand_dims(w, axis=3), axis=4)

        y = y * w
        y = F.sum(y, axis=1)
        return y


class SKNetBottleneck(Chain):
    """
    SKNet bottleneck block for residual path in SKNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_factor=2):
        super(SKNetBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = SKConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SKNetUnit(Chain):
    """
    SKNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(SKNetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = SKNetBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
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


class SKNet(Chain):
    """
    SKNet model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(SKNet, self).__init__()
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
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), SKNetUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride))
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


def get_sknet(blocks,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
              **kwargs):
    """
    Create SKNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported SKNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SKNet(
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


def sknet50(**kwargs):
    """
    SKNet-50 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=50, model_name="sknet50", **kwargs)


def sknet101(**kwargs):
    """
    SKNet-101 model from 'Selective Kernel Networks,' https://arxiv.org/abs/1903.06586.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=101, model_name="sknet101", **kwargs)


def sknet152(**kwargs):
    """
    SKNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_sknet(blocks=152, model_name="sknet152", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        sknet50,
        sknet101,
        sknet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sknet50 or weight_count == 27479784)
        assert (model != sknet101 or weight_count == 48736040)
        assert (model != sknet152 or weight_count == 66295656)

        x = np.zeros((14, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
