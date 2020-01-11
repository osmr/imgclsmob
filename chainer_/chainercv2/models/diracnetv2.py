"""
    DiracNetV2 for ImageNet-1K, implemented in Chainer.
    Original paper: 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.
"""

__all__ = ['DiracNetV2', 'diracnet18v2', 'diracnet34v2']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class DiracConv(Chain):
    """
    DiracNetV2 specific convolution block with pre-activation.

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
        super(DiracConv, self).__init__()
        with self.init_scope():
            self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=False)

    def __call__(self, x):
        x = self.activ(x)
        x = self.conv(x)
        return x


def dirac_conv3x3(in_channels,
                  out_channels):
    """
    3x3 version of the DiracNetV2 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DiracConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=1,
        pad=1)


class DiracInitBlock(Chain):
    """
    DiracNetV2 specific initial block.

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
        super(DiracInitBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=7,
                stride=2,
                pad=3,
                nobias=False)
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DiracNetV2(Chain):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

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
        super(DiracNetV2, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", DiracInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), dirac_conv3x3(
                                in_channels=in_channels,
                                out_channels=out_channels))
                            in_channels = out_channels
                        if i != len(channels) - 1:
                            setattr(stage, "pool{}".format(i + 1), partial(
                                F.max_pooling_2d,
                                ksize=2,
                                stride=2,
                                pad=0))
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_activ", F.relu)
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


def get_diracnetv2(blocks,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".chainer", "models"),
                   **kwargs):
    """
    Create DiracNetV2 model with specific parameters.

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
    if blocks == 18:
        layers = [4, 4, 4, 4]
    elif blocks == 34:
        layers = [6, 8, 12, 6]
    else:
        raise ValueError("Unsupported DiracNetV2 with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    init_block_channels = 64

    net = DiracNetV2(
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


def diracnet18v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=18, model_name="diracnet18v2", **kwargs)


def diracnet34v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=34, model_name="diracnet34v2", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        diracnet18v2,
        diracnet34v2,
    ]

    for model in models:
        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diracnet18v2 or weight_count == 11511784)
        assert (model != diracnet34v2 or weight_count == 21616232)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
