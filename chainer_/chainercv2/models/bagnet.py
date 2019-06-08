"""
    BagNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.
"""

__all__ = ['BagNet', 'bagnet9', 'bagnet17', 'bagnet33']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv1x1_block, conv3x3_block, ConvBlock, SimpleSequential


class BagNetBottleneck(Chain):
    """
    BagNet bottleneck block for residual path in BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size of the second convolution.
    stride : int or tuple/list of 2 int
        Stride of the second convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 bottleneck_factor=4):
        super(BagNetBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                ksize=ksize,
                stride=stride,
                pad=0)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BagNetUnit(Chain):
    """
    BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size of the second body convolution.
    stride : int or tuple/list of 2 int
        Stride of the second body convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride):
        super(BagNetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = BagNetBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
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
        if x.shape[-1] != identity.shape[-1]:
            diff = identity.shape[-1] - x.shape[-1]
            identity = identity[:, :, :-diff, :-diff]
        x = x + identity
        x = self.activ(x)
        return x


class BagNetInitBlock(Chain):
    """
    BagNet specific initial block.

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
        super(BagNetInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                pad=0)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BagNet(Chain):
    """
    BagNet model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_pool_size : int
        Size of the pooling windows for final pool.
    normal_kernel_sizes : list of int
        Count of the first units with 3x3 convolution window size for each stage.
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
                 final_pool_size,
                 normal_kernel_sizes,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(BagNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", BagNetInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != len(channels) - 1) else 1
                            ksize = 3 if j < normal_kernel_sizes[i] else 1
                            setattr(stage, "unit{}".format(j + 1), BagNetUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                ksize=ksize,
                                stride=stride))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=final_pool_size,
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


def get_bagnet(field,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create BagNet model with specific parameters.

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

    layers = [3, 4, 6, 3]

    if field == 9:
        normal_kernel_sizes = [1, 1, 0, 0]
        final_pool_size = 27
    elif field == 17:
        normal_kernel_sizes = [1, 1, 1, 0]
        final_pool_size = 26
    elif field == 33:
        normal_kernel_sizes = [1, 1, 1, 1]
        final_pool_size = 24
    else:
        raise ValueError("Unsupported BagNet with field: {}".format(field))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BagNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_pool_size=final_pool_size,
        normal_kernel_sizes=normal_kernel_sizes,
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


def bagnet9(**kwargs):
    """
    BagNet-9 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=9, model_name="bagnet9", **kwargs)


def bagnet17(**kwargs):
    """
    BagNet-17 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=17, model_name="bagnet17", **kwargs)


def bagnet33(**kwargs):
    """
    BagNet-33 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=33, model_name="bagnet33", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        bagnet9,
        bagnet17,
        bagnet33,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bagnet9 or weight_count == 15688744)
        assert (model != bagnet17 or weight_count == 16213032)
        assert (model != bagnet33 or weight_count == 18310184)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
