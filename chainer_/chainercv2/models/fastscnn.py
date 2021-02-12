"""
    Fast-SCNN for image segmentation, implemented in Chainer.
    Original paper: 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502.
"""

__all__ = ['FastSCNN', 'fastscnn_cityscapes']

import os
import chainer.functions as F
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwsconv3x3_block, Concurrent,\
    InterpolationBlock, SimpleSequential


class Stem(Chain):
    """
    Fast-SCNN specific stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : tuple/list of 3 int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 **kwargs):
        super(Stem, self).__init__(**kwargs)
        assert (len(channels) == 3)

        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=channels[0],
                stride=2,
                pad=0)
            self.conv2 = dwsconv3x3_block(
                in_channels=channels[0],
                out_channels=channels[1],
                stride=2)
            self.conv3 = dwsconv3x3_block(
                in_channels=channels[1],
                out_channels=channels[2],
                stride=2)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LinearBottleneck(Chain):
    """
    Fast-SCNN specific Linear Bottleneck layer from MobileNetV2.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the second convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class FeatureExtractor(Chain):
    """
    Fast-SCNN specific feature extractor/encoder.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : list of list of int
        Number of output channels for each unit.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != len(channels) - 1) else 1
                            setattr(stage, "unit{}".format(j + 1), LinearBottleneck(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)

    def __call__(self, x):
        x = self.features(x)
        return x


class PoolingBranch(Chain):
    """
    Fast-SCNN specific pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    down_size : int
        Spatial size of downscaled image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 down_size,
                 **kwargs):
        super(PoolingBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.down_size = down_size

        with self.init_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=in_size)

    def __call__(self, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = F.average_pooling_2d(x, ksize=(in_size[0] // self.down_size, in_size[1] // self.down_size))
        x = self.conv(x)
        x = self.up(x, in_size)
        return x


class FastPyramidPooling(Chain):
    """
    Fast-SCNN specific fast pyramid pooling block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 **kwargs):
        super(FastPyramidPooling, self).__init__(**kwargs)
        down_sizes = [1, 2, 3, 6]
        mid_channels = in_channels // 4

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", F.identity)
                for i, down_size in enumerate(down_sizes):
                    setattr(self.branches, "branch{}".format(i + 2), PoolingBranch(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        in_size=in_size,
                        down_size=down_size))
                self.conv = conv1x1_block(
                    in_channels=(in_channels * 2),
                    out_channels=out_channels)

    def __call__(self, x):
        x = self.branches(x)
        x = self.conv(x)
        return x


class FeatureFusion(Chain):
    """
    Fast-SCNN specific feature fusion block.

    Parameters:
    ----------
    x_in_channels : int
        Number of high resolution (x) input channels.
    y_in_channels : int
        Number of low resolution (y) input channels.
    out_channels : int
        Number of output channels.
    x_in_size : tuple of 2 int or None
        Spatial size of high resolution (x) input image.
    """
    def __init__(self,
                 x_in_channels,
                 y_in_channels,
                 out_channels,
                 x_in_size,
                 **kwargs):
        super(FeatureFusion, self).__init__(**kwargs)
        self.x_in_size = x_in_size

        with self.init_scope():
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=x_in_size)
            self.low_dw_conv = dwconv3x3_block(
                in_channels=y_in_channels,
                out_channels=out_channels)
            self.low_pw_conv = conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=True,
                activation=None)
            self.high_conv = conv1x1_block(
                in_channels=x_in_channels,
                out_channels=out_channels,
                use_bias=True,
                activation=None)
            self.activ = F.relu

    def __call__(self, x, y):
        x_in_size = self.x_in_size if self.x_in_size is not None else x.shape[2:]
        y = self.up(y, x_in_size)
        y = self.low_dw_conv(y)
        y = self.low_pw_conv(y)
        x = self.high_conv(x)
        out = x + y
        return self.activ(out)


class Head(Chain):
    """
    Fast-SCNN head (classifier) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 classes):
        super(Head, self).__init__()
        with self.init_scope():
            self.conv1 = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels)
            self.conv2 = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels)
            self.dropout = partial(
                F.dropout,
                ratio=0.1)
            self.conv3 = conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class AuxHead(Chain):
    """
    Fast-SCNN auxiliary (after stem) head (classifier) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 classes):
        super(AuxHead, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.dropout = partial(
                F.dropout,
                ratio=0.1)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=classes,
                use_bias=True)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class FastSCNN(Chain):
    """
    Fast-SCNN from 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502.

    Parameters:
    ----------
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 1024)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(1024, 1024),
                 classes=19,
                 **kwargs):
        super(FastSCNN, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.init_scope():
            steam_channels = [32, 48, 64]
            self.stem = Stem(
                in_channels=in_channels,
                channels=steam_channels)
            in_channels = steam_channels[-1]
            feature_channels = [[64, 64, 64], [96, 96, 96], [128, 128, 128]]
            self.features = FeatureExtractor(
                in_channels=in_channels,
                channels=feature_channels)
            pool_out_size = (in_size[0] // 32, in_size[1] // 32) if fixed_size else None
            self.pool = FastPyramidPooling(
                in_channels=feature_channels[-1][-1],
                out_channels=feature_channels[-1][-1],
                in_size=pool_out_size)
            fusion_out_size = (in_size[0] // 8, in_size[1] // 8) if fixed_size else None
            fusion_out_channels = 128
            self.fusion = FeatureFusion(
                x_in_channels=steam_channels[-1],
                y_in_channels=feature_channels[-1][-1],
                out_channels=fusion_out_channels,
                x_in_size=fusion_out_size)
            self.head = Head(
                in_channels=fusion_out_channels,
                classes=classes)
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=in_size)

            if self.aux:
                self.aux_head = AuxHead(
                    in_channels=64,
                    mid_channels=64,
                    classes=classes)

    def __call__(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x = self.stem(x)
        y = self.features(x)
        y = self.pool(y)
        y = self.fusion(x, y)
        y = self.head(y)
        y = self.up(y, in_size)

        if self.aux:
            x = self.aux_head(x)
            x = self.up(x, in_size)
            return y, x
        return y


def get_fastscnn(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".chainer", "models"),
                 **kwargs):
    """
    Create Fast-SCNN model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    net = FastSCNN(
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


def fastscnn_cityscapes(classes=19, aux=True, **kwargs):
    """
    Fast-SCNN model for Cityscapes from 'Fast-SCNN: Fast Semantic Segmentation Network,'
    https://arxiv.org/abs/1902.04502.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_fastscnn(classes=classes, aux=aux, model_name="fastscnn_cityscapes", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (1024, 2048)
    aux = True
    fixed_size = False
    pretrained = False

    models = [
        (fastscnn_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size, aux=aux)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != fastscnn_cityscapes or weight_count == 1176278)
        else:
            assert (model != fastscnn_cityscapes or weight_count == 1138051)

        x = np.zeros((1, 3, in_size[0], in_size[1]), np.float32)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
