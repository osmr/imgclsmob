"""
    ResNet(A) with average downsampling for ImageNet-1K, implemented in Chainer.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['ResNetA', 'resneta10', 'resnetabc14b', 'resneta18', 'resneta50b', 'resneta101b', 'resneta152b']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, SimpleSequential
from .resnet import ResBlock, ResBottleneck
from .senet import SEInitBlock


class ResADownBlock(Chain):
    """
    ResNet(A) downsample block for the identity branch of a residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilate=1,
                 **kwargs):
        super(ResADownBlock, self).__init__(**kwargs)
        with self.init_scope():
            # self.pool = partial(
            #     F.average_pooling_2d,
            #     ksize=(stride if dilate == 1 else 1),
            #     stride=(stride if dilate == 1 else 1))
            self.pool = partial(
                F.average_pooling_nd,
                ksize=(stride if dilate == 1 else 1),
                stride=(stride if dilate == 1 else 1),
                pad_value=None)
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ResAUnit(Chain):
    """
    ResNet(A) unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 pad=1,
                 dilate=1,
                 bottleneck=True,
                 conv1_stride=False,
                 **kwargs):
        super(ResAUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    pad=pad,
                    dilate=dilate,
                    conv1_stride=conv1_stride)
            else:
                self.body = ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)
            if self.resize_identity:
                self.identity_block = ResADownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilate=dilate)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_block(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ResNetA(Chain):
    """
    ResNet(A) with average downsampling model from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

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
    dilated : bool, default False
        Whether to use dilation.
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
                 dilated=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(ResNetA, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", SEInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            if dilated:
                                stride = 2 if ((j == 0) and (i != 0) and (i < 2)) else 1
                                dilate = (2 ** max(0, i - 1 - int(j == 0)))
                            else:
                                stride = 2 if (j == 0) and (i != 0) else 1
                                dilate = 1
                            setattr(stage, "unit{}".format(j + 1), ResAUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                pad=dilate,
                                dilate=dilate,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=(in_size[0] // 32, in_size[1] // 32)))

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


def get_resneta(blocks,
                bottleneck=None,
                conv1_stride=True,
                width_scale=1.0,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".chainer", "models"),
                **kwargs):
    """
    Create ResNet(A) with average downsampling model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet(A) with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = ResNetA(
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


def resneta10(**kwargs):
    """
    ResNet(A)-10 with average downsampling model from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=10, model_name="resneta10", **kwargs)


def resnetabc14b(**kwargs):
    """
    ResNet(A)-BC-14b with average downsampling model from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=14, bottleneck=True, conv1_stride=False, model_name="resnetabc14b", **kwargs)


def resneta18(**kwargs):
    """
    ResNet(A)-18 with average downsampling model from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=18, model_name="resneta18", **kwargs)


def resneta50b(**kwargs):
    """
    ResNet(A)-50 with average downsampling model with stride at the second convolution in bottleneck block
    from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=50, conv1_stride=False, model_name="resneta50b", **kwargs)


def resneta101b(**kwargs):
    """
    ResNet(A)-101 with average downsampling model with stride at the second convolution in bottleneck
    block from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=101, conv1_stride=False, model_name="resneta101b", **kwargs)


def resneta152b(**kwargs):
    """
    ResNet(A)-152 with average downsampling model with stride at the second convolution in bottleneck
    block from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=152, conv1_stride=False, model_name="resneta152b", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        resneta10,
        resnetabc14b,
        resneta18,
        resneta50b,
        resneta101b,
        resneta152b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resneta10 or weight_count == 5438024)
        assert (model != resnetabc14b or weight_count == 10084168)
        assert (model != resneta18 or weight_count == 11708744)
        assert (model != resneta50b or weight_count == 25576264)
        assert (model != resneta101b or weight_count == 44568392)
        assert (model != resneta152b or weight_count == 60212040)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
