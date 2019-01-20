"""
    MobileNetV2, implemented in Chainer.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""

__all__ = ['MobileNetV2', 'mobilenetv2_w1', 'mobilenetv2_w3d4', 'mobilenetv2_wd2', 'mobilenetv2_wd4']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import ReLU6, conv1x1_block, conv3x3_block, dwconv3x3_block, SimpleSequential


class LinearBottleneck(Chain):
    """
    So-called 'Linear Bottleneck' layer. It is used as a MobileNetV2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the second convolution layer.
    expansion : bool
        Whether do expansion of channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion):
        super(LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=ReLU6())
            self.conv2 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=ReLU6())
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None,
                activate=False)

    def __call__(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV2(Chain):
    """
    MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
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
                 final_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(MobileNetV2, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    stride=2,
                    activation=ReLU6()))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            expansion = (i != 0) or (j != 0)
                            setattr(stage, "unit{}".format(j + 1), LinearBottleneck(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                expansion=expansion))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'final_block', conv1x1_block(
                    in_channels=in_channels,
                    out_channels=final_block_channels,
                    activation=ReLU6()))
                in_channels = final_block_channels
                setattr(self.features, 'final_pool', partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, 'final_conv', L.Convolution2D(
                    in_channels=in_channels,
                    out_channels=classes,
                    ksize=1,
                    nobias=True))
                setattr(self.output, 'final_flatten', partial(
                    F.reshape,
                    shape=(-1, classes)))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenetv2(width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join('~', '.chainer', 'models'),
                    **kwargs):
    """
    Create MobileNetV2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [[]])

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    net = MobileNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
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


def mobilenetv2_w1(**kwargs):
    """
    1.0 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=1.0, model_name="mobilenetv2_w1", **kwargs)


def mobilenetv2_w3d4(**kwargs):
    """
    0.75 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=0.75, model_name="mobilenetv2_w3d4", **kwargs)


def mobilenetv2_wd2(**kwargs):
    """
    0.5 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=0.5, model_name="mobilenetv2_wd2", **kwargs)


def mobilenetv2_wd4(**kwargs):
    """
    0.25 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv2(width_scale=0.25, model_name="mobilenetv2_wd4", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        mobilenetv2_w1,
        mobilenetv2_w3d4,
        mobilenetv2_wd2,
        mobilenetv2_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenetv2_w1 or weight_count == 3504960)
        assert (model != mobilenetv2_w3d4 or weight_count == 2627592)
        assert (model != mobilenetv2_wd2 or weight_count == 1964736)
        assert (model != mobilenetv2_wd4 or weight_count == 1516392)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
