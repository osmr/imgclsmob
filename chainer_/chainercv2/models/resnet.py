"""
    ResNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['ResNet', 'resnet10', 'resnet12', 'resnet14', 'resnetbc14b', 'resnet16', 'resnet18_wd4', 'resnet18_wd2',
           'resnet18_w3d4', 'resnet18', 'resnet26', 'resnetbc26b', 'resnet34', 'resnetbc38b', 'resnet50', 'resnet50b',
           'resnet101', 'resnet101b', 'resnet152', 'resnet152b', 'resnet200', 'resnet200b', 'ResBlock', 'ResBottleneck',
           'ResUnit', 'ResInitBlock']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, conv7x7_block, SimpleSequential


class ResBlock(Chain):
    """
    Simple ResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_bias=False,
                 use_bn=True):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=use_bias,
                use_bn=use_bn)
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(Chain):
    """
    ResNet bottleneck block for residual path in ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 pad=1,
                 dilate=1,
                 conv1_stride=False,
                 bottleneck_factor=4):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=(stride if conv1_stride else 1))
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=(1 if conv1_stride else stride),
                pad=pad,
                dilate=dilate)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResUnit(Chain):
    """
    ResNet unit with residual connection.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
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
                 use_bias=False,
                 use_bn=True,
                 bottleneck=True,
                 conv1_stride=False):
        super(ResUnit, self).__init__()
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
                    stride=stride,
                    use_bias=use_bias,
                    use_bn=use_bn)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_bias=use_bias,
                    use_bn=use_bn,
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


class ResInitBlock(Chain):
    """
    ResNet specific initial block.

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
        super(ResInitBlock, self).__init__()
        with self.init_scope():
            self.conv = conv7x7_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2)
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


class ResNet(Chain):
    """
    ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

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
        super(ResNet, self).__init__()
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
                            setattr(stage, "unit{}".format(j + 1), ResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride))
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


def get_resnet(blocks,
               bottleneck=None,
               conv1_stride=True,
               width_scale=1.0,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create ResNet model with specific parameters.

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
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

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

    net = ResNet(
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


def resnet10(**kwargs):
    """
    ResNet-10 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=10, model_name="resnet10", **kwargs)


def resnet12(**kwargs):
    """
    ResNet-12 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=12, model_name="resnet12", **kwargs)


def resnet14(**kwargs):
    """
    ResNet-14 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=14, model_name="resnet14", **kwargs)


def resnetbc14b(**kwargs):
    """
    ResNet-BC-14b model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=14, bottleneck=True, conv1_stride=False, model_name="resnetbc14b", **kwargs)


def resnet16(**kwargs):
    """
    ResNet-16 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=16, model_name="resnet16", **kwargs)


def resnet18_wd4(**kwargs):
    """
    ResNet-18 model with 0.25 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.25, model_name="resnet18_wd4", **kwargs)


def resnet18_wd2(**kwargs):
    """
    ResNet-18 model with 0.5 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.5, model_name="resnet18_wd2", **kwargs)


def resnet18_w3d4(**kwargs):
    """
    ResNet-18 model with 0.75 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.75, model_name="resnet18_w3d4", **kwargs)


def resnet18(**kwargs):
    """
    ResNet-18 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="resnet18", **kwargs)


def resnet26(**kwargs):
    """
    ResNet-26 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=26, bottleneck=False, model_name="resnet26", **kwargs)


def resnetbc26b(**kwargs):
    """
    ResNet-BC-26b model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="resnetbc26b", **kwargs)


def resnet34(**kwargs):
    """
    ResNet-34 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="resnet34", **kwargs)


def resnetbc38b(**kwargs):
    """
    ResNet-BC-38b model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="resnetbc38b", **kwargs)


def resnet50(**kwargs):
    """
    ResNet-50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="resnet50", **kwargs)


def resnet50b(**kwargs):
    """
    ResNet-50 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, conv1_stride=False, model_name="resnet50b", **kwargs)


def resnet101(**kwargs):
    """
    ResNet-101 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="resnet101", **kwargs)


def resnet101b(**kwargs):
    """
    ResNet-101 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, conv1_stride=False, model_name="resnet101b", **kwargs)


def resnet152(**kwargs):
    """
    ResNet-152 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="resnet152", **kwargs)


def resnet152b(**kwargs):
    """
    ResNet-152 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, conv1_stride=False, model_name="resnet152b", **kwargs)


def resnet200(**kwargs):
    """
    ResNet-200 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, model_name="resnet200", **kwargs)


def resnet200b(**kwargs):
    """
    ResNet-200 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, conv1_stride=False, model_name="resnet200b", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        resnet10,
        resnet12,
        resnet14,
        resnetbc14b,
        resnet16,
        resnet18_wd4,
        resnet18_wd2,
        resnet18_w3d4,
        resnet18,
        resnet26,
        resnetbc26b,
        resnet34,
        resnetbc38b,
        resnet50,
        resnet50b,
        resnet101,
        resnet101b,
        resnet152,
        resnet152b,
        resnet200,
        resnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet10 or weight_count == 5418792)
        assert (model != resnet12 or weight_count == 5492776)
        assert (model != resnet14 or weight_count == 5788200)
        assert (model != resnetbc14b or weight_count == 10064936)
        assert (model != resnet16 or weight_count == 6968872)
        assert (model != resnet18_wd4 or weight_count == 3937400)
        assert (model != resnet18_wd2 or weight_count == 5804296)
        assert (model != resnet18_w3d4 or weight_count == 8476056)
        assert (model != resnet18 or weight_count == 11689512)
        assert (model != resnet26 or weight_count == 17960232)
        assert (model != resnetbc26b or weight_count == 15995176)
        assert (model != resnet34 or weight_count == 21797672)
        assert (model != resnetbc38b or weight_count == 21925416)
        assert (model != resnet50 or weight_count == 25557032)
        assert (model != resnet50b or weight_count == 25557032)
        assert (model != resnet101 or weight_count == 44549160)
        assert (model != resnet101b or weight_count == 44549160)
        assert (model != resnet152 or weight_count == 60192808)
        assert (model != resnet152b or weight_count == 60192808)
        assert (model != resnet200 or weight_count == 64673832)
        assert (model != resnet200b or weight_count == 64673832)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
