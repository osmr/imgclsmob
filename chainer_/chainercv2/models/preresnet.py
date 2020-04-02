"""
    PreResNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
"""

__all__ = ['PreResNet', 'preresnet10', 'preresnet12', 'preresnet14', 'preresnetbc14b', 'preresnet16', 'preresnet18_wd4',
           'preresnet18_wd2', 'preresnet18_w3d4', 'preresnet18', 'preresnet26', 'preresnetbc26b', 'preresnet34',
           'preresnetbc38b', 'preresnet50', 'preresnet50b', 'preresnet101', 'preresnet101b', 'preresnet152',
           'preresnet152b', 'preresnet200', 'preresnet200b', 'preresnet269b', 'PreResBlock', 'PreResBottleneck',
           'PreResUnit', 'PreResInitBlock', 'PreResActivation']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import pre_conv1x1_block, pre_conv3x3_block, conv1x1, SimpleSequential


class PreResBlock(Chain):
    """
    Simple PreResNet block for residual path in PreResNet unit.

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
        super(PreResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=use_bias,
                use_bn=use_bn,
                return_preact=True)
            self.conv2 = pre_conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn)

    def __call__(self, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        return x, x_pre_activ


class PreResBottleneck(Chain):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 conv1_stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=(stride if conv1_stride else 1),
                return_preact=True)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=(1 if conv1_stride else stride))
            self.conv3 = pre_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x, x_pre_activ


class PreResUnit(Chain):
    """
    PreResNet unit with residual connection.

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
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_bias=False,
                 use_bn=True,
                 bottleneck=True,
                 conv1_stride=False):
        super(PreResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            if bottleneck:
                self.body = PreResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv1_stride=conv1_stride)
            else:
                self.body = PreResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_bias=use_bias,
                    use_bn=use_bn)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_bias=use_bias)

    def __call__(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class PreResInitBlock(Chain):
    """
    PreResNet specific initial block.

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
        super(PreResInitBlock, self).__init__()
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


class PreResActivation(Chain):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreResActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            self.activ = F.relu

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class PreResNet(Chain):
    """
    PreResNet model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

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
        super(PreResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", PreResInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), PreResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(
                    in_channels=in_channels))
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


def get_preresnet(blocks,
                  bottleneck=None,
                  conv1_stride=True,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create PreResNet model with specific parameters.

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
    elif blocks == 269:
        layers = [3, 30, 48, 8]
    else:
        raise ValueError("Unsupported PreResNet with number of blocks: {}".format(blocks))

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

    net = PreResNet(
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


def preresnet10(**kwargs):
    """
    PreResNet-10 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=10, model_name="preresnet10", **kwargs)


def preresnet12(**kwargs):
    """
    PreResNet-12 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=12, model_name="preresnet12", **kwargs)


def preresnet14(**kwargs):
    """
    PreResNet-14 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=14, model_name="preresnet14", **kwargs)


def preresnetbc14b(**kwargs):
    """
    PreResNet-BC-14b model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=14, bottleneck=True, conv1_stride=False, model_name="preresnetbc14b", **kwargs)


def preresnet16(**kwargs):
    """
    PreResNet-16 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=16, model_name="preresnet16", **kwargs)


def preresnet18_wd4(**kwargs):
    """
    PreResNet-18 model with 0.25 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.25, model_name="preresnet18_wd4", **kwargs)


def preresnet18_wd2(**kwargs):
    """
    PreResNet-18 model with 0.5 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.5, model_name="preresnet18_wd2", **kwargs)


def preresnet18_w3d4(**kwargs):
    """
    PreResNet-18 model with 0.75 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.75, model_name="preresnet18_w3d4", **kwargs)


def preresnet18(**kwargs):
    """
    PreResNet-18 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, model_name="preresnet18", **kwargs)


def preresnet26(**kwargs):
    """
    PreResNet-26 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=26, bottleneck=False, model_name="preresnet26", **kwargs)


def preresnetbc26b(**kwargs):
    """
    PreResNet-BC-26b model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="preresnetbc26b", **kwargs)


def preresnet34(**kwargs):
    """
    PreResNet-34 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=34, model_name="preresnet34", **kwargs)


def preresnetbc38b(**kwargs):
    """
    PreResNet-BC-38b model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="preresnetbc38b", **kwargs)


def preresnet50(**kwargs):
    """
    PreResNet-50 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, model_name="preresnet50", **kwargs)


def preresnet50b(**kwargs):
    """
    PreResNet-50 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, conv1_stride=False, model_name="preresnet50b", **kwargs)


def preresnet101(**kwargs):
    """
    PreResNet-101 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, model_name="preresnet101", **kwargs)


def preresnet101b(**kwargs):
    """
    PreResNet-101 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, conv1_stride=False, model_name="preresnet101b", **kwargs)


def preresnet152(**kwargs):
    """
    PreResNet-152 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, model_name="preresnet152", **kwargs)


def preresnet152b(**kwargs):
    """
    PreResNet-152 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, conv1_stride=False, model_name="preresnet152b", **kwargs)


def preresnet200(**kwargs):
    """
    PreResNet-200 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, model_name="preresnet200", **kwargs)


def preresnet200b(**kwargs):
    """
    PreResNet-200 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, conv1_stride=False, model_name="preresnet200b", **kwargs)


def preresnet269b(**kwargs):
    """
    PreResNet-269 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=269, conv1_stride=False, model_name="preresnet269b", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        preresnet10,
        preresnet12,
        preresnet14,
        preresnetbc14b,
        preresnet16,
        preresnet18_wd4,
        preresnet18_wd2,
        preresnet18_w3d4,
        preresnet18,
        preresnet26,
        preresnetbc26b,
        preresnet34,
        preresnetbc38b,
        preresnet50,
        preresnet50b,
        preresnet101,
        preresnet101b,
        preresnet152,
        preresnet152b,
        preresnet200,
        preresnet200b,
        preresnet269b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != preresnet10 or weight_count == 5417128)
        assert (model != preresnet12 or weight_count == 5491112)
        assert (model != preresnet14 or weight_count == 5786536)
        assert (model != preresnetbc14b or weight_count == 10057384)
        assert (model != preresnet16 or weight_count == 6967208)
        assert (model != preresnet18_wd4 or weight_count == 3935960)
        assert (model != preresnet18_wd2 or weight_count == 5802440)
        assert (model != preresnet18_w3d4 or weight_count == 8473784)
        assert (model != preresnet18 or weight_count == 11687848)
        assert (model != preresnet26 or weight_count == 17958568)
        assert (model != preresnetbc26b or weight_count == 15987624)
        assert (model != preresnet34 or weight_count == 21796008)
        assert (model != preresnetbc38b or weight_count == 21917864)
        assert (model != preresnet50 or weight_count == 25549480)
        assert (model != preresnet50b or weight_count == 25549480)
        assert (model != preresnet101 or weight_count == 44541608)
        assert (model != preresnet101b or weight_count == 44541608)
        assert (model != preresnet152 or weight_count == 60185256)
        assert (model != preresnet152b or weight_count == 60185256)
        assert (model != preresnet200 or weight_count == 64666280)
        assert (model != preresnet200b or weight_count == 64666280)
        assert (model != preresnet269b or weight_count == 102065832)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
