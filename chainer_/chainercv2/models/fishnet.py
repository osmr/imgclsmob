"""
    FishNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.
"""

__all__ = ['FishNet', 'fishnet99', 'fishnet150']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import pre_conv1x1_block, pre_conv3x3_block, conv1x1, SesquialteralHourglass, SimpleSequential
from .preresnet import PreResActivation
from .senet import SEInitBlock


def channel_squeeze(x,
                    groups):
    """
    Channel squeeze operation.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input variable.
    groups : int
        Number of groups.

    Returns:
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted variable.
    """
    batch, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, channels_per_group, groups, height, width))
    x = F.sum(x, axis=2)
    return x


class ChannelSqueeze(Chain):
    """
    Channel squeeze layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups):
        super(ChannelSqueeze, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def __call__(self, x):
        return channel_squeeze(x, self.groups)


class InterpolationBlock(Chain):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : int
        Multiplier for spatial size.
    """
    def __init__(self,
                 scale_factor):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor

    def __call__(self, x):
        return F.unpooling_2d(
            x=x,
            ksize=self.scale_factor,
            cover_all=False)


class PreSEAttBlock(Chain):
    """
    FishNet specific Squeeze-and-Excitation attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int, default 16
        Squeeze reduction value.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=16):
        super(PreSEAttBlock, self).__init__()
        mid_cannels = out_channels // reduction

        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=out_channels,
                use_bias=True)

    def __call__(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = F.average_pooling_2d(x, ksize=x.shape[2:])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class FishBottleneck(Chain):
    """
    FishNet bottleneck block for residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    dilate : int or tuple/list of 2 int
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilate):
        super(FishBottleneck, self).__init__()
        mid_channels = out_channels // 4

        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                pad=dilate,
                dilate=dilate)
            self.conv3 = pre_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class FishBlock(Chain):
    """
    FishNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    squeeze : bool, default False
        Whether to use a channel squeeze operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilate=1,
                 squeeze=False):
        super(FishBlock, self).__init__()
        self.squeeze = squeeze
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = FishBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilate=dilate)
            if self.squeeze:
                assert (in_channels // 2 == out_channels)
                self.c_squeeze = ChannelSqueeze(
                    channels=in_channels,
                    groups=2)
            elif self.resize_identity:
                self.identity_conv = pre_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)

    def __call__(self, x):
        if self.squeeze:
            identity = self.c_squeeze(x)
        elif self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class DownUnit(Chain):
    """
    FishNet down unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(DownUnit, self).__init__()
        with self.init_scope():
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i, out_channels in enumerate(out_channels_list):
                    setattr(self.blocks, "block{}".format(i + 1), FishBlock(
                        in_channels=in_channels,
                        out_channels=out_channels))
                    in_channels = out_channels
            self.pool = partial(
                F.max_pooling_2d,
                ksize=2,
                stride=2,
                cover_all=False)

    def __call__(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return x


class UpUnit(Chain):
    """
    FishNet up unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 dilate=1):
        super(UpUnit, self).__init__()
        with self.init_scope():
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i, out_channels in enumerate(out_channels_list):
                    squeeze = (dilate > 1) and (i == 0)
                    setattr(self.blocks, "block{}".format(i + 1), FishBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dilate=dilate,
                        squeeze=squeeze))
                    in_channels = out_channels
            self.upsample = InterpolationBlock(scale_factor=2)

    def __call__(self, x):
        x = self.blocks(x)
        x = self.upsample(x)
        return x


class SkipUnit(Chain):
    """
    FishNet skip connection unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(SkipUnit, self).__init__()
        with self.init_scope():
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i, out_channels in enumerate(out_channels_list):
                    setattr(self.blocks, "block{}".format(i + 1), FishBlock(
                        in_channels=in_channels,
                        out_channels=out_channels))
                    in_channels = out_channels

    def __call__(self, x):
        x = self.blocks(x)
        return x


class SkipAttUnit(Chain):
    """
    FishNet skip connection unit with attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(SkipAttUnit, self).__init__()
        mid_channels1 = in_channels // 2
        mid_channels2 = 2 * in_channels

        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels1)
            self.conv2 = pre_conv1x1_block(
                in_channels=mid_channels1,
                out_channels=mid_channels2,
                use_bias=True)
            in_channels = mid_channels2

            self.se = PreSEAttBlock(
                in_channels=mid_channels2,
                out_channels=out_channels_list[-1])

            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i, out_channels in enumerate(out_channels_list):
                    setattr(self.blocks, "block{}".format(i + 1), FishBlock(
                        in_channels=in_channels,
                        out_channels=out_channels))
                    in_channels = out_channels

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        w = self.se(x)
        x = self.blocks(x)
        x = x * w + w
        return x


class FishFinalBlock(Chain):
    """
    FishNet final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(FishFinalBlock, self).__init__()
        mid_channels = in_channels // 2

        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.preactiv = PreResActivation(
                in_channels=mid_channels)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.preactiv(x)
        return x


class FishNet(Chain):
    """
    FishNet model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    direct_channels : list of list of list of int
        Number of output channels for each unit along the straight path.
    skip_channels : list of list of list of int
        Number of output channels for each skip connection unit.
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
                 direct_channels,
                 skip_channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(FishNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        depth = len(direct_channels[0])
        down1_channels = direct_channels[0]
        up_channels = direct_channels[1]
        down2_channels = direct_channels[2]
        skip1_channels = skip_channels[0]
        skip2_channels = skip_channels[1]

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", SEInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels

                down1_seq = SimpleSequential()
                skip1_seq = SimpleSequential()
                for i in range(depth + 1):
                    skip1_channels_list = skip1_channels[i]
                    if i < depth:
                        with skip1_seq.init_scope():
                            setattr(skip1_seq, "unit{}".format(i + 1), SkipUnit(
                                in_channels=in_channels,
                                out_channels_list=skip1_channels_list))
                        down1_channels_list = down1_channels[i]
                        with down1_seq.init_scope():
                            setattr(down1_seq, "unit{}".format(i + 1), DownUnit(
                                in_channels=in_channels,
                                out_channels_list=down1_channels_list))
                        in_channels = down1_channels_list[-1]
                    else:
                        with skip1_seq.init_scope():
                            setattr(skip1_seq, "unit{}".format(i + 1), SkipAttUnit(
                                in_channels=in_channels,
                                out_channels_list=skip1_channels_list))
                        in_channels = skip1_channels_list[-1]

                up_seq = SimpleSequential()
                skip2_seq = SimpleSequential()
                for i in range(depth + 1):
                    skip2_channels_list = skip2_channels[i]
                    if i > 0:
                        in_channels += skip1_channels[depth - i][-1]
                    if i < depth:
                        with skip2_seq.init_scope():
                            setattr(skip2_seq, "unit{}".format(i + 1), SkipUnit(
                                in_channels=in_channels,
                                out_channels_list=skip2_channels_list))
                        up_channels_list = up_channels[i]
                        dilate = 2 ** i
                        with up_seq.init_scope():
                            setattr(up_seq, "unit{}".format(i + 1), UpUnit(
                                in_channels=in_channels,
                                out_channels_list=up_channels_list,
                                dilate=dilate))
                        in_channels = up_channels_list[-1]
                    else:
                        with skip2_seq.init_scope():
                            setattr(skip2_seq, "unit{}".format(i + 1), F.identity)

                down2_seq = SimpleSequential()
                with down2_seq.init_scope():
                    for i in range(depth):
                        down2_channels_list = down2_channels[i]
                        setattr(down2_seq, "unit{}".format(i + 1), DownUnit(
                            in_channels=in_channels,
                            out_channels_list=down2_channels_list))
                        in_channels = down2_channels_list[-1] + skip2_channels[depth - 1 - i][-1]

                setattr(self.features, "hg", SesquialteralHourglass(
                    down1_seq=down1_seq,
                    skip1_seq=skip1_seq,
                    up_seq=up_seq,
                    skip2_seq=skip2_seq,
                    down2_seq=down2_seq))
                setattr(self.features, "final_block", FishFinalBlock(in_channels=in_channels))
                in_channels = in_channels // 2
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "final_conv", conv1x1(
                    in_channels=in_channels,
                    out_channels=classes,
                    use_bias=True))
                setattr(self.output, "final_flatten", partial(
                    F.reshape,
                    shape=(-1, classes)))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_fishnet(blocks,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".chainer", "models"),
                **kwargs):
    """
    Create FishNet model with specific parameters.

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

    if blocks == 99:
        direct_layers = [[2, 2, 6], [1, 1, 1], [1, 2, 2]]
        skip_layers = [[1, 1, 1, 2], [4, 1, 1, 0]]
    elif blocks == 150:
        direct_layers = [[2, 4, 8], [2, 2, 2], [2, 2, 4]]
        skip_layers = [[2, 2, 2, 4], [4, 2, 2, 0]]
    else:
        raise ValueError("Unsupported FishNet with number of blocks: {}".format(blocks))

    direct_channels_per_layers = [[128, 256, 512], [512, 384, 256], [320, 832, 1600]]
    skip_channels_per_layers = [[64, 128, 256, 512], [512, 768, 512, 0]]

    direct_channels = [[[b] * c for (b, c) in zip(*a)] for a in
                       ([(ci, li) for (ci, li) in zip(direct_channels_per_layers, direct_layers)])]
    skip_channels = [[[b] * c for (b, c) in zip(*a)] for a in
                     ([(ci, li) for (ci, li) in zip(skip_channels_per_layers, skip_layers)])]

    init_block_channels = 64

    net = FishNet(
        direct_channels=direct_channels,
        skip_channels=skip_channels,
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


def fishnet99(**kwargs):
    """
    FishNet-99 model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_fishnet(blocks=99, model_name="fishnet99", **kwargs)


def fishnet150(**kwargs):
    """
    FishNet-150 model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_fishnet(blocks=150, model_name="fishnet150", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        fishnet99,
        fishnet150,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fishnet99 or weight_count == 16628904)
        assert (model != fishnet150 or weight_count == 24959400)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
