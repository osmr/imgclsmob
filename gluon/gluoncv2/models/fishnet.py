"""
    FishNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.
"""

__all__ = ['FishNet', 'fishnet99', 'fishnet150', 'ChannelSqueeze']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import Identity
from .common import pre_conv1x1_block, pre_conv3x3_block, conv1x1, SesquialteralHourglass, InterpolationBlock
from .preresnet import PreResActivation
from .senet import SEInitBlock


def channel_squeeze(x,
                    channels_per_group):
    """
    Channel squeeze operation.

    Parameters:
    ----------
    x : NDArray
        Input tensor.
    channels_per_group : int
        Number of channels per group.

    Returns:
    -------
    NDArray
        Resulted tensor.
    """
    return x.reshape((0, -4, channels_per_group, -1, -2)).sum(axis=2)


class ChannelSqueeze(HybridBlock):
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
                 groups,
                 **kwargs):
        super(ChannelSqueeze, self).__init__(**kwargs)
        assert (channels % groups == 0)
        self.channels_per_group = channels // groups

    def hybrid_forward(self, F, x):
        return channel_squeeze(x, self.channels_per_group)


class PreSEAttBlock(HybridBlock):
    """
    FishNet specific Squeeze-and-Excitation attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    reduction : int, default 16
        Squeeze reduction value.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 reduction=16,
                 **kwargs):
        super(PreSEAttBlock, self).__init__(**kwargs)
        mid_cannels = out_channels // reduction

        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.relu = nn.Activation("relu")
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=out_channels,
                use_bias=True)
            self.sigmoid = nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.relu(x)
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class FishBottleneck(HybridBlock):
    """
    FishNet bottleneck block for residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilation,
                 bn_use_global_stats,
                 **kwargs):
        super(FishBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                padding=dilation,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = pre_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class FishBlock(HybridBlock):
    """
    FishNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    squeeze : bool, default False
        Whether to use a channel squeeze operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 squeeze=False,
                 **kwargs):
        super(FishBlock, self).__init__(**kwargs)
        self.squeeze = squeeze
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = FishBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats)
            if self.squeeze:
                assert (in_channels // 2 == out_channels)
                self.c_squeeze = ChannelSqueeze(
                    channels=in_channels,
                    groups=2)
            elif self.resize_identity:
                self.identity_conv = pre_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        if self.squeeze:
            identity = self.c_squeeze(x)
        elif self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class DownUnit(HybridBlock):
    """
    FishNet down unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_use_global_stats,
                 **kwargs):
        super(DownUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.blocks = nn.HybridSequential(prefix="")
            for i, out_channels in enumerate(out_channels_list):
                self.blocks.add(FishBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = out_channels
            self.pool = nn.MaxPool2D(
                pool_size=2,
                strides=2)

    def hybrid_forward(self, F, x):
        x = self.blocks(x)
        x = self.pool(x)
        return x


class UpUnit(HybridBlock):
    """
    FishNet up unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 dilation=1,
                 bn_use_global_stats=False,
                 **kwargs):
        super(UpUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.blocks = nn.HybridSequential(prefix="")
            for i, out_channels in enumerate(out_channels_list):
                squeeze = (dilation > 1) and (i == 0)
                self.blocks.add(FishBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    squeeze=squeeze))
                in_channels = out_channels
            self.upsample = InterpolationBlock(scale_factor=2, bilinear=False)

    def hybrid_forward(self, F, x):
        x = self.blocks(x)
        x = self.upsample(x)
        return x


class SkipUnit(HybridBlock):
    """
    FishNet skip connection unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_use_global_stats,
                 **kwargs):
        super(SkipUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.blocks = nn.HybridSequential(prefix="")
            for i, out_channels in enumerate(out_channels_list):
                self.blocks.add(FishBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = out_channels

    def hybrid_forward(self, F, x):
        x = self.blocks(x)
        return x


class SkipAttUnit(HybridBlock):
    """
    FishNet skip connection unit with attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels for each block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_use_global_stats,
                 **kwargs):
        super(SkipAttUnit, self).__init__(**kwargs)
        mid_channels1 = in_channels // 2
        mid_channels2 = 2 * in_channels

        with self.name_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels1,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = pre_conv1x1_block(
                in_channels=mid_channels1,
                out_channels=mid_channels2,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats)
            in_channels = mid_channels2

            self.se = PreSEAttBlock(
                in_channels=mid_channels2,
                out_channels=out_channels_list[-1],
                bn_use_global_stats=bn_use_global_stats)

            self.blocks = nn.HybridSequential(prefix="")
            for i, out_channels in enumerate(out_channels_list):
                self.blocks.add(FishBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = out_channels

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        w = self.se(x)
        x = self.blocks(x)
        x = F.broadcast_add(F.broadcast_mul(x, w), w)
        return x


class FishFinalBlock(HybridBlock):
    """
    FishNet final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(FishFinalBlock, self).__init__(**kwargs)
        mid_channels = in_channels // 2

        with self.name_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.preactiv = PreResActivation(
                in_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.preactiv(x)
        return x


class FishNet(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(FishNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        depth = len(direct_channels[0])
        down1_channels = direct_channels[0]
        up_channels = direct_channels[1]
        down2_channels = direct_channels[2]
        skip1_channels = skip_channels[0]
        skip2_channels = skip_channels[1]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(SEInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels

            down1_seq = nn.HybridSequential(prefix="")
            skip1_seq = nn.HybridSequential(prefix="")
            for i in range(depth + 1):
                skip1_channels_list = skip1_channels[i]
                if i < depth:
                    skip1_seq.add(SkipUnit(
                        in_channels=in_channels,
                        out_channels_list=skip1_channels_list,
                        bn_use_global_stats=bn_use_global_stats))
                    down1_channels_list = down1_channels[i]
                    down1_seq.add(DownUnit(
                        in_channels=in_channels,
                        out_channels_list=down1_channels_list,
                        bn_use_global_stats=bn_use_global_stats))
                    in_channels = down1_channels_list[-1]
                else:
                    skip1_seq.add(SkipAttUnit(
                        in_channels=in_channels,
                        out_channels_list=skip1_channels_list,
                        bn_use_global_stats=bn_use_global_stats))
                    in_channels = skip1_channels_list[-1]

            up_seq = nn.HybridSequential(prefix="")
            skip2_seq = nn.HybridSequential(prefix="")
            for i in range(depth + 1):
                skip2_channels_list = skip2_channels[i]
                if i > 0:
                    in_channels += skip1_channels[depth - i][-1]
                if i < depth:
                    skip2_seq.add(SkipUnit(
                        in_channels=in_channels,
                        out_channels_list=skip2_channels_list,
                        bn_use_global_stats=bn_use_global_stats))
                    up_channels_list = up_channels[i]
                    dilation = 2 ** i
                    up_seq.add(UpUnit(
                        in_channels=in_channels,
                        out_channels_list=up_channels_list,
                        dilation=dilation,
                        bn_use_global_stats=bn_use_global_stats))
                    in_channels = up_channels_list[-1]
                else:
                    skip2_seq.add(Identity())

            down2_seq = nn.HybridSequential(prefix="")
            for i in range(depth):
                down2_channels_list = down2_channels[i]
                down2_seq.add(DownUnit(
                    in_channels=in_channels,
                    out_channels_list=down2_channels_list,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = down2_channels_list[-1] + skip2_channels[depth - 1 - i][-1]

            self.features.add(SesquialteralHourglass(
                down1_seq=down1_seq,
                skip1_seq=skip1_seq,
                up_seq=up_seq,
                skip2_seq=skip2_seq,
                down2_seq=down2_seq))
            self.features.add(FishFinalBlock(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = in_channels // 2
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_fishnet(blocks,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def fishnet99(**kwargs):
    """
    FishNet-99 model from 'FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction,'
    http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_fishnet(blocks=150, model_name="fishnet150", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        fishnet99,
        fishnet150,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fishnet99 or weight_count == 16628904)
        assert (model != fishnet150 or weight_count == 24959400)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
