"""
    ResAttNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.
"""

__all__ = ['ResAttNet', 'resattnet56', 'resattnet92', 'resattnet128', 'resattnet164', 'resattnet200', 'resattnet236',
           'resattnet452']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv7x7_block, pre_conv1x1_block, pre_conv3x3_block, Hourglass


class PreResBottleneck(HybridBlock):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 **kwargs):
        super(PreResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                return_preact=True)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                strides=strides)
            self.conv3 = pre_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x, x_pre_activ


class ResBlock(HybridBlock):
    """
    Residual block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 bn_use_global_stats=False,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = PreResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)

    def hybrid_forward(self, F, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class InterpolationBlock(HybridBlock):
    """
    Interpolation block.

    Parameters:
    ----------
    size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 size,
                 **kwargs):
        super(InterpolationBlock, self).__init__(**kwargs)
        self.size = size

    def hybrid_forward(self, F, x):
        return F.contrib.BilinearResize2D(x, height=self.size[0], width=self.size[1])


class DoubleSkipBlock(HybridBlock):
    """
    Double skip connection block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(DoubleSkipBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.skip1 = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = x + self.skip1(x)
        return x


class ResBlockSequence(HybridBlock):
    """
    Sequence of residual blocks with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of sequence.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length,
                 bn_use_global_stats,
                 **kwargs):
        super(ResBlockSequence, self).__init__(**kwargs)
        with self.name_scope():
            self.blocks = nn.HybridSequential(prefix="")
            for i in range(length):
                self.blocks.add(ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.blocks(x)
        return x


class DownAttBlock(HybridBlock):
    """
    Down sub-block for hourglass of attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of residual blocks list.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length,
                 bn_use_global_stats,
                 **kwargs):
        super(DownAttBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)
            self.res_blocks = ResBlockSequence(
                in_channels=in_channels,
                out_channels=out_channels,
                length=length,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.res_blocks(x)
        return x


class UpAttBlock(HybridBlock):
    """
    Up sub-block for hourglass of attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of residual blocks list.
    size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length,
                 size,
                 bn_use_global_stats,
                 **kwargs):
        super(UpAttBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.res_blocks = ResBlockSequence(
                in_channels=in_channels,
                out_channels=out_channels,
                length=length,
                bn_use_global_stats=bn_use_global_stats)
            self.upsample = InterpolationBlock(size)

    def hybrid_forward(self, F, x):
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


class MiddleAttBlock(HybridBlock):
    """
    Middle sub-block for attention block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats,
                 **kwargs):
        super(MiddleAttBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=channels,
                out_channels=channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = pre_conv1x1_block(
                in_channels=channels,
                out_channels=channels,
                bn_use_global_stats=bn_use_global_stats)
            self.sigmoid = nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class AttBlock(HybridBlock):
    """
    Attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hourglass_depth : int
        Depth of hourglass block.
    att_scales : list of int
        Attention block specific scales.
    in_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hourglass_depth,
                 att_scales,
                 in_size,
                 bn_use_global_stats,
                 **kwargs):
        super(AttBlock, self).__init__(**kwargs)
        assert (len(att_scales) == 3)
        scale_factor = 2
        scale_p, scale_t, scale_r = att_scales

        with self.name_scope():
            self.init_blocks = ResBlockSequence(
                in_channels=in_channels,
                out_channels=out_channels,
                length=scale_p,
                bn_use_global_stats=bn_use_global_stats)

            down_seq = nn.HybridSequential(prefix="")
            up_seq = nn.HybridSequential(prefix="")
            skip_seq = nn.HybridSequential(prefix="")
            for i in range(hourglass_depth):
                down_seq.add(DownAttBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    length=scale_r,
                    bn_use_global_stats=bn_use_global_stats))
                up_seq.add(UpAttBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    length=scale_r,
                    size=in_size,
                    bn_use_global_stats=bn_use_global_stats))
                in_size = tuple([x // scale_factor for x in in_size])
                if i == 0:
                    skip_seq.add(ResBlockSequence(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        length=scale_t,
                        bn_use_global_stats=bn_use_global_stats))
                else:
                    skip_seq.add(DoubleSkipBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bn_use_global_stats=bn_use_global_stats))
            self.hg = Hourglass(
                down_seq=down_seq,
                up_seq=up_seq,
                skip_seq=skip_seq,
                return_first_skip=True)

            self.middle_block = MiddleAttBlock(
                channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.final_block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.init_blocks(x)
        x, y = self.hg(x)
        x = self.middle_block(x)
        x = (1 + x) * y
        x = self.final_block(x)
        return x


class ResAttInitBlock(HybridBlock):
    """
    ResAttNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(ResAttInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv7x7_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PreActivation(HybridBlock):
    """
    Pre-activation block without convolution layer. It's used by itself as the final block in PreResNet.

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
        super(PreActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ResAttNet(HybridBlock):
    """
    ResAttNet model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    attentions : list of list of int
        Whether to use a attention unit or residual one.
    att_scales : list of int
        Attention block specific scales.
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
                 channels,
                 init_block_channels,
                 attentions,
                 att_scales,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(ResAttNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(ResAttInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            in_size = tuple([x // 4 for x in in_size])
            for i, channels_per_stage in enumerate(channels):
                hourglass_depth = len(channels) - 1 - i
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        if attentions[i][j]:
                            stage.add(AttBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                hourglass_depth=hourglass_depth,
                                att_scales=att_scales,
                                in_size=in_size,
                                bn_use_global_stats=bn_use_global_stats))
                        else:
                            stage.add(ResBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                        in_size = tuple([x // strides for x in in_size])
                self.features.add(stage)
            self.features.add(PreActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_resattnet(blocks,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create ResAttNet model with specific parameters.

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
    if blocks == 56:
        att_layers = [1, 1, 1]
        att_scales = [1, 2, 1]
    elif blocks == 92:
        att_layers = [1, 2, 3]
        att_scales = [1, 2, 1]
    elif blocks == 128:
        att_layers = [2, 3, 4]
        att_scales = [1, 2, 1]
    elif blocks == 164:
        att_layers = [3, 4, 5]
        att_scales = [1, 2, 1]
    elif blocks == 200:
        att_layers = [4, 5, 6]
        att_scales = [1, 2, 1]
    elif blocks == 236:
        att_layers = [5, 6, 7]
        att_scales = [1, 2, 1]
    elif blocks == 452:
        att_layers = [5, 6, 7]
        att_scales = [2, 4, 3]
    else:
        raise ValueError("Unsupported ResAttNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    layers = att_layers + [2]
    channels = [[ci] * (li + 1) for (ci, li) in zip(channels_per_layers, layers)]
    attentions = [[0] + [1] * li for li in att_layers] + [[0] * 3]

    net = ResAttNet(
        channels=channels,
        init_block_channels=init_block_channels,
        attentions=attentions,
        att_scales=att_scales,
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


def resattnet56(**kwargs):
    """
    ResAttNet-56 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=56, model_name="resattnet56", **kwargs)


def resattnet92(**kwargs):
    """
    ResAttNet-92 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=92, model_name="resattnet92", **kwargs)


def resattnet128(**kwargs):
    """
    ResAttNet-128 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=128, model_name="resattnet128", **kwargs)


def resattnet164(**kwargs):
    """
    ResAttNet-164 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=164, model_name="resattnet164", **kwargs)


def resattnet200(**kwargs):
    """
    ResAttNet-200 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=200, model_name="resattnet200", **kwargs)


def resattnet236(**kwargs):
    """
    ResAttNet-236 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=236, model_name="resattnet236", **kwargs)


def resattnet452(**kwargs):
    """
    ResAttNet-452 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=452, model_name="resattnet452", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        resattnet56,
        resattnet92,
        resattnet128,
        resattnet164,
        resattnet200,
        resattnet236,
        resattnet452,
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
        assert (model != resattnet56 or weight_count == 31810728)
        assert (model != resattnet92 or weight_count == 52466344)
        assert (model != resattnet128 or weight_count == 65294504)
        assert (model != resattnet164 or weight_count == 78122664)
        assert (model != resattnet200 or weight_count == 90950824)
        assert (model != resattnet236 or weight_count == 103778984)
        assert (model != resattnet452 or weight_count == 182285224)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
