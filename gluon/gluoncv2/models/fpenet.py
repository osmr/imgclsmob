"""
    FPENet for image segmentation, implemented in Gluon.
    Original paper: 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.
"""

__all__ = ['FPENet', 'fpenet_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv1x1_block, conv3x3_block, SEBlock, InterpolationBlock, MultiOutputSequential


class FPEBlock(HybridBlock):
    """
    FPENet block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(FPEBlock, self).__init__(**kwargs)
        dilations = [1, 2, 4, 8]
        assert (channels % len(dilations) == 0)
        mid_channels = channels // len(dilations)

        with self.name_scope():
            self.blocks = nn.HybridSequential(prefix="")
            for i, dilation in enumerate(dilations):
                self.blocks.add(conv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    groups=mid_channels,
                    dilation=dilation,
                    padding=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))

    def hybrid_forward(self, F, x):
        xs = F.split(x, axis=1, num_outputs=len(self.blocks._children))
        ys = []
        for bi, xsi in zip(self.blocks._children.values(), xs):
            if len(ys) == 0:
                ys.append(bi(xsi))
            else:
                ys.append(bi(xsi + ys[-1]))
        x = F.concat(*ys, dim=1)
        return x


class FPEUnit(HybridBlock):
    """
    FPENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int
        Bottleneck factor.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor,
                 use_se,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(FPEUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)
        self.use_se = use_se
        mid1_channels = in_channels * bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid1_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.block = FPEBlock(channels=mid1_channels)
            self.conv2 = conv1x1_block(
                in_channels=mid1_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            if self.use_se:
                self.se = SEBlock(channels=out_channels)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.block(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class FPEStage(HybridBlock):
    """
    FPENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 use_se,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(FPEStage, self).__init__(**kwargs)
        self.use_block = (layers > 1)

        with self.name_scope():
            if self.use_block:
                self.down = FPEUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=2,
                    bottleneck_factor=4,
                    use_se=use_se,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)
                self.blocks = nn.HybridSequential(prefix="")
                for i in range(layers - 1):
                    self.blocks.add(FPEUnit(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        strides=1,
                        bottleneck_factor=1,
                        use_se=use_se,
                        bn_use_global_stats=bn_use_global_stats,
                        bn_cudnn_off=bn_cudnn_off))
            else:
                self.down = FPEUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=1,
                    bottleneck_factor=1,
                    use_se=use_se,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.down(x)
        if self.use_block:
            y = self.blocks(x)
            x = x + y
        return x


class MEUBlock(HybridBlock):
    """
    FPENet specific mutual embedding upsample (MEU) block.

    Parameters:
    ----------
    in_channels_high : int
        Number of input channels for x_high.
    in_channels_low : int
        Number of input channels for x_low.
    out_channels : int
        Number of output channels.
    out_size : tuple of 2 int
        Spatial size of output image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels_high,
                 in_channels_low,
                 out_channels,
                 out_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(MEUBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv_high = conv1x1_block(
                in_channels=in_channels_high,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            self.conv_low = conv1x1_block(
                in_channels=in_channels_low,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            self.conv_w_high = conv1x1(
                in_channels=out_channels,
                out_channels=out_channels)
            self.conv_w_low = conv1x1(
                in_channels=1,
                out_channels=1)
            self.sigmoid = nn.Activation("sigmoid")
            self.relu = nn.Activation("relu")
            self.up = InterpolationBlock(scale_factor=2, out_size=out_size)

    def hybrid_forward(self, F, x_high, x_low):
        x_high = self.conv_high(x_high)
        x_low = self.conv_low(x_low)

        w_high = F.contrib.AdaptiveAvgPooling2D(x_high, output_size=1)
        w_high = self.conv_w_high(w_high)
        w_high = self.relu(w_high)
        w_high = self.sigmoid(w_high)

        w_low = x_low.mean(axis=1, keepdims=True)
        w_low = self.conv_w_low(w_low)
        w_low = self.sigmoid(w_low)

        x_high = self.up(x_high)

        x_high = F.broadcast_mul(x_high, w_low)
        x_low = F.broadcast_mul(x_low, w_high)

        out = x_high + x_low
        return out


class FPENet(HybridBlock):
    """
    FPENet model from 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    meu_channels : list of int
        Number of output channels for MEU blocks.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 layers,
                 channels,
                 init_block_channels,
                 meu_channels,
                 use_se,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 **kwargs):
        super(FPENet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        with self.name_scope():
            self.stem = conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            in_channels = init_block_channels

            self.encoder = MultiOutputSequential(return_last=False)
            for i, (layers_i, out_channels) in enumerate(zip(layers, channels)):
                stage = FPEStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    layers=layers_i,
                    use_se=use_se,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)
                stage.do_output = True
                self.encoder.add(stage)
                in_channels = out_channels

            self.meu1 = MEUBlock(
                in_channels_high=channels[-1],
                in_channels_low=channels[-2],
                out_channels=meu_channels[0],
                out_size=((in_size[0] // 4, in_size[1] // 4) if fixed_size else None),
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.meu2 = MEUBlock(
                in_channels_high=meu_channels[0],
                in_channels_low=channels[-3],
                out_channels=meu_channels[1],
                out_size=((in_size[0] // 2, in_size[1] // 2) if fixed_size else None),
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            in_channels = meu_channels[1]

            self.classifier = conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True)

            self.up = InterpolationBlock(
                scale_factor=2,
                out_size=(in_size if fixed_size else None))

    def hybrid_forward(self, F, x):
        x = self.stem(x)
        y = self.encoder(x)
        x = self.meu1(y[2], y[1])
        x = self.meu2(x, y[0])
        x = self.classifier(x)
        x = self.up(x)
        return x


def get_fpenet(model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create FPENet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    width = 16
    channels = [int(width * (2 ** i)) for i in range(3)]
    init_block_channels = width
    layers = [1, 3, 9]
    meu_channels = [64, 32]
    use_se = False

    net = FPENet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        meu_channels=meu_channels,
        use_se=use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

    return net


def fpenet_cityscapes(classes=19, **kwargs):
    """
    FPENet model for Cityscapes from 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_fpenet(classes=classes, model_name="fpenet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def _test():
    import mxnet as mx

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        fpenet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fpenet_cityscapes or weight_count == 115125)

        batch = 4
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
