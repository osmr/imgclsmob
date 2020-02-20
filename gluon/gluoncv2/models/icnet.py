"""
    ICNet for image segmentation, implemented in Gluon.
    Original paper: 'ICNet for Real-Time Semantic Segmentation on High-Resolution Images,'
    https://arxiv.org/abs/1704.08545.
"""

__all__ = ['ICNet', 'icnet_resnetd50b_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv1x1_block, conv3x3_block, InterpolationBlock, MultiOutputSequential
from .pspnet import PyramidPooling
from .resnetd import resnetd50b


class ICInitBlock(HybridBlock):
    """
    ICNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(ICInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv3 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PSPBlock(HybridBlock):
    """
    ICNet specific PSPNet reduced head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    upscale_out_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    bottleneck_factor : int
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 upscale_out_size,
                 bottleneck_factor,
                 **kwargs):
        super(PSPBlock, self).__init__(**kwargs)
        assert (in_channels % bottleneck_factor == 0)
        mid_channels = in_channels // bottleneck_factor

        with self.name_scope():
            self.pool = PyramidPooling(
                in_channels=in_channels,
                upscale_out_size=upscale_out_size)
            self.conv = conv3x3_block(
                in_channels=4096,
                out_channels=mid_channels)
            self.dropout = nn.Dropout(rate=0.1)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class CFFBlock(HybridBlock):
    """
    Cascade Feature Fusion block.

    Parameters:
    ----------
    in_channels_low : int
        Number of input channels (low input).
    in_channels_high : int
        Number of input channels (low high).
    out_channels : int
        Number of output channels.
    out_size : tuple of two ints
        Spatial size of the expected output image.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels_low,
                 in_channels_high,
                 out_channels,
                 out_size,
                 classes,
                 **kwargs):
        super(CFFBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.up = InterpolationBlock(
                scale_factor=2,
                out_size=out_size)
            self.conv_low = conv3x3_block(
                in_channels=in_channels_low,
                out_channels=out_channels,
                padding=2,
                dilation=2,
                activation=None)
            self.conv_hign = conv1x1_block(
                in_channels=in_channels_high,
                out_channels=out_channels,
                activation=None)
            self.activ = nn.Activation("relu")
            self.conv_cls = conv1x1(
                in_channels=out_channels,
                out_channels=classes)

    def hybrid_forward(self, F, xl, xh):
        xl = self.up(xl)
        xl = self.conv_low(xl)
        xh = self.conv_hign(xh)
        x = xl + xh
        x = self.activ(x)
        x_cls = self.conv_cls(xl)
        return x, x_cls


class ICHeadBlock(HybridBlock):
    """
    ICNet head block.

    Parameters:
    ----------
    in_size : tuple of two ints
        Spatial size of the expected output image.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_size,
                 classes,
                 **kwargs):
        super(ICHeadBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.cff_12 = CFFBlock(
                in_channels_low=128,
                in_channels_high=64,
                out_channels=128,
                classes=classes,
                out_size=(in_size[0] // 8, in_size[1] // 8) if in_size is not None else None)
            self.cff_24 = CFFBlock(
                in_channels_low=256,
                in_channels_high=256,
                out_channels=128,
                classes=classes,
                out_size=(in_size[0] // 16, in_size[1] // 16) if in_size is not None else None)
            self.up_x2 = InterpolationBlock(
                scale_factor=2,
                out_size=(in_size[0] // 4, in_size[1] // 4) if in_size is not None else None)
            self.up_x8 = InterpolationBlock(
                scale_factor=4,
                out_size=in_size)
            self.conv_cls = conv1x1(
                in_channels=128,
                out_channels=classes)

    def hybrid_forward(self, F, x1, x2, x4):
        outputs = []

        x_cff_24, x_24_cls = self.cff_24(x4, x2)
        outputs.append(x_24_cls)

        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x1)
        outputs.append(x_12_cls)

        up_x2 = self.up_x2(x_cff_12)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x8 = self.up_x8(up_x2)
        outputs.append(up_x8)

        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return tuple(outputs)


class ICNet(HybridBlock):
    """
    ICNet model from 'ICNet for Real-Time Semantic Segmentation on High-Resolution Images,'
    https://arxiv.org/abs/1704.08545.

    Parameters:
    ----------
    backbones : tuple of nn.Sequential
        Feature extractors.
    backbones_out_channels : tuple of int
        Number of output channels form each feature extractor.
    channels : tuple of int
        Number of output channels for each branch.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    classes : int, default 21
        Number of segmentation classes.
    """
    def __init__(self,
                 backbones,
                 backbones_out_channels,
                 channels,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=21,
                 **kwargs):
        super(ICNet, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size
        psp_pool_out_size = (self.in_size[0] // 32, self.in_size[1] // 32) if fixed_size else None
        psp_head_out_channels = 512

        with self.name_scope():
            self.branch1 = ICInitBlock(
                in_channels=in_channels,
                out_channels=channels[0])

            self.branch2 = MultiOutputSequential(prefix="")
            self.branch2.add(InterpolationBlock(
                scale_factor=2,
                out_size=(in_size[0] // 2, in_size[1] // 2) if fixed_size else None,
                up=False))
            backbones[0].do_output = True
            self.branch2.add(backbones[0])

            self.branch2.add(InterpolationBlock(
                scale_factor=2,
                out_size=(in_size[0] // 32, in_size[1] // 32) if fixed_size else None,
                up=False))
            self.branch2.add(backbones[1])
            self.branch2.add(PSPBlock(
                in_channels=backbones_out_channels[1],
                upscale_out_size=psp_pool_out_size,
                bottleneck_factor=4))
            self.branch2.add(conv1x1_block(
                in_channels=psp_head_out_channels,
                out_channels=channels[2]))

            self.conv_y2 = conv1x1_block(
                in_channels=backbones_out_channels[0],
                out_channels=channels[1])

            self.final_block = ICHeadBlock(
                in_size=in_size if fixed_size else None,
                classes=classes)

    def hybrid_forward(self, F, x):
        y1 = self.branch1(x)
        y3, y2 = self.branch2(x)
        y2 = self.conv_y2(y2)
        x = self.final_block(y1, y2, y3)
        if self.aux:
            return x
        else:
            return x[0]


def get_icnet(backbones,
              backbones_out_channels,
              classes,
              aux=False,
              model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create ICNet model with specific parameters.

    Parameters:
    ----------
    backbones : tuple of nn.Sequential
        Feature extractors.
    backbones_out_channels : tuple of int
        Number of output channels form each feature extractor.
    classes : int
        Number of segmentation classes.
    aux : bool, default False
        Whether to output an auxiliary result.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    channels = (64, 256, 256)

    backbones[0].multi_output = False
    backbones[1].multi_output = False

    net = ICNet(
        backbones=backbones,
        backbones_out_channels=backbones_out_channels,
        channels=channels,
        classes=classes,
        aux=aux,
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


def icnet_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    ICNet model on the base of ResNet(D)-50b for Cityscapes from 'ICNet for Real-Time Semantic Segmentation on
    High-Resolution Images,' https://arxiv.org/abs/1704.08545.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=None).features[:-1]
    backbones = (backbone[:3], backbone[3:])
    backbones_out_channels = (512, 2048)
    return get_icnet(backbones=backbones, backbones_out_channels=backbones_out_channels, classes=classes, aux=aux,
                     model_name="icnet_resnetd50b_cityscapes", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (1024, 2048)
    aux = False
    pretrained = False
    fixed_size = False

    models = [
        (icnet_resnetd50b_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size, aux=aux)

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
        assert (model != icnet_resnetd50b_cityscapes or weight_count == 47489184)

        x = mx.nd.zeros((1, 3, in_size[0], in_size[1]), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
