"""
    IBPPose for COCO Keypoint, implemented in Gluon.
    Original paper: 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.
"""

__all__ = ['IbpPose', 'ibppose_coco']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import get_activation_layer, conv1x1_block, conv3x3_block, conv7x7_block, SEBlock, Hourglass,\
    InterpolationBlock


class IbpResBottleneck(HybridBlock):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_bias=False,
                 bottleneck_factor=2,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(IbpResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                activation=activation)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                use_bias=use_bias,
                activation=activation)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IbpResUnit(HybridBlock):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 use_bias=False,
                 bottleneck_factor=2,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(IbpResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = IbpResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                use_bias=use_bias,
                bottleneck_factor=bottleneck_factor,
                activation=activation)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    use_bias=use_bias,
                    activation=None)
            self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class IbpBackbone(HybridBlock):
    """
    IBPPose backbone.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation,
                 **kwargs):
        super(IbpBackbone, self).__init__(**kwargs)
        dilations = (3, 3, 4, 4, 5, 5)
        mid1_channels = out_channels // 4
        mid2_channels = out_channels // 2

        with self.name_scope():
            self.conv1 = conv7x7_block(
                in_channels=in_channels,
                out_channels=mid1_channels,
                strides=2,
                activation=activation)
            self.res1 = IbpResUnit(
                in_channels=mid1_channels,
                out_channels=mid2_channels,
                activation=activation)
            self.pool = nn.MaxPool2D(
                pool_size=2,
                strides=2)
            self.res2 = IbpResUnit(
                in_channels=mid2_channels,
                out_channels=mid2_channels,
                activation=activation)
            self.dilation_branch = nn.HybridSequential(prefix="")
            for dilation in dilations:
                self.dilation_branch.add(conv3x3_block(
                    in_channels=mid2_channels,
                    out_channels=mid2_channels,
                    padding=dilation,
                    dilation=dilation,
                    activation=activation))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        y = self.dilation_branch(x)
        x = F.concat(x, y, dim=1)
        return x


class IbpDownBlock(HybridBlock):
    """
    IBPPose down block for the hourglass.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation,
                 **kwargs):
        super(IbpDownBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.down = nn.MaxPool2D(
                pool_size=2,
                strides=2)
            self.res = IbpResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation)

    def hybrid_forward(self, F, x):
        x = self.down(x)
        x = self.res(x)
        return x


class IbpUpBlock(HybridBlock):
    """
    IBPPose up block for the hourglass.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn,
                 activation,
                 **kwargs):
        super(IbpUpBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.res = IbpResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation)
            self.up = InterpolationBlock(
                scale_factor=2,
                bilinear=False)
            self.conv = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=(not use_bn),
                use_bn=use_bn,
                activation=activation)

    def hybrid_forward(self, F, x):
        x = self.res(x)
        x = self.up(x)
        x = self.conv(x)
        return x


class MergeBlock(HybridBlock):
    """
    IBPPose merge block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn,
                 **kwargs):
        super(MergeBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=(not use_bn),
                use_bn=use_bn,
                activation=None)

    def hybrid_forward(self, F, x):
        return self.conv(x)


class IbpPreBlock(HybridBlock):
    """
    IBPPose preliminary decoder block.

    Parameters:
    ----------
    out_channels : int
        Number of output channels.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 out_channels,
                 use_bn,
                 activation,
                 **kwargs):
        super(IbpPreBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=(not use_bn),
                use_bn=use_bn,
                activation=activation)
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=(not use_bn),
                use_bn=use_bn,
                activation=activation)
            self.se = SEBlock(
                channels=out_channels,
                use_conv=False,
                mid_activation=activation)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x


class IbpPass(HybridBlock):
    """
    IBPPose single pass decoder block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    mid_channels : int
        Number of middle channels.
    depth : int
        Depth of hourglass.
    growth_rate : int
        Addition for number of channel for each level.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 mid_channels,
                 depth,
                 growth_rate,
                 merge,
                 use_bn,
                 activation,
                 **kwargs):
        super(IbpPass, self).__init__(**kwargs)
        self.merge = merge

        with self.name_scope():
            down_seq = nn.HybridSequential(prefix="")
            up_seq = nn.HybridSequential(prefix="")
            skip_seq = nn.HybridSequential(prefix="")
            top_channels = channels
            bottom_channels = channels
            for i in range(depth + 1):
                skip_seq.add(IbpResUnit(
                    in_channels=top_channels,
                    out_channels=top_channels,
                    activation=activation))
                bottom_channels += growth_rate
                if i < depth:
                    down_seq.add(IbpDownBlock(
                        in_channels=top_channels,
                        out_channels=bottom_channels,
                        activation=activation))
                    up_seq.add(IbpUpBlock(
                        in_channels=bottom_channels,
                        out_channels=top_channels,
                        use_bn=use_bn,
                        activation=activation))
                top_channels = bottom_channels
            self.hg = Hourglass(
                down_seq=down_seq,
                up_seq=up_seq,
                skip_seq=skip_seq)

            self.pre_block = IbpPreBlock(
                out_channels=channels,
                use_bn=use_bn,
                activation=activation)
            self.post_block = conv1x1_block(
                in_channels=channels,
                out_channels=mid_channels,
                use_bias=True,
                use_bn=False,
                activation=None)

            if self.merge:
                self.pre_merge_block = MergeBlock(
                    in_channels=channels,
                    out_channels=channels,
                    use_bn=use_bn)
                self.post_merge_block = MergeBlock(
                    in_channels=mid_channels,
                    out_channels=channels,
                    use_bn=use_bn)

    def hybrid_forward(self, F, x, x_prev):
        x = self.hg(x)
        if x_prev is not None:
            x = x + x_prev
        y = self.pre_block(x)
        z = self.post_block(y)
        if self.merge:
            z = self.post_merge_block(z) + self.pre_merge_block(y)
        return z


class IbpPose(HybridBlock):
    """
    IBPPose model from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.

    Parameters:
    ----------
    passes : int
        Number of passes.
    backbone_out_channels : int
        Number of output channels for the backbone.
    outs_channels : int
        Number of output channels for the backbone.
    depth : int
        Depth of hourglass.
    growth_rate : int
        Addition for number of channel for each level.
    use_bn : bool
        Whether to use BatchNorm layer.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 256)
        Spatial size of the expected input image.
    """
    def __init__(self,
                 passes,
                 backbone_out_channels,
                 outs_channels,
                 depth,
                 growth_rate,
                 use_bn,
                 in_channels=3,
                 in_size=(256, 256),
                 **kwargs):
        super(IbpPose, self).__init__(**kwargs)
        self.in_size = in_size
        activation = (lambda: nn.LeakyReLU(alpha=0.01))

        with self.name_scope():
            self.backbone = IbpBackbone(
                in_channels=in_channels,
                out_channels=backbone_out_channels,
                activation=activation)

            self.decoder = nn.HybridSequential(prefix="")
            for i in range(passes):
                merge = (i != passes - 1)
                self.decoder.add(IbpPass(
                    channels=backbone_out_channels,
                    mid_channels=outs_channels,
                    depth=depth,
                    growth_rate=growth_rate,
                    merge=merge,
                    use_bn=use_bn,
                    activation=activation))

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        x_prev = None
        for block in self.decoder._children.values():
            if x_prev is not None:
                x = x + x_prev
            x_prev = block(x, x_prev)
        return x_prev


def get_ibppose(model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create IBPPose model with specific parameters.

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
    passes = 4
    backbone_out_channels = 256
    outs_channels = 50
    depth = 4
    growth_rate = 128
    use_bn = True

    net = IbpPose(
        passes=passes,
        backbone_out_channels=backbone_out_channels,
        outs_channels=outs_channels,
        depth=depth,
        growth_rate=growth_rate,
        use_bn=use_bn,
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


def ibppose_coco(**kwargs):
    """
    IBPPose model for COCO Keypoint from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person
    Pose Estimation,' https://arxiv.org/abs/1911.10529.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_ibppose(model_name="ibppose_coco", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (256, 256)
    pretrained = False

    models = [
        ibppose_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibppose_coco or weight_count == 95827784)

        batch = 14
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, 50, in_size[0] // 4, in_size[0] // 4))


if __name__ == "__main__":
    _test()
