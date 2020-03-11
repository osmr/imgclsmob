"""
    IBPPose for COCO Keypoint, implemented in Chainer.
    Original paper: 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.
"""

__all__ = ['IbpPose', 'ibppose_coco']

import os
import chainer.functions as F
from functools import partial
from chainer import Chain
from chainer.serializers import load_npz
from .common import get_activation_layer, conv1x1_block, conv3x3_block, conv7x7_block, SEBlock, Hourglass,\
    SimpleSequential, InterpolationBlock


class IbpResBottleneck(Chain):
    """
    Bottleneck block for residual path in the residual unit.

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
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_bias=False,
                 bottleneck_factor=2,
                 activation=(lambda: F.relu)):
        super(IbpResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                activation=activation)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                use_bias=use_bias,
                activation=activation)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IbpResUnit(Chain):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_bias=False,
                 bottleneck_factor=2,
                 activation=(lambda: F.relu)):
        super(IbpResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = IbpResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=use_bias,
                bottleneck_factor=bottleneck_factor,
                activation=activation)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_bias=use_bias,
                    activation=None)
            self.activ = get_activation_layer(activation)

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class IbpBackbone(Chain):
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
                 activation):
        super(IbpBackbone, self).__init__()
        dilations = (3, 3, 4, 4, 5, 5)
        mid1_channels = out_channels // 4
        mid2_channels = out_channels // 2

        with self.init_scope():
            self.conv1 = conv7x7_block(
                in_channels=in_channels,
                out_channels=mid1_channels,
                stride=2,
                activation=activation)
            self.res1 = IbpResUnit(
                in_channels=mid1_channels,
                out_channels=mid2_channels,
                activation=activation)
            self.pool = partial(
                F.max_pooling_2d,
                ksize=2,
                stride=2)
            self.res2 = IbpResUnit(
                in_channels=mid2_channels,
                out_channels=mid2_channels,
                activation=activation)
            self.dilation_branch = SimpleSequential()
            with self.dilation_branch.init_scope():
                for i, dilation in enumerate(dilations):
                    setattr(self.dilation_branch, "block{}".format(i + 1), conv3x3_block(
                        in_channels=mid2_channels,
                        out_channels=mid2_channels,
                        pad=dilation,
                        dilate=dilation,
                        activation=activation))

    def __call__(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        y = self.dilation_branch(x)
        x = F.concat((x, y), axis=1)
        return x


class IbpDownBlock(Chain):
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
                 activation):
        super(IbpDownBlock, self).__init__()
        with self.init_scope():
            self.down = partial(
                F.max_pooling_2d,
                ksize=2,
                stride=2)
            self.res = IbpResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation)

    def __call__(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class IbpUpBlock(Chain):
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
                 activation):
        super(IbpUpBlock, self).__init__()
        with self.init_scope():
            self.res = IbpResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation)
            self.up = InterpolationBlock(
                scale_factor=2,
                mode="nearest")
            self.conv = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=(not use_bn),
                use_bn=use_bn,
                activation=activation)

    def __call__(self, x):
        x = self.res(x)
        x = self.up(x)
        x = self.conv(x)
        return x


class MergeBlock(Chain):
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
                 use_bn):
        super(MergeBlock, self).__init__()
        with self.init_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=(not use_bn),
                use_bn=use_bn,
                activation=None)

    def __call__(self, x):
        return self.conv(x)


class IbpPreBlock(Chain):
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
                 activation):
        super(IbpPreBlock, self).__init__()
        with self.init_scope():
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

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x


class IbpPass(Chain):
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
                 activation):
        super(IbpPass, self).__init__()
        self.merge = merge

        with self.init_scope():
            down_seq = SimpleSequential()
            up_seq = SimpleSequential()
            skip_seq = SimpleSequential()
            top_channels = channels
            bottom_channels = channels
            for i in range(depth + 1):
                with skip_seq.init_scope():
                    setattr(skip_seq, "skip{}".format(i + 1), IbpResUnit(
                        in_channels=top_channels,
                        out_channels=top_channels,
                        activation=activation))
                bottom_channels += growth_rate
                if i < depth:
                    with down_seq.init_scope():
                        setattr(down_seq, "down{}".format(i + 1), IbpDownBlock(
                            in_channels=top_channels,
                            out_channels=bottom_channels,
                            activation=activation))
                    with up_seq.init_scope():
                        setattr(up_seq, "up{}".format(i + 1), IbpUpBlock(
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

    def __call__(self, x, x_prev):
        x = self.hg(x)
        if x_prev is not None:
            x = x + x_prev
        y = self.pre_block(x)
        z = self.post_block(y)
        if self.merge:
            z = self.post_merge_block(z) + self.pre_merge_block(y)
        return z


class IbpPose(Chain):
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
                 in_size=(256, 256)):
        super(IbpPose, self).__init__()
        self.in_size = in_size
        activation = partial(F.leaky_relu, slope=0.01)

        with self.init_scope():
            self.backbone = IbpBackbone(
                in_channels=in_channels,
                out_channels=backbone_out_channels,
                activation=activation)

            self.decoder = SimpleSequential()
            with self.decoder.init_scope():
                for i in range(passes):
                    merge = (i != passes - 1)
                    setattr(self.decoder, "pass{}".format(i + 1), IbpPass(
                        channels=backbone_out_channels,
                        mid_channels=outs_channels,
                        depth=depth,
                        growth_rate=growth_rate,
                        merge=merge,
                        use_bn=use_bn,
                        activation=activation))

    def __call__(self, x):
        x = self.backbone(x)
        x_prev = None
        for block_name in self.decoder.layer_names:
            block = self.decoder[block_name]
            if x_prev is not None:
                x = x + x_prev
            x_prev = block(x, x_prev)
        return x_prev


def get_ibppose(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".chainer", "models"),
                **kwargs):
    """
    Create IBPPose model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def ibppose_coco(**kwargs):
    """
    IBPPose model for COCO Keypoint from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person
    Pose Estimation,' https://arxiv.org/abs/1911.10529.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_ibppose(model_name="ibppose_coco", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (256, 256)
    pretrained = False

    models = [
        ibppose_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibppose_coco or weight_count == 95827784)

        batch = 14
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == 50))
        assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))


if __name__ == "__main__":
    _test()
