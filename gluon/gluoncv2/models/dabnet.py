"""
    DABNet for image segmentation, implemented in Gluon.
    Original paper: 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.
"""

__all__ = ['DABNet', 'dabnet_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv3x3, conv3x3_block, ConvBlock, NormActivation, Concurrent, InterpolationBlock,\
    DualPathSequential, PReLU2


class DwaConvBlock(HybridBlock):
    """
    Depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(DwaConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(kernel_size, 1),
                strides=strides,
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=activation)
            self.conv2 = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, kernel_size),
                strides=strides,
                padding=(0, padding),
                dilation=(1, dilation),
                groups=channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=activation)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def dwa_conv3x3_block(channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      use_bias=False,
                      use_bn=True,
                      bn_epsilon=1e-5,
                      bn_use_global_stats=False,
                      bn_cudnn_off=False,
                      activation=(lambda: nn.Activation("relu")),
                      **kwargs):
    """
    3x3 version of the depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    return DwaConvBlock(
        channels=channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        bn_cudnn_off=bn_cudnn_off,
        activation=activation,
        **kwargs)


class DABBlock(HybridBlock):
    """
    DABNet specific base block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for a dilated branch in the unit.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 dilation,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DABBlock, self).__init__(**kwargs)
        mid_channels = channels // 2

        with self.name_scope():
            self.norm_activ1 = NormActivation(
                in_channels=channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(channels)))
            self.conv1 = conv3x3_block(
                in_channels=channels,
                out_channels=mid_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(mid_channels)))

            self.branches = Concurrent(stack=True)
            self.branches.add(dwa_conv3x3_block(
                channels=mid_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(mid_channels))))
            self.branches.add(dwa_conv3x3_block(
                channels=mid_channels,
                padding=dilation,
                dilation=dilation,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(mid_channels))))

            self.norm_activ2 = NormActivation(
                in_channels=mid_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(mid_channels)))
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels)

    def hybrid_forward(self, F, x):
        identity = x

        x = self.norm_activ1(x)
        x = self.conv1(x)

        x = self.branches(x)
        x = x.sum(axis=1)

        x = self.norm_activ2(x)
        x = self.conv2(x)

        x = x + identity
        return x


class DownBlock(HybridBlock):
    """
    DABNet specific downsample block for the main branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DownBlock, self).__init__(**kwargs)
        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        with self.name_scope():
            self.conv = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2)
            if self.expand:
                self.pool = nn.MaxPool2D(
                    pool_size=2,
                    strides=2)
            self.norm_activ = NormActivation(
                in_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))

    def hybrid_forward(self, F, x):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = F.concat(y, z, dim=1)

        y = self.norm_activ(y)
        return y


class DABUnit(HybridBlock):
    """
    DABNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilations : list of int
        Dilations for blocks.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DABUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.down = DownBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.blocks = nn.HybridSequential(prefix="")
            for i, dilation in enumerate(dilations):
                self.blocks.add(DABBlock(
                    channels=mid_channels,
                    dilation=dilation,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))

    def hybrid_forward(self, F, x):
        x = self.down(x)
        y = self.blocks(x)
        x = F.concat(y, x, dim=1)
        return x


class DABStage(HybridBlock):
    """
    DABNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    dilations : list of int
        Dilations for blocks.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 dilations,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DABStage, self).__init__(**kwargs)
        self.use_unit = (len(dilations) > 0)

        with self.name_scope():
            self.x_down = nn.AvgPool2D(
                pool_size=3,
                strides=2,
                padding=1)

            if self.use_unit:
                self.unit = DABUnit(
                    in_channels=y_in_channels,
                    out_channels=(y_out_channels - x_channels),
                    dilations=dilations,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)

            self.norm_activ = NormActivation(
                in_channels=y_out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(y_out_channels)))

    def hybrid_forward(self, F, y, x):
        x = self.x_down(x)
        if self.use_unit:
            y = self.unit(y)
        y = F.concat(y, x, dim=1)
        y = self.norm_activ(y)
        return y, x


class DABInitBlock(HybridBlock):
    """
    DABNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DABInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))
            self.conv3 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DABNet(HybridBlock):
    """
    DABNet model from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of list of int
        Dilations for blocks.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 channels,
                 init_block_channels,
                 dilations,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 **kwargs):
        super(DABNet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=0)
            self.features.add(DABInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            y_in_channels = init_block_channels

            for i, (y_out_channels, dilations_i) in enumerate(zip(channels, dilations)):
                self.features.add(DABStage(
                    x_channels=in_channels,
                    y_in_channels=y_in_channels,
                    y_out_channels=y_out_channels,
                    dilations=dilations_i,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                y_in_channels = y_out_channels

            self.classifier = conv1x1(
                in_channels=y_in_channels,
                out_channels=classes)

            self.up = InterpolationBlock(scale_factor=8)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        y = self.features(x, x)
        y = self.classifier(y)
        y = self.up(y, in_size)
        return y


def get_dabnet(model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create DABNet model with specific parameters.

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
    init_block_channels = 32
    channels = [35, 131, 259]
    dilations = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]
    bn_epsilon = 1e-3

    net = DABNet(
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        bn_epsilon=bn_epsilon,
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


def dabnet_cityscapes(classes=19, **kwargs):
    """
    DABNet model for Cityscapes from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

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
    return get_dabnet(classes=classes, model_name="dabnet_cityscapes", **kwargs)


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
        dabnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dabnet_cityscapes or weight_count == 756643)

        batch = 4
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
