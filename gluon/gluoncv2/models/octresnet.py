"""
    Oct-ResNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
    Convolution,' https://arxiv.org/abs/1904.05049.
"""

__all__ = ['OctResNet', 'octresnet50b']

import os
from inspect import isfunction
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ReLU6, DualPathSequential
from .resnet import ResInitBlock, ResUnit


class UpSamplingBlock(HybridBlock):
    """
    UpSampling block (neighbour nearest).

    Parameters:
    ----------
    scale : int
        Multiplier for spatial size.
    sample_type : str, default 'nearest'
        Type of interpolation.
    """
    def __init__(self,
                 scale,
                 sample_type="nearest",
                 **kwargs):
        super(UpSamplingBlock, self).__init__(**kwargs)
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


class OctConv(HybridBlock):
    """
    Octave convolution layer (actually block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 oct_alpha=0.0,
                 oct_mode="last",
                 **kwargs):
        super(OctConv, self).__init__(**kwargs)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.downsample = (strides[0] > 1) or (strides[1] > 1)

        self.first = False
        self.last = False
        if oct_mode == "first":
            self.first = True
            in_alpha = 0.0
            out_alpha = oct_alpha
        elif oct_mode == "norm":
            in_alpha = oct_alpha
            out_alpha = oct_alpha
        elif oct_mode == "last":
            self.last = True
            in_alpha = oct_alpha
            out_alpha = 0.0
        else:
            raise ValueError("Unsupported octave convolution mode: {}".format(oct_mode))

        h_in_channels = int(in_channels * (1.0 - in_alpha))
        h_out_channels = int(out_channels * (1.0 - out_alpha))
        l_in_channels = in_channels - h_in_channels
        l_out_channels = out_channels - h_out_channels

        with self.name_scope():
            if self.downsample:
                self.down_pool = nn.AvgPool2D(
                    pool_size=strides,
                    strides=strides)
            self.hh_conv = nn.Conv2D(
                channels=h_out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=h_in_channels)
            if not self.last:
                self.lh_pool = nn.AvgPool2D(
                    pool_size=2,
                    strides=2)
                self.lh_conv = nn.Conv2D(
                    channels=l_out_channels,
                    kernel_size=kernel_size,
                    strides=1,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    use_bias=use_bias,
                    in_channels=h_in_channels)
            if not self.first:
                self.hl_conv = nn.Conv2D(
                    channels=h_out_channels,
                    kernel_size=kernel_size,
                    strides=1,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    use_bias=use_bias,
                    in_channels=l_in_channels)
                if not self.last:
                    if self.downsample:
                        self.ll_pool = nn.AvgPool2D(
                            pool_size=2,
                            strides=2)
                    else:
                        self.hl_upsample = UpSamplingBlock(scale=2)
                    self.ll_conv = nn.Conv2D(
                        channels=l_out_channels,
                        kernel_size=kernel_size,
                        strides=1,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        use_bias=use_bias,
                        in_channels=l_in_channels)

    def hybrid_forward(self, F, hx, lx=None):
        if self.downsample:
            hx = self.down_pool(hx)

        hhy = self.hh_conv(hx)

        if not self.first:
            hlx = self.hl_conv(lx)

        if self.last:
            hy = hhy + hlx
            ly = None
            return hy, ly

        lhx = self.lh_pool(hx)
        lhy = self.lh_conv(lhx)

        if self.first:
            hy = hhy
            ly = lhy
            return hy, ly

        if self.downsample:
            hly = hlx
            llx = self.ll_pool(lx)
        else:
            hly = self.hl_upsample(hlx)
            llx = lx
        lly = self.ll_conv(llx)

        hy = hhy + hly
        ly = lhy + lly

        return hy, ly


class OctConvBlock(HybridBlock):
    """
    Octave convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 oct_alpha=0.0,
                 oct_mode="last",
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 activate=True,
                 **kwargs):
        super(OctConvBlock, self).__init__(**kwargs)
        self.activate = activate
        self.last = (oct_mode == "last")
        out_alpha = 0.0 if self.last else oct_alpha
        h_out_channels = int(out_channels * (1.0 - out_alpha))
        l_out_channels = out_channels - h_out_channels

        with self.name_scope():
            self.conv = OctConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                oct_alpha=oct_alpha,
                oct_mode=oct_mode)
            self.h_bn = nn.BatchNorm(
                in_channels=h_out_channels,
                epsilon=bn_epsilon,
                use_global_stats=bn_use_global_stats)
            if not self.last:
                self.l_bn = nn.BatchNorm(
                    in_channels=l_out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats)
            if self.activate:
                assert (activation is not None)
                if isfunction(activation):
                    self.activ = activation()
                elif isinstance(activation, str):
                    if activation == "relu6":
                        self.activ = ReLU6()
                    else:
                        self.activ = nn.Activation(activation)
                else:
                    self.activ = activation

    def hybrid_forward(self, F, hx, lx=None):
        hx, lx = self.conv(hx, lx)
        hx = self.h_bn(hx)
        if self.activate:
            hx = self.activ(hx)
        if not self.last:
            lx = self.l_bn(lx)
            if self.activate:
                lx = self.activ(lx)
        return hx, lx


def oct_conv1x1_block(in_channels,
                      out_channels,
                      strides=1,
                      groups=1,
                      use_bias=False,
                      oct_alpha=0.0,
                      oct_mode="last",
                      bn_epsilon=1e-5,
                      bn_use_global_stats=False,
                      activation=(lambda: nn.Activation("relu")),
                      activate=True,
                      **kwargs):
    """
    1x1 version of the octave convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return OctConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        oct_alpha=oct_alpha,
        oct_mode=oct_mode,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        activate=activate,
        **kwargs)


def oct_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      use_bias=False,
                      oct_alpha=0.0,
                      oct_mode="last",
                      bn_epsilon=1e-5,
                      bn_use_global_stats=False,
                      activation=(lambda: nn.Activation("relu")),
                      activate=True,
                      **kwargs):
    """
    3x3 version of the octave convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return OctConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        oct_alpha=oct_alpha,
        oct_mode=oct_mode,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        activate=activate,
        **kwargs)


class OctResBlock(HybridBlock):
    """
    Simple Oct-ResNet block for residual path in Oct-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 oct_alpha=0.0,
                 oct_mode="last",
                 bn_use_global_stats=False,
                 **kwargs):
        super(OctResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = oct_conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                oct_alpha=oct_alpha,
                oct_mode=("first" if oct_mode == "first" else "norm"),
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = oct_conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                oct_alpha=oct_alpha,
                oct_mode=("last" if oct_mode == "last" else "norm"),
                bn_use_global_stats=bn_use_global_stats,
                activation=None,
                activate=False)

    def hybrid_forward(self, F, hx, lx=None):
        hx, lx = self.conv1(hx, lx)
        hx, lx = self.conv2(hx, lx)
        return hx, lx


class OctResBottleneck(HybridBlock):
    """
    Oct-ResNet bottleneck block for residual path in Oct-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 oct_alpha=0.0,
                 oct_mode="last",
                 bn_use_global_stats=False,
                 conv1_stride=False,
                 bottleneck_factor=4,
                 **kwargs):
        super(OctResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = oct_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=(strides if conv1_stride else 1),
                oct_alpha=oct_alpha,
                oct_mode=("first" if oct_mode == "first" else "norm"),
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = oct_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides),
                padding=padding,
                dilation=dilation,
                oct_alpha=oct_alpha,
                oct_mode="norm",
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = oct_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                oct_alpha=oct_alpha,
                oct_mode=("last" if oct_mode == "last" else "norm"),
                bn_use_global_stats=bn_use_global_stats,
                activation=None,
                activate=False)

    def hybrid_forward(self, F, hx, lx=None):
        hx, lx = self.conv1(hx, lx)
        hx, lx = self.conv2(hx, lx)
        hx, lx = self.conv3(hx, lx)
        return hx, lx


class OctResUnit(HybridBlock):
    """
    Oct-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    oct_alpha : float, default 0.0
        Octave alpha coefficient.
    oct_mode : str, default 'last'
        Octave convolution mode. It can be 'first', 'norm', or 'last'.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 oct_alpha=0.0,
                 oct_mode="last",
                 bn_use_global_stats=False,
                 bottleneck=True,
                 conv1_stride=False,
                 **kwargs):
        super(OctResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = OctResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    oct_alpha=oct_alpha,
                    oct_mode=oct_mode,
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=conv1_stride)
            else:
                self.body = OctResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    oct_alpha=oct_alpha,
                    oct_mode=oct_mode,
                    bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = oct_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    oct_alpha=oct_alpha,
                    oct_mode=oct_mode,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None,
                    activate=False)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, hx, lx=None):
        if self.resize_identity:
            h_identity, l_identity = self.identity_conv(hx, lx)
        else:
            h_identity, l_identity = hx, lx
        hx, lx = self.body(hx, lx)
        hx = hx + h_identity
        hx = self.activ(hx)
        if lx is not None:
            lx = lx + l_identity
            lx = self.activ(lx)
        return hx, lx


class OctResNet(HybridBlock):
    """
    Oct-ResNet model from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
    Convolution,' https://arxiv.org/abs/1904.05049.

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
    oct_alpha : float, default 0.5
        Octave alpha coefficient.
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
                 bottleneck,
                 conv1_stride,
                 oct_alpha=0.5,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(OctResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=2,
                prefix='')
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                if i != len(channels) - 1:
                    stage = DualPathSequential(prefix="stage{}_".format(i + 1))
                else:
                    stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        if (i == 0) and (j == 0):
                            oct_mode = "first"
                        elif (i == len(channels) - 2) and (j == len(channels_per_stage) - 1):
                            oct_mode = "last"
                        else:
                            oct_mode = "norm"
                        if i != len(channels) - 1:
                            stage.add(OctResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                oct_alpha=oct_alpha,
                                oct_mode=oct_mode,
                                bn_use_global_stats=bn_use_global_stats,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride))
                        else:
                            stage.add(ResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                bn_use_global_stats=bn_use_global_stats,
                                bottleneck=bottleneck,
                                conv1_stride=conv1_stride))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_octresnet(blocks,
                  bottleneck=None,
                  conv1_stride=True,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join('~', '.mxnet', 'models'),
                  **kwargs):
    """
    Create Oct-ResNet model with specific parameters.

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        raise ValueError("Unsupported Oct-ResNet with number of blocks: {}".format(blocks))

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

    net = OctResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def octresnet50b(**kwargs):
    """
    Oct-ResNet-50 model with stride at the second convolution in bottleneck block from 'Drop an Octave: Reducing Spatial
    Redundancy in Convolutional Neural Networks with Octave Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_octresnet(blocks=50, conv1_stride=False, model_name="octresnet50b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        octresnet50b,
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
        # assert (model != octresnet50b or weight_count == 25557032)

        x = mx.nd.zeros((14, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
