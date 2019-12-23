"""
    Oct-ResNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
    Convolution,' https://arxiv.org/abs/1904.05049.
"""

__all__ = ['OctResNet', 'octresnet10_ad2', 'octresnet50b_ad2', 'OctResUnit']

import os
from inspect import isfunction
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ReLU6, DualPathSequential
from .resnet import ResInitBlock


class OctConv(nn.Conv2D):
    """
    Octave convolution layer.

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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    oct_value : int, default 2
        Octave value.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding=1,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 oct_alpha=0.0,
                 oct_mode="std",
                 oct_value=2,
                 **kwargs):
        if isinstance(strides, int):
            strides = (strides, strides)
        self.downsample = (strides[0] > 1) or (strides[1] > 1)
        assert (strides[0] in [1, oct_value]) and (strides[1] in [1, oct_value])
        strides = (1, 1)
        if oct_mode == "first":
            in_alpha = 0.0
            out_alpha = oct_alpha
        elif oct_mode == "norm":
            in_alpha = oct_alpha
            out_alpha = oct_alpha
        elif oct_mode == "last":
            in_alpha = oct_alpha
            out_alpha = 0.0
        elif oct_mode == "std":
            in_alpha = 0.0
            out_alpha = 0.0
        else:
            raise ValueError("Unsupported octave convolution mode: {}".format(oct_mode))
        self.h_in_channels = int(in_channels * (1.0 - in_alpha))
        self.h_out_channels = int(out_channels * (1.0 - out_alpha))
        self.l_out_channels = out_channels - self.h_out_channels
        self.oct_alpha = oct_alpha
        self.oct_mode = oct_mode
        self.oct_value = oct_value
        super(OctConv, self).__init__(
            in_channels=in_channels,
            channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            **kwargs)
        self.conv_kwargs = self._kwargs.copy()
        del self.conv_kwargs["num_filter"]

    def hybrid_forward(self, F, hx, lx=None, weight=None, bias=None):
        if self.oct_mode == "std":
            return super(OctConv, self).hybrid_forward(F, hx, weight=weight, bias=bias), None

        if self.downsample:
            hx = F.Pooling(
                hx,
                kernel=(self.oct_value, self.oct_value),
                stride=(self.oct_value, self.oct_value),
                pool_type="avg")

        hhy = F.Convolution(
            hx,
            weight=weight.slice(begin=(None, None), end=(self.h_out_channels, self.h_in_channels)),
            bias=bias.slice(begin=(None,), end=(self.h_out_channels,)) if bias is not None else None,
            num_filter=self.h_out_channels,
            **self.conv_kwargs)

        if self.oct_mode != "first":
            hlx = F.Convolution(
                lx,
                weight=weight.slice(begin=(None, self.h_in_channels), end=(self.h_out_channels, None)),
                bias=bias.slice(begin=(None,), end=(self.h_out_channels,)) if bias is not None else None,
                num_filter=self.h_out_channels,
                **self.conv_kwargs)

        if self.oct_mode == "last":
            hy = hhy + hlx
            ly = None
            return hy, ly

        lhx = F.Pooling(
            hx,
            kernel=(self.oct_value, self.oct_value),
            stride=(self.oct_value, self.oct_value),
            pool_type="avg")
        lhy = F.Convolution(
            lhx,
            weight=weight.slice(begin=(self.h_out_channels, None), end=(None, self.h_in_channels)),
            bias=bias.slice(begin=(self.h_out_channels,), end=(None,)) if bias is not None else None,
            num_filter=self.l_out_channels,
            **self.conv_kwargs)

        if self.oct_mode == "first":
            hy = hhy
            ly = lhy
            return hy, ly

        if self.downsample:
            hly = hlx
            llx = F.Pooling(
                lx,
                kernel=(self.oct_value, self.oct_value),
                stride=(self.oct_value, self.oct_value),
                pool_type="avg")
        else:
            hly = F.UpSampling(hlx, scale=self.oct_value, sample_type="nearest")
            llx = lx
        lly = F.Convolution(
            llx,
            weight=weight.slice(begin=(self.h_out_channels, self.h_in_channels), end=(None, None)),
            bias=bias.slice(begin=(self.h_out_channels,), end=(None,)) if bias is not None else None,
            num_filter=self.l_out_channels,
            **self.conv_kwargs)

        hy = hhy + hly
        ly = lhy + lly
        return hy, ly

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ', oct_alpha={}'.format(self.oct_alpha)
        s += ', oct_mode={}'.format(self.oct_mode)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)


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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation("relu")
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
                 oct_mode="std",
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 activate=True,
                 **kwargs):
        super(OctConvBlock, self).__init__(**kwargs)
        self.activate = activate
        self.last = (oct_mode == "last") or (oct_mode == "std")
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
                      oct_mode="std",
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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation("relu")
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
                      oct_mode="std",
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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation("relu")
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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 oct_alpha=0.0,
                 oct_mode="std",
                 bn_use_global_stats=False,
                 **kwargs):
        super(OctResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = oct_conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                oct_alpha=oct_alpha,
                oct_mode=oct_mode,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = oct_conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                oct_alpha=oct_alpha,
                oct_mode=("std" if oct_mode == "last" else (oct_mode if oct_mode != "first" else "norm")),
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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
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
                 oct_mode="std",
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
                oct_mode=(oct_mode if oct_mode != "last" else "norm"),
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = oct_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides),
                padding=padding,
                dilation=dilation,
                oct_alpha=oct_alpha,
                oct_mode=(oct_mode if oct_mode != "first" else "norm"),
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = oct_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                oct_alpha=oct_alpha,
                oct_mode=("std" if oct_mode == "last" else (oct_mode if oct_mode != "first" else "norm")),
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
    oct_mode : str, default 'std'
        Octave convolution mode. It can be 'first', 'norm', 'last', or 'std'.
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
                 oct_mode="std",
                 bn_use_global_stats=False,
                 bottleneck=True,
                 conv1_stride=False,
                 **kwargs):
        super(OctResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1) or\
                               ((oct_mode == "first") and (oct_alpha != 0.0))

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
                last_ordinals=1,
                prefix="")
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        if (i == 0) and (j == 0):
                            oct_mode = "first"
                        elif (i == len(channels) - 1) and (j == 0):
                            oct_mode = "last"
                        elif (i == len(channels) - 1) and (j != 0):
                            oct_mode = "std"
                        else:
                            oct_mode = "norm"
                        stage.add(OctResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            oct_alpha=oct_alpha,
                            oct_mode=oct_mode,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
                        in_channels = out_channels
                self.features.add(stage)
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


def get_octresnet(blocks,
                  bottleneck=None,
                  conv1_stride=True,
                  oct_alpha=0.5,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
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
    oct_alpha : float, default 0.5
        Octave alpha coefficient.
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
        oct_alpha=oct_alpha,
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


def octresnet10_ad2(**kwargs):
    """
    Oct-ResNet-10 (alpha=1/2) model from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks
    with Octave Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_octresnet(blocks=10, oct_alpha=0.5, model_name="octresnet10_ad2", **kwargs)


def octresnet50b_ad2(**kwargs):
    """
    Oct-ResNet-50b (alpha=1/2) model from 'Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks
    with Octave Convolution,' https://arxiv.org/abs/1904.05049.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_octresnet(blocks=50, conv1_stride=False, oct_alpha=0.5, model_name="octresnet50b_ad2", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        octresnet10_ad2,
        octresnet50b_ad2,
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
        assert (model != octresnet10_ad2 or weight_count == 5423016)
        assert (model != octresnet50b_ad2 or weight_count == 25557032)

        x = mx.nd.zeros((14, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
