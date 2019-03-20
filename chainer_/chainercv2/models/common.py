"""
    Common routines for models in Chainer.
"""

__all__ = ['ReLU6', 'conv1x1', 'conv3x3', 'depthwise_conv3x3', 'ConvBlock', 'conv1x1_block', 'conv3x3_block',
           'conv7x7_block', 'dwconv3x3_block', 'PreConvBlock', 'pre_conv1x1_block', 'pre_conv3x3_block',
           'ChannelShuffle', 'ChannelShuffle2', 'SEBlock', 'SimpleSequential', 'DualPathSequential', 'Concurrent',
           'ParametricSequential', 'ParametricConcurrent', 'Hourglass', 'SesquialteralHourglass',
           'MultiOutputSequential']

from inspect import isfunction
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class ReLU6(Chain):
    """
    ReLU6 activation layer.
    """
    def __init__(self):
        super(ReLU6, self).__init__()

    def __call__(self, x):
        return F.clip(x, 0.0, 6.0)


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            use_bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return L.Convolution2D(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        nobias=(not use_bias),
        groups=groups)


def conv3x3(in_channels,
            out_channels,
            stride=1,
            pad=1,
            dilate=1,
            groups=1,
            use_bias=False):
    """
    Convolution 3x3 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return L.Convolution2D(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        nobias=(not use_bias),
        dilate=dilate,
        groups=groups)


def depthwise_conv3x3(channels,
                      stride):
    """
    Depthwise convolution 3x3 layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    return L.Convolution2D(
        in_channels=channels,
        out_channels=channels,
        ksize=3,
        stride=stride,
        pad=1,
        nobias=True,
        groups=channels)


class ConvBlock(Chain):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 activation=(lambda: F.relu),
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                groups=groups)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            if self.activate:
                assert (activation is not None)
                if isfunction(activation):
                    self.activ = activation()
                elif isinstance(activation, str):
                    if activation == "relu":
                        self.activ = F.relu
                    elif activation == "relu6":
                        self.activ = ReLU6()
                    else:
                        raise NotImplementedError()
                else:
                    self.activ = activation

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  groups=1,
                  use_bias=False,
                  activation=(lambda: F.relu),
                  activate=True):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        activate=activate)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  pad=1,
                  dilate=1,
                  groups=1,
                  use_bias=False,
                  activation=(lambda: F.relu),
                  activate=True):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=groups,
        use_bias=use_bias,
        activation=activation,
        activate=activate)


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  pad=3,
                  use_bias=False,
                  activation=(lambda: F.relu),
                  activate=True):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=7,
        stride=stride,
        pad=pad,
        use_bias=use_bias,
        activation=activation,
        activate=activate)


def dwconv3x3_block(in_channels,
                    out_channels,
                    stride,
                    pad=1,
                    dilate=1,
                    use_bias=False,
                    activation=(lambda: F.relu),
                    activate=True):
    """
    3x3 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return conv3x3_block(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=out_channels,
        use_bias=use_bias,
        activation=activation,
        activate=activate)


class PreConvBlock(Chain):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 return_preact=False,
                 activate=True):
        super(PreConvBlock, self).__init__()
        self.return_preact = return_preact
        self.activate = activate

        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            if self.activate:
                self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate)

    def __call__(self, x):
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv1x1_block(in_channels,
                      out_channels,
                      stride=1,
                      use_bias=False,
                      return_preact=False,
                      activate=True):
    """
    1x1 version of the pre-activated convolution block.

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
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        use_bias=use_bias,
        return_preact=return_preact,
        activate=activate)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      stride=1,
                      pad=1,
                      dilate=1,
                      return_preact=False,
                      activate=True):
    """
    3x3 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        return_preact=return_preact,
        activate=activate)


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input variable.
    groups : int
        Number of groups.

    Returns
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted variable.
    """
    batch, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, groups, channels_per_group, height, width))
    x = F.swapaxes(x, axis1=1, axis2=2)
    x = F.reshape(x, shape=(batch, channels, height, width))
    return x


class ChannelShuffle(Chain):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.

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
        super(ChannelShuffle, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def __call__(self, x):
        return channel_shuffle(x, self.groups)


def channel_shuffle2(x,
                     groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.

    Parameters:
    ----------
    x : chainer.Variable or numpy.ndarray or cupy.ndarray
        Input variable.
    groups : int
        Number of groups.

    Returns
    -------
    chainer.Variable or numpy.ndarray or cupy.ndarray
        Resulted variable.
    """
    batch, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, channels_per_group, groups, height, width))
    x = F.swapaxes(x, axis1=1, axis2=2)
    x = F.reshape(x, shape=(batch, channels, height, width))
    return x


class ChannelShuffle2(Chain):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

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
        super(ChannelShuffle2, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def __call__(self, x):
        return channel_shuffle2(x, self.groups)


class SEBlock(Chain):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    """
    def __init__(self,
                 channels,
                 reduction=16):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        with self.init_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_cannels,
                use_bias=True)
            self.conv2 = conv1x1(
                in_channels=mid_cannels,
                out_channels=channels,
                use_bias=True)

    def __call__(self, x):
        w = F.average_pooling_2d(x, ksize=x.shape[2:])
        w = self.conv1(w)
        w = F.relu(w)
        w = self.conv2(w)
        w = F.sigmoid(w)
        x = x * w
        return x


class SimpleSequential(Chain):
    """
    A sequential chain that can be used instead of Sequential.
    """
    def __init__(self):
        super(SimpleSequential, self).__init__()
        self.layer_names = []

    def __setattr__(self, name, value):
        super(SimpleSequential, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self.layer_names.append(name)

    def __delattr__(self, name):
        super(SimpleSequential, self).__delattr__(name)
        try:
            self.layer_names.remove(name)
        except ValueError:
            pass

    def __len__(self):
        return len(self.layer_names)

    def __call__(self, x):
        for name in self.layer_names:
            x = self[name](x)
        return x


class DualPathSequential(SimpleSequential):
    """
    A sequential container for blocks with dual inputs/outputs.
    Blocks will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first blocks with single input/output.
    last_ordinals : int, default 0
        Number of the final blocks with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a block.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal block.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda block, x1, x2: block(x1, x2)),
                 dual_path_scheme_ordinal=(lambda block, x1, x2: (block(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def __call__(self, x1, x2=None):
        length = len(self.layer_names)
        for i, block_name in enumerate(self.layer_names):
            block = self[block_name]
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(block, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class Concurrent(SimpleSequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=1):
        super(Concurrent, self).__init__()
        self.axis = axis

    def __call__(self, x):
        out = []
        for name in self.layer_names:
            out.append(self[name](x))
        out = F.concat(tuple(out), axis=self.axis)
        return out


class ParametricSequential(SimpleSequential):
    """
    A sequential container for modules with parameters.
    Blocks will be executed in the order they are added.
    """
    def __init__(self):
        super(ParametricSequential, self).__init__()

    def __call__(self, x, **kwargs):
        for name in self.layer_names:
            x = self[name](x, **kwargs)
        return x


class ParametricConcurrent(SimpleSequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=1):
        super(ParametricConcurrent, self).__init__()
        self.axis = axis

    def __call__(self, x, **kwargs):
        out = []
        for name in self.layer_names:
            out.append(self[name](x, **kwargs))
        out = F.concat(tuple(out), axis=self.axis)
        return out


class Hourglass(Chain):
    """
    A hourglass block.

    Parameters:
    ----------
    down_seq : SimpleSequential
        Down modules as sequential.
    up_seq : SimpleSequential
        Up modules as sequential.
    skip_seq : SimpleSequential
        Skip connection modules as sequential.
    merge_type : str, default 'add'
        Type of concatenation of up and skip outputs.
    return_first_skip : bool, default False
        Whether return the first skip connection output. Used in ResAttNet.
    """
    def __init__(self,
                 down_seq,
                 up_seq,
                 skip_seq,
                 merge_type="add",
                 return_first_skip=False):
        super(Hourglass, self).__init__()
        assert (len(up_seq) == len(down_seq))
        assert (len(skip_seq) == len(down_seq))
        assert (merge_type in ["add"])
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.depth = len(down_seq)

        with self.init_scope():
            self.down_seq = down_seq
            self.up_seq = up_seq
            self.skip_seq = skip_seq

    def __call__(self, x):
        y = None
        down_outs = [x]
        for down_module_name in self.down_seq.layer_names:
            down_module = self.down_seq[down_module_name]
            x = down_module(x)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module_name = self.skip_seq.layer_names[self.depth - i]
                skip_module = self.skip_seq[skip_module_name]
                y = skip_module(y)
                if (y is not None) and (self.merge_type == "add"):
                    x = x + y
            if i != len(down_outs) - 1:
                up_module_name = self.up_seq.layer_names[self.depth - 1 - i]
                up_module = self.up_seq[up_module_name]
                x = up_module(x)
        if self.return_first_skip:
            return x, y
        else:
            return x


class SesquialteralHourglass(Chain):
    """
    A sesquialteral hourglass block.

    Parameters:
    ----------
    down1_seq : SimpleSequential
        The first down modules as sequential.
    skip1_seq : SimpleSequential
        The first skip connection modules as sequential.
    up_seq : SimpleSequential
        Up modules as sequential.
    skip2_seq : SimpleSequential
        The second skip connection modules as sequential.
    down2_seq : SimpleSequential
        The second down modules as sequential.
    merge_type : str, default 'con'
        Type of concatenation of up and skip outputs.
    """
    def __init__(self,
                 down1_seq,
                 skip1_seq,
                 up_seq,
                 skip2_seq,
                 down2_seq,
                 merge_type="cat"):
        super(SesquialteralHourglass, self).__init__()
        assert (len(down1_seq) == len(up_seq))
        assert (len(down1_seq) == len(down2_seq))
        assert (len(skip1_seq) == len(skip2_seq))
        assert (len(down1_seq) == len(skip1_seq) - 1)
        assert (merge_type in ["cat", "add"])
        self.merge_type = merge_type
        self.depth = len(down1_seq)

        with self.init_scope():
            self.down1_seq = down1_seq
            self.skip1_seq = skip1_seq
            self.up_seq = up_seq
            self.skip2_seq = skip2_seq
            self.down2_seq = down2_seq

    def _merge(self, x, y):
        if y is not None:
            if self.merge_type == "cat":
                x = F.concat((x, y), axis=1)
            elif self.merge_type == "add":
                x = x + y
        return x

    def __call__(self, x):
        y = self.skip1_seq[self.skip1_seq.layer_names[0]](x)
        skip1_outs = [y]
        for i in range(self.depth):
            x = self.down1_seq[self.down1_seq.layer_names[i]](x)
            y = self.skip1_seq[self.skip1_seq.layer_names[i + 1]](x)
            skip1_outs.append(y)
        x = skip1_outs[self.depth]
        y = self.skip2_seq[self.skip2_seq.layer_names[0]](x)
        skip2_outs = [y]
        for i in range(self.depth):
            x = self.up_seq[self.up_seq.layer_names[i]](x)
            y = skip1_outs[self.depth - 1 - i]
            x = self._merge(x, y)
            y = self.skip2_seq[self.skip2_seq.layer_names[i + 1]](x)
            skip2_outs.append(y)
        x = self.skip2_seq[self.skip2_seq.layer_names[self.depth]](x)
        for i in range(self.depth):
            x = self.down2_seq[self.down2_seq.layer_names[i]](x)
            y = skip2_outs[self.depth - 1 - i]
            x = self._merge(x, y)
        return x


class MultiOutputSequential(SimpleSequential):
    """
    A sequential container with multiple outputs.
    Blocks will be executed in the order they are added.
    """
    def __init__(self):
        super(MultiOutputSequential, self).__init__()

    def __call__(self, x):
        outs = []
        for name in self.layer_names:
            block = self[name]
            x = block(x)
            if hasattr(block, "do_output") and block.do_output:
                outs.append(x)
        return [x] + outs
