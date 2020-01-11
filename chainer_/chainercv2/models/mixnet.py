"""
    MixNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.
"""

__all__ = ['MixNet', 'mixnet_s', 'mixnet_m', 'mixnet_l']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import round_channels, get_activation_layer, conv1x1_block, conv3x3_block, dwconv3x3_block, SEBlock,\
    SimpleSequential


class MixConv(Chain):
    """
    Mixed convolution layer from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    axis : int, default 1
        The axis on which to concatenate the outputs.
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
                 axis=1):
        super(MixConv, self).__init__()
        self.layer_names = []
        ksize = ksize if isinstance(ksize, list) else [ksize]
        pad = pad if isinstance(pad, list) else [pad]
        kernel_count = len(ksize)
        splitted_in_channels = self.split_channels(in_channels, kernel_count)
        splitted_out_channels = self.split_channels(out_channels, kernel_count)
        self.axis = axis
        import numpy as np
        self.in_channel_inds = list(np.add.accumulate(splitted_in_channels))[:-1]

        with self.init_scope():
            for i, ksize_i in enumerate(ksize):
                in_channels_i = splitted_in_channels[i]
                out_channels_i = splitted_out_channels[i]
                pad_i = pad[i]
                setattr(self, "conv{}".format(i + 1), L.Convolution2D(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    ksize=ksize_i,
                    stride=stride,
                    pad=pad_i,
                    nobias=(not use_bias),
                    dilate=dilate,
                    groups=(out_channels_i if out_channels == groups else groups)))

    def __setattr__(self, name, value):
        super(MixConv, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self.layer_names.append(name)

    def __delattr__(self, name):
        super(MixConv, self).__delattr__(name)
        try:
            self.layer_names.remove(name)
        except ValueError:
            pass

    def __len__(self):
        return len(self.layer_names)

    def __call__(self, x):
        xx = F.split_axis(x, self.in_channel_inds, axis=self.axis)
        out = [self[name_i](x_i) for x_i, name_i in zip(xx, self.layer_names)]
        x = F.concat(tuple(out), axis=self.axis)
        return x

    @staticmethod
    def split_channels(channels, kernel_count):
        splitted_channels = [channels // kernel_count] * kernel_count
        splitted_channels[0] += channels - sum(splitted_channels)
        return splitted_channels


class MixConvBlock(Chain):
    """
    Mixed convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.activate
        Activation function or name of activation function.
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
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu)):
        super(MixConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.init_scope():
            self.conv = MixConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                groups=groups,
                use_bias=use_bias)
            if self.use_bn:
                self.bn = L.BatchNormalization(
                    size=out_channels,
                    eps=bn_eps)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def __call__(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def mixconv1x1_block(in_channels,
                     out_channels,
                     kernel_count,
                     stride=1,
                     groups=1,
                     bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=(lambda: F.relu)):
    """
    1x1 version of the mixed convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_count : int
        Kernel count.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str, or None, default F.activate
        Activation function or name of activation function.
    """
    return MixConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=([1] * kernel_count),
        stride=stride,
        pad=([0] * kernel_count),
        groups=groups,
        use_bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class MixUnit(Chain):
    """
    MixNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    stride : int or tuple/list of 2 int
        Stride of the second convolution layer.
    exp_kernel_count : int
        Expansion convolution kernel count for each unit.
    conv1_kernel_count : int
        Conv1 kernel count for each unit.
    conv2_kernel_count : int
        Conv2 kernel count for each unit.
    exp_factor : int
        Expansion factor for each unit.
    se_factor : int
        SE reduction factor for each unit.
    activation : str
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 exp_kernel_count,
                 conv1_kernel_count,
                 conv2_kernel_count,
                 exp_factor,
                 se_factor,
                 activation):
        super(MixUnit, self).__init__()
        assert (exp_factor >= 1)
        assert (se_factor >= 0)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = exp_factor * in_channels
        self.use_exp_conv = exp_factor > 1

        with self.init_scope():
            if self.use_exp_conv:
                if exp_kernel_count == 1:
                    self.exp_conv = conv1x1_block(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        activation=activation)
                else:
                    self.exp_conv = mixconv1x1_block(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        kernel_count=exp_kernel_count,
                        activation=activation)
            if conv1_kernel_count == 1:
                self.conv1 = dwconv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    stride=stride,
                    activation=activation)
            else:
                self.conv1 = MixConvBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    ksize=[3 + 2 * i for i in range(conv1_kernel_count)],
                    stride=stride,
                    pad=[1 + i for i in range(conv1_kernel_count)],
                    groups=mid_channels,
                    activation=activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=(exp_factor * se_factor),
                    round_mid=False,
                    mid_activation=activation)
            if conv2_kernel_count == 1:
                self.conv2 = conv1x1_block(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    activation=None)
            else:
                self.conv2 = mixconv1x1_block(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_count=conv2_kernel_count,
                    activation=None)

    def __call__(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MixInitBlock(Chain):
    """
    MixNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(MixInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2)
            self.conv2 = MixUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                exp_kernel_count=1,
                conv1_kernel_count=1,
                conv2_kernel_count=1,
                exp_factor=1,
                se_factor=0,
                activation="relu")

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MixNet(Chain):
    """
    MixNet model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    exp_kernel_counts : list of list of int
        Expansion convolution kernel count for each unit.
    conv1_kernel_counts : list of list of int
        Conv1 kernel count for each unit.
    conv2_kernel_counts : list of list of int
        Conv2 kernel count for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
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
                 final_block_channels,
                 exp_kernel_counts,
                 conv1_kernel_counts,
                 conv2_kernel_counts,
                 exp_factors,
                 se_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(MixNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", MixInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if ((j == 0) and (i != 3)) or\
                                          ((j == len(channels_per_stage) // 2) and (i == 3)) else 1
                            exp_kernel_count = exp_kernel_counts[i][j]
                            conv1_kernel_count = conv1_kernel_counts[i][j]
                            conv2_kernel_count = conv2_kernel_counts[i][j]
                            exp_factor = exp_factors[i][j]
                            se_factor = se_factors[i][j]
                            activation = "relu" if i == 0 else "swish"
                            setattr(stage, "unit{}".format(j + 1), MixUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                exp_kernel_count=exp_kernel_count,
                                conv1_kernel_count=conv1_kernel_count,
                                conv2_kernel_count=conv2_kernel_count,
                                exp_factor=exp_factor,
                                se_factor=se_factor,
                                activation=activation))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", conv1x1_block(
                    in_channels=in_channels,
                    out_channels=final_block_channels))
                in_channels = final_block_channels
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mixnet(version,
               width_scale,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create MixNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('s' or 'm').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if version == "s":
        init_block_channels = 16
        channels = [[24, 24], [40, 40, 40, 40], [80, 80, 80], [120, 120, 120, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 1, 1], [2, 2, 2, 1, 1, 1]]
        conv1_kernel_counts = [[1, 1], [3, 2, 2, 2], [3, 2, 2], [3, 4, 4, 5, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [2, 2, 2], [2, 2, 2, 1, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6], [6, 3, 3, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4], [2, 2, 2, 2, 2, 2]]
    elif version == "m":
        init_block_channels = 24
        channels = [[32, 32], [40, 40, 40, 40], [80, 80, 80, 80], [120, 120, 120, 120, 200, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2, 1, 1, 1, 1]]
        conv1_kernel_counts = [[3, 1], [4, 2, 2, 2], [3, 4, 4, 4], [1, 4, 4, 4, 4, 4, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2, 1, 2, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6, 6], [6, 3, 3, 3, 6, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4, 4], [2, 2, 2, 2, 2, 2, 2, 2]]
    else:
        raise ValueError("Unsupported MixNet version {}".format(version))

    final_block_channels = 1536

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale)

    net = MixNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        exp_kernel_counts=exp_kernel_counts,
        conv1_kernel_counts=conv1_kernel_counts,
        conv2_kernel_counts=conv2_kernel_counts,
        exp_factors=exp_factors,
        se_factors=se_factors,
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


def mixnet_s(**kwargs):
    """
    MixNet-S model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="s", width_scale=1.0, model_name="mixnet_s", **kwargs)


def mixnet_m(**kwargs):
    """
    MixNet-M model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="m", width_scale=1.0, model_name="mixnet_m", **kwargs)


def mixnet_l(**kwargs):
    """
    MixNet-L model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="m", width_scale=1.3, model_name="mixnet_l", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        mixnet_s,
        mixnet_m,
        mixnet_l,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mixnet_s or weight_count == 4134606)
        assert (model != mixnet_m or weight_count == 5014382)
        assert (model != mixnet_l or weight_count == 7329252)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
