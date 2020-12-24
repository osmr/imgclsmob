"""
    HarDNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.
"""

__all__ = ['HarDNet', 'hardnet39ds', 'hardnet68ds', 'hardnet68', 'hardnet85']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv_block, MaxPool2d, SimpleSequential,\
    flatten, get_channel_axis, is_channels_first


class InvDwsConvBlock(nn.Layer):
    """
    Inverse depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    pw_activation : function or str or None, default 'relu'
        Activation function after the pointwise convolution block.
    dw_activation : function or str or None, default 'relu'
        Activation function after the depthwise convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 pw_activation="relu",
                 dw_activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(InvDwsConvBlock, self).__init__(**kwargs)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=pw_activation,
            data_format=data_format,
            name="pw_conv")
        self.dw_conv = dwconv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=dw_activation,
            data_format=data_format,
            name="dw_conv")

    def call(self, x, training=None):
        x = self.pw_conv(x, training=training)
        x = self.dw_conv(x, training=training)
        return x


def invdwsconv3x3_block(in_channels,
                        out_channels,
                        strides=1,
                        padding=1,
                        dilation=1,
                        use_bias=False,
                        use_bn=True,
                        bn_eps=1e-5,
                        pw_activation="relu",
                        dw_activation="relu",
                        data_format="channels_last",
                        **kwargs):
    """
    3x3 inverse depthwise separable version of the standard convolution block.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    pw_activation : function or str or None, default 'relu'
        Activation function after the pointwise convolution block.
    dw_activation : function or str or None, default 'relu'
        Activation function after the depthwise convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return InvDwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        pw_activation=pw_activation,
        dw_activation=dw_activation,
        data_format=data_format,
        **kwargs)


class HarDUnit(nn.Layer):
    """
    HarDNet unit.

    Parameters:
    ----------
    in_channels_list : list of int
        Number of input channels for each block.
    out_channels_list : list of int
        Number of output channels for each block.
    links_list : list of list of int
        List of indices for each layer.
    use_deptwise : bool
        Whether to use depthwise downsampling.
    use_dropout : bool
        Whether to use dropout module.
    downsampling : bool
        Whether to downsample input.
    activation : str
        Name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 links_list,
                 use_deptwise,
                 use_dropout,
                 downsampling,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(HarDUnit, self).__init__(**kwargs)
        self.data_format = data_format
        self.links_list = links_list
        self.use_dropout = use_dropout
        self.downsampling = downsampling

        self.blocks = SimpleSequential(name="blocks")
        for i in range(len(links_list)):
            in_channels = in_channels_list[i]
            out_channels = out_channels_list[i]
            if use_deptwise:
                unit = invdwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    pw_activation=activation,
                    dw_activation=None,
                    data_format=data_format,
                    name="block{}".format(i + 1))
            else:
                unit = conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    data_format=data_format,
                    name="block{}".format(i + 1))
            self.blocks.add(unit)

        if self.use_dropout:
            self.dropout = nn.Dropout(
                rate=0.1,
                name="dropout")
        self.conv = conv1x1_block(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            activation=activation,
            data_format=data_format,
            name="conv")

        if self.downsampling:
            if use_deptwise:
                self.downsample = dwconv3x3_block(
                    in_channels=out_channels_list[-1],
                    out_channels=out_channels_list[-1],
                    strides=2,
                    activation=None,
                    data_format=data_format,
                    name="downsample")
            else:
                self.downsample = MaxPool2d(
                    pool_size=2,
                    strides=2,
                    data_format=data_format,
                    name="downsample")

    def call(self, x, training=None):
        axis = get_channel_axis(self.data_format)
        layer_outs = [x]
        for links_i, layer_i in zip(self.links_list, self.blocks.children):
            layer_in = []
            for idx_ij in links_i:
                layer_in.append(layer_outs[idx_ij])
            if len(layer_in) > 1:
                x = tf.concat(layer_in, axis=axis)
            else:
                x = layer_in[0]
            out = layer_i(x, training=training)
            layer_outs.append(out)

        outs = []
        for i, layer_out_i in enumerate(layer_outs):
            if (i == len(layer_outs) - 1) or (i % 2 == 1):
                outs.append(layer_out_i)
        x = tf.concat(outs, axis=axis)

        if self.use_dropout:
            x = self.dropout(x, training=training)
        x = self.conv(x, training=training)

        if self.downsampling:
            x = self.downsample(x, training=training)
        return x


class HarDInitBlock(nn.Layer):
    """
    HarDNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_deptwise : bool
        Whether to use depthwise downsampling.
    activation : str
        Name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_deptwise,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(HarDInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            activation=activation,
            data_format=data_format,
            name="conv1")
        conv2_block_class = conv1x1_block if use_deptwise else conv3x3_block
        self.conv2 = conv2_block_class(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=activation,
            data_format=data_format,
            name="conv2")
        if use_deptwise:
            self.downsample = dwconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=2,
                activation=None,
                data_format=data_format,
                name="downsample")
        else:
            self.downsample = MaxPool2d(
                pool_size=3,
                strides=2,
                padding=1,
                data_format=data_format,
                name="downsample")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.downsample(x, training=training)
        return x


class HarDNet(tf.keras.Model):
    """
    HarDNet model from 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    init_block_channels : int
        Number of output channels for the initial unit.
    unit_in_channels : list of list of list of int
        Number of input channels for each layer in each stage.
    unit_out_channels : list list of of list of int
        Number of output channels for each layer in each stage.
    unit_links : list of list of list of int
        List of indices for each layer in each stage.
    use_deptwise : bool
        Whether to use depthwise downsampling.
    use_last_dropout : bool
        Whether to use dropouts in the last unit.
    output_dropout_rate : float
        Parameter of Dropout layer before classifier. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 init_block_channels,
                 unit_in_channels,
                 unit_out_channels,
                 unit_links,
                 use_deptwise,
                 use_last_dropout,
                 output_dropout_rate,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(HarDNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        activation = "relu6"

        self.features = SimpleSequential(name="features")
        self.features.add(HarDInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            use_deptwise=use_deptwise,
            activation=activation,
            data_format=data_format,
            name="init_block"))
        for i, (in_channels_list_i, out_channels_list_i) in enumerate(zip(unit_in_channels, unit_out_channels)):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, (in_channels_list_ij, out_channels_list_ij) in enumerate(zip(in_channels_list_i,
                                                                                out_channels_list_i)):
                use_dropout = ((j == len(in_channels_list_i) - 1) and (i == len(unit_in_channels) - 1) and
                               use_last_dropout)
                downsampling = ((j == len(in_channels_list_i) - 1) and (i != len(unit_in_channels) - 1))
                stage.add(HarDUnit(
                    in_channels_list=in_channels_list_ij,
                    out_channels_list=out_channels_list_ij,
                    links_list=unit_links[i][j],
                    use_deptwise=use_deptwise,
                    use_dropout=use_dropout,
                    downsampling=downsampling,
                    activation=activation,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
            self.features.add(stage)
        in_channels = unit_out_channels[-1][-1][-1]
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(nn.Dropout(
            rate=output_dropout_rate,
            name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_hardnet(blocks,
                use_deptwise=True,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create HarDNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    use_deepwise : bool, default True
        Whether to use depthwise separable version of the model.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if blocks == 39:
        init_block_channels = 48
        growth_factor = 1.6
        dropout_rate = 0.05 if use_deptwise else 0.1
        layers = [4, 16, 8, 4]
        channels_per_layers = [96, 320, 640, 1024]
        growth_rates = [16, 20, 64, 160]
        downsamples = [1, 1, 1, 0]
        use_dropout = False
    elif blocks == 68:
        init_block_channels = 64
        growth_factor = 1.7
        dropout_rate = 0.05 if use_deptwise else 0.1
        layers = [8, 16, 16, 16, 4]
        channels_per_layers = [128, 256, 320, 640, 1024]
        growth_rates = [14, 16, 20, 40, 160]
        downsamples = [1, 0, 1, 1, 0]
        use_dropout = False
    elif blocks == 85:
        init_block_channels = 96
        growth_factor = 1.7
        dropout_rate = 0.05 if use_deptwise else 0.2
        layers = [8, 16, 16, 16, 16, 4]
        channels_per_layers = [192, 256, 320, 480, 720, 1280]
        growth_rates = [24, 24, 28, 36, 48, 256]
        downsamples = [1, 0, 1, 0, 1, 0]
        use_dropout = True
    else:
        raise ValueError("Unsupported HarDNet version with number of layers {}".format(blocks))

    assert (downsamples[-1] == 0)

    def calc_stage_params():

        def calc_unit_params():

            def calc_blocks_params(layer_idx,
                                   base_channels,
                                   growth_rate):
                if layer_idx == 0:
                    return base_channels, 0, []
                out_channels_ij = growth_rate
                links_ij = []
                for k in range(10):
                    dv = 2 ** k
                    if layer_idx % dv == 0:
                        t = layer_idx - dv
                        links_ij.append(t)
                        if k > 0:
                            out_channels_ij *= growth_factor
                out_channels_ij = int(int(out_channels_ij + 1) / 2) * 2
                in_channels_ij = 0
                for t in links_ij:
                    out_channels_ik, _, _ = calc_blocks_params(
                        layer_idx=t,
                        base_channels=base_channels,
                        growth_rate=growth_rate)
                    in_channels_ij += out_channels_ik
                return out_channels_ij, in_channels_ij, links_ij

            unit_out_channels = []
            unit_in_channels = []
            unit_links = []
            for num_layers, growth_rate, base_channels, channels_per_layers_i in zip(
                    layers, growth_rates, [init_block_channels] + channels_per_layers[:-1], channels_per_layers):
                stage_out_channels_i = 0
                unit_out_channels_i = []
                unit_in_channels_i = []
                unit_links_i = []
                for j in range(num_layers):
                    out_channels_ij, in_channels_ij, links_ij = calc_blocks_params(
                        layer_idx=(j + 1),
                        base_channels=base_channels,
                        growth_rate=growth_rate)
                    unit_out_channels_i.append(out_channels_ij)
                    unit_in_channels_i.append(in_channels_ij)
                    unit_links_i.append(links_ij)
                    if (j % 2 == 0) or (j == num_layers - 1):
                        stage_out_channels_i += out_channels_ij
                unit_in_channels_i.append(stage_out_channels_i)
                unit_out_channels_i.append(channels_per_layers_i)
                unit_out_channels.append(unit_out_channels_i)
                unit_in_channels.append(unit_in_channels_i)
                unit_links.append(unit_links_i)
            return unit_out_channels, unit_in_channels, unit_links

        unit_out_channels, unit_in_channels, unit_links = calc_unit_params()

        stage_out_channels = []
        stage_in_channels = []
        stage_links = []
        stage_out_channels_k = None
        for i in range(len(layers)):
            if stage_out_channels_k is None:
                stage_out_channels_k = []
                stage_in_channels_k = []
                stage_links_k = []
            stage_out_channels_k.append(unit_out_channels[i])
            stage_in_channels_k.append(unit_in_channels[i])
            stage_links_k.append(unit_links[i])
            if (downsamples[i] == 1) or (i == len(layers) - 1):
                stage_out_channels.append(stage_out_channels_k)
                stage_in_channels.append(stage_in_channels_k)
                stage_links.append(stage_links_k)
                stage_out_channels_k = None

        return stage_out_channels, stage_in_channels, stage_links

    stage_out_channels, stage_in_channels, stage_links = calc_stage_params()

    net = HarDNet(
        init_block_channels=init_block_channels,
        unit_in_channels=stage_in_channels,
        unit_out_channels=stage_out_channels,
        unit_links=stage_links,
        use_deptwise=use_deptwise,
        use_last_dropout=use_dropout,
        output_dropout_rate=dropout_rate,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def hardnet39ds(**kwargs):
    """
    HarDNet-39DS (Depthwise Separable) model from 'HarDNet: A Low Memory Traffic Network,'
    https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=39, use_deptwise=True, model_name="hardnet39ds", **kwargs)


def hardnet68ds(**kwargs):
    """
    HarDNet-68DS (Depthwise Separable) model from 'HarDNet: A Low Memory Traffic Network,'
    https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=68, use_deptwise=True, model_name="hardnet68ds", **kwargs)


def hardnet68(**kwargs):
    """
    HarDNet-68 model from 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=68, use_deptwise=False, model_name="hardnet68", **kwargs)


def hardnet85(**kwargs):
    """
    HarDNet-85 model from 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=85, use_deptwise=False, model_name="hardnet85", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        hardnet39ds,
        hardnet68ds,
        hardnet68,
        hardnet85,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != hardnet39ds or weight_count == 3488228)
        assert (model != hardnet68ds or weight_count == 4180602)
        assert (model != hardnet68 or weight_count == 17565348)
        assert (model != hardnet85 or weight_count == 36670212)


if __name__ == "__main__":
    _test()
