"""
    HarDNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.
"""

__all__ = ['HarDNet', 'hardnet39ds', 'hardnet68ds', 'hardnet68', 'hardnet85']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv_block, SimpleSequential


class InvDwsConvBlock(Chain):
    """
    Inverse depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    pw_activation : function or str or None, default F.relu
        Activation function after the pointwise convolution block.
    dw_activation : function or str or None, default F.relu
        Activation function after the depthwise convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 pw_activation=(lambda: F.relu),
                 dw_activation=(lambda: F.relu)):
        super(InvDwsConvBlock, self).__init__()
        with self.init_scope():
            self.pw_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=pw_activation)
            self.dw_conv = dwconv_block(
                in_channels=out_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=dw_activation)

    def __call__(self, x):
        x = self.pw_conv(x)
        x = self.dw_conv(x)
        return x


def invdwsconv3x3_block(in_channels,
                        out_channels,
                        stride=1,
                        pad=1,
                        dilate=1,
                        use_bias=False,
                        bn_eps=1e-5,
                        pw_activation=(lambda: F.relu),
                        dw_activation=(lambda: F.relu)):
    """
    3x3 inverse depthwise separable version of the standard convolution block.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    pw_activation : function or str or None, default F.relu
        Activation function after the pointwise convolution block.
    dw_activation : function or str or None, default F.relu
        Activation function after the depthwise convolution block.
    """
    return InvDwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        bn_eps=bn_eps,
        pw_activation=pw_activation,
        dw_activation=dw_activation)


class HarDUnit(Chain):
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
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 links_list,
                 use_deptwise,
                 use_dropout,
                 downsampling,
                 activation):
        super(HarDUnit, self).__init__()
        self.links_list = links_list
        self.use_dropout = use_dropout
        self.downsampling = downsampling

        with self.init_scope():
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i in range(len(links_list)):
                    in_channels = in_channels_list[i]
                    out_channels = out_channels_list[i]
                    if use_deptwise:
                        unit = invdwsconv3x3_block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            pw_activation=activation,
                            dw_activation=None)
                    else:
                        unit = conv3x3_block(
                            in_channels=in_channels,
                            out_channels=out_channels)
                    setattr(self.blocks, "block{}".format(i + 1), unit)

            if self.use_dropout:
                self.dropout = partial(
                    F.dropout,
                    ratio=0.1)
            self.conv = conv1x1_block(
                in_channels=in_channels_list[-1],
                out_channels=out_channels_list[-1],
                activation=activation)

            if self.downsampling:
                if use_deptwise:
                    self.downsample = dwconv3x3_block(
                        in_channels=out_channels_list[-1],
                        out_channels=out_channels_list[-1],
                        stride=2,
                        activation=None)
                else:
                    self.downsample = partial(
                        F.max_pooling_2d,
                        ksize=2,
                        stride=2,
                        cover_all=False)

    def __call__(self, x):
        layer_outs = [x]
        for links_i, layer_name_i in zip(self.links_list, self.blocks.layer_names):
            layer_i = self.blocks[layer_name_i]
            layer_in = []
            for idx_ij in links_i:
                layer_in.append(layer_outs[idx_ij])
            if len(layer_in) > 1:
                x = F.concat(layer_in, axis=1)
            else:
                x = layer_in[0]
            out = layer_i(x)
            layer_outs.append(out)

        outs = []
        for i, layer_out_i in enumerate(layer_outs):
            if (i == len(layer_outs) - 1) or (i % 2 == 1):
                outs.append(layer_out_i)
        x = F.concat(outs, axis=1)

        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv(x)

        if self.downsampling:
            x = self.downsample(x)
        return x


class HarDInitBlock(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_deptwise,
                 activation):
        super(HarDInitBlock, self).__init__()
        mid_channels = out_channels // 2

        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2,
                activation=activation)
            conv2_block_class = conv1x1_block if use_deptwise else conv3x3_block
            self.conv2 = conv2_block_class(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=activation)
            if use_deptwise:
                self.downsample = dwconv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=2,
                    activation=None)
            else:
                self.downsample = partial(
                    F.max_pooling_2d,
                    ksize=3,
                    stride=2,
                    pad=1,
                    cover_all=False)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)
        return x


class HarDNet(Chain):
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
                 classes=1000):
        super(HarDNet, self).__init__()
        self.in_size = in_size
        self.classes = classes
        activation = "relu6"

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", HarDInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    use_deptwise=use_deptwise,
                    activation=activation))
                for i, (in_channels_list_i, out_channels_list_i) in enumerate(zip(unit_in_channels, unit_out_channels)):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, (in_channels_list_ij, out_channels_list_ij) in enumerate(zip(in_channels_list_i,
                                                                                            out_channels_list_i)):
                            use_dropout = ((j == len(in_channels_list_i) - 1) and (i == len(unit_in_channels) - 1) and
                                           use_last_dropout)
                            downsampling = ((j == len(in_channels_list_i) - 1) and (i != len(unit_in_channels) - 1))
                            setattr(stage, "unit{}".format(j + 1), HarDUnit(
                                in_channels_list=in_channels_list_ij,
                                out_channels_list=out_channels_list_ij,
                                links_list=unit_links[i][j],
                                use_deptwise=use_deptwise,
                                use_dropout=use_dropout,
                                downsampling=downsampling,
                                activation=activation))
                    setattr(self.features, "stage{}".format(i + 1), stage)
                in_channels = unit_out_channels[-1][-1][-1]
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "dropout", partial(
                    F.dropout,
                    ratio=output_dropout_rate))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_hardnet(blocks,
                use_deptwise=True,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def hardnet39ds(**kwargs):
    """
    HarDNet-39DS (Depthwise Separable) model from 'HarDNet: A Low Memory Traffic Network,'
    https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=85, use_deptwise=False, model_name="hardnet85", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        hardnet39ds,
        hardnet68ds,
        hardnet68,
        hardnet85,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != hardnet39ds or weight_count == 3488228)
        assert (model != hardnet68ds or weight_count == 4180602)
        assert (model != hardnet68 or weight_count == 17565348)
        assert (model != hardnet85 or weight_count == 36670212)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
