"""
    MSDNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.
"""

__all__ = ['MSDNet', 'msdnet22', 'MultiOutputSequential', 'MSDFeatureBlock']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block, DualPathSequential
from .resnet import ResInitBlock


class MultiOutputSequential(nn.HybridSequential):
    """
    A sequential container for blocks. Blocks will be executed in the order they are added. Output value contains
    results from all blocks.
    """
    def __init__(self, **kwargs):
        super(MultiOutputSequential, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        outs = []
        for block in self._children.values():
            x = block(x)
            outs.append(x)
        return outs


class MultiBlockSequential(nn.HybridSequential):
    """
    A sequential container for blocks. Blocks will be executed in the order they are added. Input is a list with
    length equal to number of blocks.
    """
    def __init__(self, **kwargs):
        super(MultiBlockSequential, self).__init__(**kwargs)

    def hybrid_forward(self, F, x0, x_rest):
        outs = []
        for block, x_i in zip(self._children.values(), [x0] + x_rest):
            y = block(x_i)
            outs.append(y)
        return outs


class MSDBaseBlock(HybridBlock):
    """
    MSDNet base block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factor : int
        Bottleneck factor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_bottleneck,
                 bottleneck_factor,
                 **kwargs):
        super(MSDBaseBlock, self).__init__(**kwargs)
        self.use_bottleneck = use_bottleneck
        mid_channels = min(in_channels, bottleneck_factor * out_channels) if use_bottleneck else in_channels

        with self.name_scope():
            if self.use_bottleneck:
                self.bn_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels)
            self.conv = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=strides)

    def hybrid_forward(self, F, x):
        if self.use_bottleneck:
            x = self.bn_conv(x)
        x = self.conv(x)
        return x


class MSDFirstScaleBlock(HybridBlock):
    """
    MSDNet first scale dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factor : int
        Bottleneck factor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factor,
                 **kwargs):
        super(MSDFirstScaleBlock, self).__init__(**kwargs)
        assert (out_channels > in_channels)
        inc_channels = out_channels - in_channels

        with self.name_scope():
            self.block = MSDBaseBlock(
                in_channels=in_channels,
                out_channels=inc_channels,
                strides=1,
                use_bottleneck=use_bottleneck,
                bottleneck_factor=bottleneck_factor)

    def hybrid_forward(self, F, x):
        y = self.block(x)
        y = F.concat(x, y, dim=1)
        return y


class MSDScaleBlock(HybridBlock):
    """
    MSDNet ordinary scale dense block.

    Parameters:
    ----------
    in_channels_prev : int
        Number of input channels for the previous scale.
    in_channels : int
        Number of input channels for the current scale.
    out_channels : int
        Number of output channels.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factor_prev : int
        Bottleneck factor for the previous scale.
    bottleneck_factor : int
        Bottleneck factor for the current scale.
    """

    def __init__(self,
                 in_channels_prev,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factor_prev,
                 bottleneck_factor,
                 **kwargs):
        super(MSDScaleBlock, self).__init__(**kwargs)
        assert (out_channels > in_channels)
        assert (out_channels % 2 == 0)
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels // 2

        with self.name_scope():
            self.down_block = MSDBaseBlock(
                in_channels=in_channels_prev,
                out_channels=mid_channels,
                strides=2,
                use_bottleneck=use_bottleneck,
                bottleneck_factor=bottleneck_factor_prev)
            self.curr_block = MSDBaseBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=1,
                use_bottleneck=use_bottleneck,
                bottleneck_factor=bottleneck_factor)

    def hybrid_forward(self, F, x_prev, x):
        y_prev = self.down_block(x_prev)
        y = self.curr_block(x)
        x = F.concat(x, y_prev, y, dim=1)
        return x


class MSDInitLayer(HybridBlock):
    """
    MSDNet initial (so-called first) layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : list/tuple of int
        Number of output channels for each scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(MSDInitLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.scale_blocks = MultiOutputSequential()
            for i, out_channels_per_scale in enumerate(out_channels):
                if i == 0:
                    self.scale_blocks.add(ResInitBlock(
                        in_channels=in_channels,
                        out_channels=out_channels_per_scale))
                else:
                    self.scale_blocks.add(conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels_per_scale,
                        strides=2))
                in_channels = out_channels_per_scale

    def hybrid_forward(self, F, x):
        y = self.scale_blocks(x)
        return y


class MSDLayer(HybridBlock):
    """
    MSDNet ordinary layer.

    Parameters:
    ----------
    in_channels : list/tuple of int
        Number of input channels for each input scale.
    out_channels : list/tuple of int
        Number of output channels for each output scale.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factors : list/tuple of int
        Bottleneck factor for each input scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factors,
                 **kwargs):
        super(MSDLayer, self).__init__(**kwargs)
        in_scales = len(in_channels)
        out_scales = len(out_channels)
        self.dec_scales = in_scales - out_scales
        assert (self.dec_scales >= 0)

        with self.name_scope():
            self.scale_blocks = nn.HybridSequential(prefix="")
            for i in range(out_scales):
                if (i == 0) and (self.dec_scales == 0):
                    self.scale_blocks.add(MSDFirstScaleBlock(
                        in_channels=in_channels[self.dec_scales + i],
                        out_channels=out_channels[i],
                        use_bottleneck=use_bottleneck,
                        bottleneck_factor=bottleneck_factors[self.dec_scales + i]))
                else:
                    self.scale_blocks.add(MSDScaleBlock(
                        in_channels_prev=in_channels[self.dec_scales + i - 1],
                        in_channels=in_channels[self.dec_scales + i],
                        out_channels=out_channels[i],
                        use_bottleneck=use_bottleneck,
                        bottleneck_factor_prev=bottleneck_factors[self.dec_scales + i - 1],
                        bottleneck_factor=bottleneck_factors[self.dec_scales + i]))

    def hybrid_forward(self, F, x0, x_rest):
        x = [x0] + x_rest
        outs = []
        for i in range(len(self.scale_blocks)):
            if (i == 0) and (self.dec_scales == 0):
                y = self.scale_blocks[i](x[i])
            else:
                y = self.scale_blocks[i](
                    x[self.dec_scales + i - 1],
                    x[self.dec_scales + i])
            outs.append(y)
        return outs[0], outs[1:]


class MSDTransitionLayer(HybridBlock):
    """
    MSDNet transition layer.

    Parameters:
    ----------
    in_channels : list/tuple of int
        Number of input channels for each scale.
    out_channels : list/tuple of int
        Number of output channels for each scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(MSDTransitionLayer, self).__init__(**kwargs)
        assert (len(in_channels) == len(out_channels))

        with self.name_scope():
            self.scale_blocks = MultiBlockSequential()
            for i in range(len(out_channels)):
                self.scale_blocks.add(conv1x1_block(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i]))

    def hybrid_forward(self, F, x0, x_rest):
        y = self.scale_blocks(x0, x_rest)
        return y[0], y[1:]


class MSDFeatureBlock(HybridBlock):
    """
    MSDNet feature block (stage of cascade, so-called block).

    Parameters:
    ----------
    in_channels : list of list of int
        Number of input channels for each layer and for each input scale.
    out_channels : list of list of int
        Number of output channels for each layer and for each output scale.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factors : list of list of int
        Bottleneck factor for each layer and for each input scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factors,
                 **kwargs):
        super(MSDFeatureBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.blocks = DualPathSequential(prefix="")
            for i, out_channels_per_layer in enumerate(out_channels):
                if len(bottleneck_factors[i]) == 0:
                    self.blocks.add(MSDTransitionLayer(
                        in_channels=in_channels,
                        out_channels=out_channels_per_layer))
                else:
                    self.blocks.add(MSDLayer(
                        in_channels=in_channels,
                        out_channels=out_channels_per_layer,
                        use_bottleneck=use_bottleneck,
                        bottleneck_factors=bottleneck_factors[i]))
                in_channels = out_channels_per_layer

    def hybrid_forward(self, F, x0, x_rest):
        x0, x_rest = self.blocks(x0, x_rest)
        return [x0] + x_rest


class MSDClassifier(HybridBlock):
    """
    MSDNet classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """

    def __init__(self,
                 in_channels,
                 classes,
                 **kwargs):
        super(MSDClassifier, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2))
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2))
            self.features.add(nn.AvgPool2D(
                pool_size=2,
                strides=2))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class MSDNet(HybridBlock):
    """
    MSDNet model from 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.

    Parameters:
    ----------
    channels : list of list of list of int
        Number of output channels for each unit.
    init_layer_channels : list of int
        Number of output channels for the initial layer.
    num_feature_blocks : int
        Number of subnets.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factors : list of list of int
        Bottleneck factor for each layers and for each input scale.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_layer_channels,
                 num_feature_blocks,
                 use_bottleneck,
                 bottleneck_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(MSDNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.init_layer = MSDInitLayer(
                in_channels=in_channels,
                out_channels=init_layer_channels)
            in_channels = init_layer_channels

            self.feature_blocks = nn.HybridSequential(prefix="")
            self.classifiers = nn.HybridSequential(prefix="")
            for i in range(num_feature_blocks):
                self.feature_blocks.add(MSDFeatureBlock(
                    in_channels=in_channels,
                    out_channels=channels[i],
                    use_bottleneck=use_bottleneck,
                    bottleneck_factors=bottleneck_factors[i]))
                in_channels = channels[i][-1]
                self.classifiers.add(MSDClassifier(
                    in_channels=in_channels[-1],
                    classes=classes))

    def hybrid_forward(self, F, x, only_last=True):
        x = self.init_layer(x)
        outs = []
        for feature_block, classifier in zip(self.feature_blocks, self.classifiers):
            x = feature_block(x[0], x[1:])
            y = classifier(x[-1])
            outs.append(y)
        if only_last:
            return outs[-1]
        else:
            return outs


def get_msdnet(blocks,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create MSDNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    assert (blocks == 22)

    num_scales = 4
    num_feature_blocks = 10
    base = 4
    step = 2
    reduction_rate = 0.5
    growth = 6
    growth_factor = [1, 2, 4, 4]
    use_bottleneck = True
    bottleneck_factor_per_scales = [1, 2, 4, 4]

    assert (reduction_rate > 0.0)
    init_layer_channels = [64 * c for c in growth_factor[:num_scales]]

    step_mode = "even"
    layers_per_subnets = [base]
    for i in range(num_feature_blocks - 1):
        layers_per_subnets.append(step if step_mode == 'even' else step * i + 1)
    total_layers = sum(layers_per_subnets)

    interval = math.ceil(total_layers / num_scales)
    global_layer_ind = 0

    channels = []
    bottleneck_factors = []

    in_channels_tmp = init_layer_channels
    in_scales = num_scales
    for i in range(num_feature_blocks):
        layers_per_subnet = layers_per_subnets[i]
        scales_i = []
        channels_i = []
        bottleneck_factors_i = []
        for j in range(layers_per_subnet):
            out_scales = int(num_scales - math.floor(global_layer_ind / interval))
            global_layer_ind += 1
            scales_i += [out_scales]
            scale_offset = num_scales - out_scales

            in_dec_scales = num_scales - len(in_channels_tmp)
            out_channels = [in_channels_tmp[scale_offset - in_dec_scales + k] + growth * growth_factor[scale_offset + k]
                            for k in range(out_scales)]
            in_dec_scales = num_scales - len(in_channels_tmp)
            bottleneck_factors_ij = bottleneck_factor_per_scales[in_dec_scales:][:len(in_channels_tmp)]

            in_channels_tmp = out_channels
            channels_i += [out_channels]
            bottleneck_factors_i += [bottleneck_factors_ij]

            if in_scales > out_scales:
                assert (in_channels_tmp[0] % growth_factor[scale_offset] == 0)
                out_channels1 = int(math.floor(in_channels_tmp[0] / growth_factor[scale_offset] * reduction_rate))
                out_channels = [out_channels1 * growth_factor[scale_offset + k] for k in range(out_scales)]
                in_channels_tmp = out_channels
                channels_i += [out_channels]
                bottleneck_factors_i += [[]]
            in_scales = out_scales

        in_scales = scales_i[-1]
        channels += [channels_i]
        bottleneck_factors += [bottleneck_factors_i]

    net = MSDNet(
        channels=channels,
        init_layer_channels=init_layer_channels,
        num_feature_blocks=num_feature_blocks,
        use_bottleneck=use_bottleneck,
        bottleneck_factors=bottleneck_factors,
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


def msdnet22(**kwargs):
    """
    MSDNet-22 model from 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_msdnet(blocks=22, model_name="msdnet22", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        msdnet22,
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
        assert (model != msdnet22 or weight_count == 20106676)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
