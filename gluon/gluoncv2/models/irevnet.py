"""
    i-RevNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.
"""

__all__ = ['IRevNet', 'irevnet301', 'IRevDownscale', 'IRevSplitBlock', 'IRevMergeBlock']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3, pre_conv3x3_block, DualPathSequential


class IRevDualPathSequential(DualPathSequential):
    """
    An invertible sequential container for hybrid blocks with dual inputs/outputs.
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
    last_noninvertible : int, default 0
        Number of the final blocks skipped during inverse.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda block, x1, x2: block(x1, x2)),
                 dual_path_scheme_ordinal=(lambda block, x1, x2: (block(x1), x2)),
                 last_noninvertible=0,
                 **kwargs):
        super(IRevDualPathSequential, self).__init__(
            return_two=return_two,
            first_ordinals=first_ordinals,
            last_ordinals=last_ordinals,
            dual_path_scheme=dual_path_scheme,
            dual_path_scheme_ordinal=dual_path_scheme_ordinal,
            **kwargs)
        self.last_noninvertible = last_noninvertible

    def inverse(self, x1, x2=None):
        length = len(self._children.values())
        for i, block in enumerate(reversed(self._children.values())):
            if i < self.last_noninvertible:
                pass
            elif (i < self.last_ordinals) or (i >= length - self.first_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(block.inverse, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(block.inverse, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class IRevDownscale(HybridBlock):
    """
    i-RevNet specific downscale (so-called psi-block).

    Parameters:
    ----------
    scale : int
        Scale (downscale) value.
    """
    def __init__(self,
                 scale,
                 **kwargs):
        super(IRevDownscale, self).__init__(**kwargs)
        self.scale = scale

    def hybrid_forward(self, F, x):
        batch, x_channels, x_height, x_width = x.shape
        y_channels = x_channels * self.scale * self.scale
        assert (x_height % self.scale == 0)
        y_height = x_height // self.scale

        y = x.transpose(axes=(0, 2, 3, 1))
        d2_split_seq = y.split(axis=2, num_outputs=(y.shape[2] // self.scale))
        d2_split_seq = [t.reshape(batch, y_height, y_channels) for t in d2_split_seq]
        y = F.stack(*d2_split_seq, axis=1)
        y = y.transpose(axes=(0, 3, 2, 1))
        return y

    def inverse(self, y):
        import mxnet.ndarray as F

        scale_sqr = self.scale * self.scale
        batch, y_channels, y_height, y_width = y.shape
        assert (y_channels % scale_sqr == 0)
        x_channels = y_channels // scale_sqr
        x_height = y_height * self.scale
        x_width = y_width * self.scale

        x = y.transpose(axes=(0, 2, 3, 1))
        x = x.reshape(batch, y_height, y_width, scale_sqr, x_channels)
        d3_split_seq = x.split(axis=3, num_outputs=(x.shape[3] // self.scale))
        d3_split_seq = [t.reshape(batch, y_height, x_width, x_channels) for t in d3_split_seq]
        x = F.stack(*d3_split_seq, axis=0)
        x = x.swapaxes(0, 1).transpose(axes=(0, 2, 1, 3, 4)).reshape(batch, x_height, x_width, x_channels)
        x = x.transpose(axes=(0, 3, 1, 2))
        return x


class IRevInjectivePad(HybridBlock):
    """
    i-RevNet channel zero padding block.

    Parameters:
    ----------
    padding : int
        Size of the padding.
    """
    def __init__(self,
                 padding,
                 **kwargs):
        super(IRevInjectivePad, self).__init__(**kwargs)
        self.padding = padding

    def hybrid_forward(self, F, x):
        x = x.transpose(axes=(0, 2, 1, 3))
        x = F.pad(x, mode="constant", pad_width=(0, 0, 0, 0, 0, self.padding, 0, 0), constant_value=0)
        x = x.transpose(axes=(0, 2, 1, 3))
        return x

    def inverse(self, x):
        return x[:, :x.shape[1] - self.padding, :, :]


class IRevSplitBlock(HybridBlock):
    """
    iRevNet split block.
    """
    def __init__(self,
                 **kwargs):
        super(IRevSplitBlock, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, _):
        x1, x2 = F.split(x, axis=1, num_outputs=2)
        return x1, x2

    def inverse(self, x1, x2):
        import mxnet.ndarray as F
        x = F.concat(x1, x2, dim=1)
        return x, None


class IRevMergeBlock(HybridBlock):
    """
    iRevNet merge block.
    """
    def __init__(self,
                 **kwargs):
        super(IRevMergeBlock, self).__init__(**kwargs)

    def hybrid_forward(self, F, x1, x2):
        x = F.concat(x1, x2, dim=1)
        return x, x

    def inverse(self, x, _):
        import mxnet.ndarray as F
        x1, x2 = F.split(x, axis=1, num_outputs=2)
        return x1, x2


class IRevBottleneck(HybridBlock):
    """
    iRevNet bottleneck block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 preactivate,
                 **kwargs):
        super(IRevBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            if preactivate:
                self.conv1 = pre_conv3x3_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            else:
                self.conv1 = conv3x3(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    strides=strides)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IRevUnit(HybridBlock):
    """
    iRevNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 preactivate,
                 **kwargs):
        super(IRevUnit, self).__init__(**kwargs)
        if not preactivate:
            in_channels = in_channels // 2

        padding = 2 * (out_channels - in_channels)
        self.do_padding = (padding != 0) and (strides == 1)
        self.do_downscale = (strides != 1)

        with self.name_scope():
            if self.do_padding:
                self.pad = IRevInjectivePad(padding)
            self.bottleneck = IRevBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                preactivate=preactivate)
            if self.do_downscale:
                self.psi = IRevDownscale(strides)

    def hybrid_forward(self, F, x1, x2):
        if self.do_padding:
            x = F.concat(x1, x2, dim=1)
            x = self.pad(x)
            x1, x2 = F.split(x, axis=1, num_outputs=2)
        fx2 = self.bottleneck(x2)
        if self.do_downscale:
            x1 = self.psi(x1)
            x2 = self.psi(x2)
        y1 = fx2 + x1
        return x2, y1

    def inverse(self, x2, y1):
        import mxnet.ndarray as F

        if self.do_downscale:
            x2 = self.psi.inverse(x2)
        fx2 = - self.bottleneck(x2)
        x1 = fx2 + y1
        if self.do_downscale:
            x1 = self.psi.inverse(x1)
        if self.do_padding:
            x = F.concat(x1, x2, dim=1)
            x = self.pad.inverse(x)
            x1, x2 = F.split(x, axis=1, num_outputs=2)
        return x1, x2


class IRevPostActivation(HybridBlock):
    """
    iRevNet specific post-activation block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(IRevPostActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class IRevNet(HybridBlock):
    """
    i-RevNet model from 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
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
                 final_block_channels,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(IRevNet, self).__init__(**kwargs)
        assert (in_channels > 0)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = IRevDualPathSequential(
                first_ordinals=1,
                last_ordinals=2,
                last_noninvertible=2)
            self.features.add(IRevDownscale(scale=2))
            in_channels = init_block_channels
            self.features.add(IRevSplitBlock())
            for i, channels_per_stage in enumerate(channels):
                stage = IRevDualPathSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) else 1
                        preactivate = not ((i == 0) and (j == 0))
                        stage.add(IRevUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            preactivate=preactivate))
                        in_channels = out_channels
                self.features.add(stage)
            in_channels = final_block_channels
            self.features.add(IRevMergeBlock())
            self.features.add(IRevPostActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x, return_out_bij=False):
        x, out_bij = self.features(x)
        x = self.output(x)
        if return_out_bij:
            return x, out_bij
        else:
            return x

    def inverse(self, out_bij):
        x, _ = self.features.inverse(out_bij)
        return x


def get_irevnet(blocks,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create i-RevNet model with specific parameters.

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

    if blocks == 301:
        layers = [6, 16, 72, 6]
    else:
        raise ValueError("Unsupported i-RevNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 1 == blocks)

    channels_per_layers = [24, 96, 384, 1536]
    init_block_channels = 12
    final_block_channels = 3072

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = IRevNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ignore_extra=True,
            ctx=ctx)

    return net


def irevnet301(**kwargs):
    """
    i-RevNet-301 model from 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_irevnet(blocks=301, model_name="irevnet301", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        irevnet301,
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
        assert (model != irevnet301 or weight_count == 125120356)

        x = mx.nd.random.randn(2, 3, 224, 224, ctx=ctx)
        y = net(x)
        assert (y.shape == (2, 1000))

        y, out_bij = net(x, True)
        x_ = net.inverse(out_bij)
        assert (x_.shape == (2, 3, 224, 224))

        assert ((np.max(np.abs(x.asnumpy() - x_.asnumpy())) < 1e-4) or (np.max(np.abs(y.asnumpy()) > 1e10)))


if __name__ == "__main__":
    _test()
