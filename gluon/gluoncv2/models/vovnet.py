"""
    VoVNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.
"""

__all__ = ['VoVNet', 'vovnet27s', 'vovnet39', 'vovnet57']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block, SequentialConcurrent


class VoVUnit(HybridBlock):
    """
    VoVNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    branch_channels : int
        Number of output channels for each branch.
    num_branches : int
        Number of branches.
    resize : bool
        Whether to use resize block.
    use_residual : bool
        Whether to use residual block.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 branch_channels,
                 num_branches,
                 resize,
                 use_residual,
                 bn_use_global_stats=False,
                 **kwargs):
        super(VoVUnit, self).__init__(**kwargs)
        self.resize = resize
        self.use_residual = use_residual

        with self.name_scope():
            if self.resize:
                self.pool = nn.MaxPool2D(
                    pool_size=3,
                    strides=2,
                    ceil_mode=True)

            self.branches = SequentialConcurrent(prefix="")
            with self.branches.name_scope():
                branch_in_channels = in_channels
                for i in range(num_branches):
                    self.branches.add(conv3x3_block(
                        in_channels=branch_in_channels,
                        out_channels=branch_channels,
                        bn_use_global_stats=bn_use_global_stats))
                    branch_in_channels = branch_channels

            self.concat_conv = conv1x1_block(
                in_channels=(in_channels + num_branches * branch_channels),
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        if self.resize:
            x = self.pool(x)
        if self.use_residual:
            identity = x
        x = self.branches(x)
        x = self.concat_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class VoVInitBlock(HybridBlock):
    """
    VoVNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(VoVInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class VoVNet(HybridBlock):
    """
    VoVNet model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    branch_channels : list of list of int
        Number of branch output channels for each unit.
    num_branches : int
        Number of branches for the each unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 branch_channels,
                 num_branches,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(VoVNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        init_block_channels = 128

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(VoVInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        use_residual = (j != 0)
                        resize = (j == 0) and (i != 0)
                        stage.add(VoVUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            branch_channels=branch_channels[i][j],
                            num_branches=num_branches,
                            resize=resize,
                            use_residual=use_residual,
                            bn_use_global_stats=bn_use_global_stats))
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


def get_vovnet(blocks,
               slim=False,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    slim : bool, default False
        Whether to use a slim model.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if blocks == 27:
        layers = [1, 1, 1, 1]
    elif blocks == 39:
        layers = [1, 1, 2, 2]
    elif blocks == 57:
        layers = [1, 1, 4, 3]
    else:
        raise ValueError("Unsupported VoVNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 6 + 3 == blocks)

    num_branches = 5
    channels_per_layers = [256, 512, 768, 1024]
    branch_channels_per_layers = [128, 160, 192, 224]
    if slim:
        channels_per_layers = [ci // 2 for ci in channels_per_layers]
        branch_channels_per_layers = [ci // 2 for ci in branch_channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    branch_channels = [[ci] * li for (ci, li) in zip(branch_channels_per_layers, layers)]

    net = VoVNet(
        channels=channels,
        branch_channels=branch_channels,
        num_branches=num_branches,
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


def vovnet27s(**kwargs):
    """
    VoVNet-27-slim model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=27, slim=True, model_name="vovnet27s", **kwargs)


def vovnet39(**kwargs):
    """
    VoVNet-39 model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=39, model_name="vovnet39", **kwargs)


def vovnet57(**kwargs):
    """
    VoVNet-57 model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=57, model_name="vovnet57", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        vovnet27s,
        vovnet39,
        vovnet57,
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
        assert (model != vovnet27s or weight_count == 3525736)
        assert (model != vovnet39 or weight_count == 22600296)
        assert (model != vovnet57 or weight_count == 36640296)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
