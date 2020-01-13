"""
    VoVNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.
"""

__all__ = ['VoVNet', 'vovnet27s', 'vovnet39', 'vovnet57']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, SequentialConcurrent, SimpleSequential


class VoVUnit(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 branch_channels,
                 num_branches,
                 resize,
                 use_residual):
        super(VoVUnit, self).__init__()
        self.resize = resize
        self.use_residual = use_residual

        with self.init_scope():
            if self.resize:
                self.pool = partial(
                    F.max_pooling_2d,
                    ksize=3,
                    stride=2,
                    cover_all=True)

            self.branches = SequentialConcurrent()
            with self.branches.init_scope():
                branch_in_channels = in_channels
                for i in range(num_branches):
                    setattr(self.branches, "branch{}".format(i + 1), conv3x3_block(
                        in_channels=branch_in_channels,
                        out_channels=branch_channels))
                    branch_in_channels = branch_channels

            self.concat_conv = conv1x1_block(
                in_channels=(in_channels + num_branches * branch_channels),
                out_channels=out_channels)

    def __call__(self, x):
        if self.resize:
            x = self.pool(x)
        if self.use_residual:
            identity = x
        x = self.branches(x)
        x = self.concat_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class VoVInitBlock(Chain):
    """
    VoVNet specific initial block.

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
        super(VoVInitBlock, self).__init__()
        mid_channels = out_channels // 2

        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.conv3 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                stride=2)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class VoVNet(Chain):
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(VoVNet, self).__init__()
        self.in_size = in_size
        self.classes = classes
        init_block_channels = 128

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", VoVInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            use_residual = (j != 0)
                            resize = (j == 0) and (i != 0)
                            setattr(stage, "unit{}".format(j + 1), VoVUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                branch_channels=branch_channels[i][j],
                                num_branches=num_branches,
                                resize=resize,
                                use_residual=use_residual))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
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


def get_vovnet(blocks,
               slim=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def vovnet27s(**kwargs):
    """
    VoVNet-27-slim model from 'An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection,'
    https://arxiv.org/abs/1904.09730.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_vovnet(blocks=57, model_name="vovnet57", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        vovnet27s,
        vovnet39,
        vovnet57,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != vovnet27s or weight_count == 3525736)
        assert (model != vovnet39 or weight_count == 22600296)
        assert (model != vovnet57 or weight_count == 36640296)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
