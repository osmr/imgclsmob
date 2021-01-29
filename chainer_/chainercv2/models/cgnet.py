"""
    CGNet for image segmentation, implemented in Chainer.
    Original paper: 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.
"""

__all__ = ['CGNet', 'cgnet_cityscapes']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock, Concurrent,\
    DualPathSequential, SimpleSequential, InterpolationBlock


class CGBlock(Chain):
    """
    CGNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilate : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    down : bool
        Whether to downsample.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilate,
                 se_reduction,
                 down,
                 bn_eps,
                 **kwargs):
        super(CGBlock, self).__init__(**kwargs)
        self.down = down
        if self.down:
            mid1_channels = out_channels
            mid2_channels = 2 * out_channels
        else:
            mid1_channels = out_channels // 2
            mid2_channels = out_channels

        with self.init_scope():
            if self.down:
                self.conv1 = conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    bn_eps=bn_eps,
                    activation=(lambda: L.PReLU(out_channels)))
            else:
                self.conv1 = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid1_channels,
                    bn_eps=bn_eps,
                    activation=(lambda: L.PReLU(mid1_channels)))

            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branches1", depthwise_conv3x3(channels=mid1_channels))
                setattr(self.branches, "branches2", depthwise_conv3x3(
                    channels=mid1_channels,
                    pad=dilate,
                    dilate=dilate))

            self.norm_activ = NormActivation(
                in_channels=mid2_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(mid2_channels)))

            if self.down:
                self.conv2 = conv1x1(
                    in_channels=mid2_channels,
                    out_channels=out_channels)

            self.se = SEBlock(
                channels=out_channels,
                reduction=se_reduction,
                use_conv=False)

    def __call__(self, x):
        if not self.down:
            identity = x
        x = self.conv1(x)
        x = self.branches(x)
        x = self.norm_activ(x)
        if self.down:
            x = self.conv2(x)
        x = self.se(x)
        if not self.down:
            x += identity
        return x


class CGUnit(Chain):
    """
    CGNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    dilate : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 dilate,
                 se_reduction,
                 bn_eps,
                 **kwargs):
        super(CGUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.init_scope():
            self.down = CGBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                dilate=dilate,
                se_reduction=se_reduction,
                down=True,
                bn_eps=bn_eps)
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i in range(layers - 1):
                    setattr(self.blocks, "block{}".format(i + 1), CGBlock(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        dilate=dilate,
                        se_reduction=se_reduction,
                        down=False,
                        bn_eps=bn_eps))

    def __call__(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = F.concat((y, x), axis=1)  # NB: This differs from the original implementation.
        return x


class CGStage(Chain):
    """
    CGNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    layers : int
        Number of layers in the unit.
    dilate : int
        Dilation for blocks.
    se_reduction : int
        SE-block reduction value for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 dilate,
                 se_reduction,
                 bn_eps,
                 **kwargs):
        super(CGStage, self).__init__(**kwargs)
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        with self.init_scope():
            if self.use_x:
                self.x_down = partial(
                    F.average_pooling_2d,
                    ksize=3,
                    stride=2,
                    pad=1)

            if self.use_unit:
                self.unit = CGUnit(
                    in_channels=y_in_channels,
                    out_channels=(y_out_channels - x_channels),
                    layers=layers,
                    dilate=dilate,
                    se_reduction=se_reduction,
                    bn_eps=bn_eps)

            self.norm_activ = NormActivation(
                in_channels=y_out_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(y_out_channels)))

    def __call__(self, y, x=None):
        if self.use_unit:
            y = self.unit(y)
        if self.use_x:
            x = self.x_down(x)
            y = F.concat((y, x), axis=1)
        y = self.norm_activ(y)
        return y, x


class CGInitBlock(Chain):
    """
    CGNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 **kwargs):
        super(CGInitBlock, self).__init__(**kwargs)
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(out_channels)))
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(out_channels)))
            self.conv3 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(out_channels)))

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CGNet(Chain):
    """
    CGNet model from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilates : list of int
        Dilations for each unit.
    se_reductions : list of int
        SE-block reduction value for each unit.
    cut_x : list of int
        Whether to concatenate with x-branch for each unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 layers,
                 channels,
                 init_block_channels,
                 dilates,
                 se_reductions,
                 cut_x,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 **kwargs):
        super(CGNet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        with self.init_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=0)
            with self.features.init_scope():
                setattr(self.features, "init_block", CGInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    bn_eps=bn_eps))
                y_in_channels = init_block_channels

                for i, (layers_i, y_out_channels) in enumerate(zip(layers, channels)):
                    setattr(self.features, "stage{}".format(i + 1), CGStage(
                        x_channels=in_channels if cut_x[i] == 1 else 0,
                        y_in_channels=y_in_channels,
                        y_out_channels=y_out_channels,
                        layers=layers_i,
                        dilate=dilates[i],
                        se_reduction=se_reductions[i],
                        bn_eps=bn_eps))
                    y_in_channels = y_out_channels

            self.classifier = conv1x1(
                in_channels=y_in_channels,
                out_channels=classes)

            self.up = InterpolationBlock(
                scale_factor=8,
                align_corners=False)

    def __call__(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        y = self.features(x, x)
        y = self.classifier(y)
        y = self.up(y, size=in_size)
        return y


def get_cgnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
              **kwargs):
    """
    Create CGNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [0, 3, 21]
    channels = [35, 131, 256]
    dilates = [0, 2, 4]
    se_reductions = [0, 8, 16]
    cut_x = [1, 1, 0]
    bn_eps = 1e-3

    net = CGNet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        dilates=dilates,
        se_reductions=se_reductions,
        cut_x=cut_x,
        bn_eps=bn_eps,
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


def cgnet_cityscapes(classes=19, **kwargs):
    """
    CGNet model for Cityscapes from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_cgnet(classes=classes, model_name="cgnet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        cgnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cgnet_cityscapes or weight_count == 496306)

        batch = 4
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
