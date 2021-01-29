"""
    CGNet for image segmentation, implemented in Gluon.
    Original paper: 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.
"""

__all__ = ['CGNet', 'cgnet_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock, Concurrent,\
    DualPathSequential, InterpolationBlock, PReLU2


class CGBlock(HybridBlock):
    """
    CGNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilation : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    down : bool
        Whether to downsample.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 se_reduction,
                 down,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(CGBlock, self).__init__(**kwargs)
        self.down = down
        if self.down:
            mid1_channels = out_channels
            mid2_channels = 2 * out_channels
        else:
            mid1_channels = out_channels // 2
            mid2_channels = out_channels

        with self.name_scope():
            if self.down:
                self.conv1 = conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=2,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off,
                    activation=(lambda: PReLU2(out_channels)))
            else:
                self.conv1 = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid1_channels,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off,
                    activation=(lambda: PReLU2(mid1_channels)))

            self.branches = Concurrent()
            self.branches.add(depthwise_conv3x3(channels=mid1_channels))
            self.branches.add(depthwise_conv3x3(
                channels=mid1_channels,
                padding=dilation,
                dilation=dilation))

            self.norm_activ = NormActivation(
                in_channels=mid2_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(mid2_channels)))

            if self.down:
                self.conv2 = conv1x1(
                    in_channels=mid2_channels,
                    out_channels=out_channels)

            self.se = SEBlock(
                channels=out_channels,
                reduction=se_reduction,
                use_conv=False)

    def hybrid_forward(self, F, x):
        if not self.down:
            identity = x
        x = self.conv1(x)
        x = self.branches(x)
        x = self.norm_activ(x)
        if self.down:
            x = self.conv2(x)
        x = self.se(x)
        if not self.down:
            x = x + identity
        return x


class CGUnit(HybridBlock):
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
    dilation : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(CGUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.down = CGBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                dilation=dilation,
                se_reduction=se_reduction,
                down=True,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.blocks = nn.HybridSequential(prefix="")
            for i in range(layers - 1):
                self.blocks.add(CGBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    dilation=dilation,
                    se_reduction=se_reduction,
                    down=False,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))

    def hybrid_forward(self, F, x):
        x = self.down(x)
        y = self.blocks(x)
        x = F.concat(y, x, dim=1)  # NB: This differs from the original implementation.
        return x


class CGStage(HybridBlock):
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
    dilation : int
        Dilation for blocks.
    se_reduction : int
        SE-block reduction value for blocks.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(CGStage, self).__init__(**kwargs)
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        with self.name_scope():
            if self.use_x:
                self.x_down = nn.AvgPool2D(
                    pool_size=3,
                    strides=2,
                    padding=1)

            if self.use_unit:
                self.unit = CGUnit(
                    in_channels=y_in_channels,
                    out_channels=(y_out_channels - x_channels),
                    layers=layers,
                    dilation=dilation,
                    se_reduction=se_reduction,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)

            self.norm_activ = NormActivation(
                in_channels=y_out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(y_out_channels)))

    def hybrid_forward(self, F, y, x=None):
        if self.use_unit:
            y = self.unit(y)
        if self.use_x:
            x = self.x_down(x)
            y = F.concat(y, x, dim=1)
        y = self.norm_activ(y)
        return y, x


class CGInitBlock(HybridBlock):
    """
    CGNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(CGInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))
            self.conv3 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(out_channels)))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CGNet(HybridBlock):
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
    dilations : list of int
        Dilations for each unit.
    se_reductions : list of int
        SE-block reduction value for each unit.
    cut_x : list of int
        Whether to concatenate with x-branch for each unit.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
                 dilations,
                 se_reductions,
                 cut_x,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
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

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=0)
            self.features.add(CGInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            y_in_channels = init_block_channels

            for i, (layers_i, y_out_channels) in enumerate(zip(layers, channels)):
                self.features.add(CGStage(
                    x_channels=in_channels if cut_x[i] == 1 else 0,
                    y_in_channels=y_in_channels,
                    y_out_channels=y_out_channels,
                    layers=layers_i,
                    dilation=dilations[i],
                    se_reduction=se_reductions[i],
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                y_in_channels = y_out_channels

            self.classifier = conv1x1(
                in_channels=y_in_channels,
                out_channels=classes)

            self.up = InterpolationBlock(scale_factor=8)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        y = self.features(x, x)
        y = self.classifier(y)
        y = self.up(y, in_size)
        return y


def get_cgnet(model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create CGNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [0, 3, 21]
    channels = [35, 131, 256]
    dilations = [0, 2, 4]
    se_reductions = [0, 8, 16]
    cut_x = [1, 1, 0]
    bn_epsilon = 1e-3

    net = CGNet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        se_reductions=se_reductions,
        cut_x=cut_x,
        bn_epsilon=bn_epsilon,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cgnet(classes=classes, model_name="cgnet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def _test():
    import mxnet as mx

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        cgnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cgnet_cityscapes or weight_count == 496306)

        batch = 4
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
