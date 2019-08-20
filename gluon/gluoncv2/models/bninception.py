"""
    BN-Inception for ImageNet-1K, implemented in Gluon.
    Original paper: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,'
    https://arxiv.org/abs/1502.03167.
"""

__all__ = ['BNInception', 'bninception']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv1x1_block, conv3x3_block, conv7x7_block


class Inception3x3Branch(HybridBlock):
    """
    BN-Inception 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 strides=1,
                 use_bias=True,
                 use_bn=True,
                 bn_use_global_stats=False,
                 **kwargs):
        super(Inception3x3Branch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=strides,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InceptionDouble3x3Branch(HybridBlock):
    """
    BN-Inception double 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 strides=1,
                 use_bias=True,
                 use_bn=True,
                 bn_use_global_stats=False,
                 **kwargs):
        super(InceptionDouble3x3Branch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=strides,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class InceptionPoolBranch(HybridBlock):
    """
    BN-Inception avg-pool branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    avg_pool : bool
        Whether use average pooling or max pooling.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 avg_pool,
                 use_bias,
                 use_bn,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionPoolBranch, self).__init__(**kwargs)
        with self.name_scope():
            if avg_pool:
                self.pool = nn.AvgPool2D(
                    pool_size=3,
                    strides=1,
                    padding=1,
                    ceil_mode=True,
                    count_include_pad=True)
            else:
                self.pool = nn.MaxPool2D(
                    pool_size=3,
                    strides=1,
                    padding=1,
                    ceil_mode=True)
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class StemBlock(HybridBlock):
    """
    BN-Inception stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 use_bias,
                 use_bn,
                 bn_use_global_stats,
                 **kwargs):
        super(StemBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv7x7_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)
            self.pool1 = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0,
                ceil_mode=True)
            self.conv2 = Inception3x3Branch(
                in_channels=mid_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats)
            self.pool2 = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0,
                ceil_mode=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x


class InceptionBlock(HybridBlock):
    """
    BN-Inception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels_list : list of int
        Number of pre-middle channels for branches.
    mid2_channels_list : list of int
        Number of middle channels for branches.
    avg_pool : bool
        Whether use average pooling or max pooling.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels_list,
                 mid2_channels_list,
                 avg_pool,
                 use_bias,
                 use_bn,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        assert (len(mid1_channels_list) == 2)
        assert (len(mid2_channels_list) == 4)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=mid2_channels_list[0],
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(Inception3x3Branch(
                in_channels=in_channels,
                out_channels=mid2_channels_list[1],
                mid_channels=mid1_channels_list[0],
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(InceptionDouble3x3Branch(
                in_channels=in_channels,
                out_channels=mid2_channels_list[2],
                mid_channels=mid1_channels_list[1],
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(InceptionPoolBranch(
                in_channels=in_channels,
                out_channels=mid2_channels_list[3],
                avg_pool=avg_pool,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class ReductionBlock(HybridBlock):
    """
    BN-Inception reduction block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels_list : list of int
        Number of pre-middle channels for branches.
    mid2_channels_list : list of int
        Number of middle channels for branches.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels_list,
                 mid2_channels_list,
                 use_bias,
                 use_bn,
                 bn_use_global_stats,
                 **kwargs):
        super(ReductionBlock, self).__init__(**kwargs)
        assert (len(mid1_channels_list) == 2)
        assert (len(mid2_channels_list) == 4)

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Inception3x3Branch(
                in_channels=in_channels,
                out_channels=mid2_channels_list[1],
                mid_channels=mid1_channels_list[0],
                strides=2,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(InceptionDouble3x3Branch(
                in_channels=in_channels,
                out_channels=mid2_channels_list[2],
                mid_channels=mid1_channels_list[1],
                strides=2,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0,
                ceil_mode=True))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class BNInception(HybridBlock):
    """
    BN-Inception model from 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift,' https://arxiv.org/abs/1502.03167.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels_list : list of int
        Number of output channels for the initial unit.
    mid1_channels_list : list of list of list of int
        Number of pre-middle channels for each unit.
    mid2_channels_list : list of list of list of int
        Number of middle channels for each unit.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
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
                 init_block_channels_list,
                 mid1_channels_list,
                 mid2_channels_list,
                 use_bias=True,
                 use_bn=True,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(BNInception, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(StemBlock(
                in_channels=in_channels,
                out_channels=init_block_channels_list[1],
                mid_channels=init_block_channels_list[0],
                use_bias=use_bias,
                use_bn=use_bn,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels_list[-1]
            for i, channels_per_stage in enumerate(channels):
                mid1_channels_list_i = mid1_channels_list[i]
                mid2_channels_list_i = mid2_channels_list[i]
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        if (j == 0) and (i != 0):
                            stage.add(ReductionBlock(
                                in_channels=in_channels,
                                mid1_channels_list=mid1_channels_list_i[j],
                                mid2_channels_list=mid2_channels_list_i[j],
                                use_bias=use_bias,
                                use_bn=use_bn,
                                bn_use_global_stats=bn_use_global_stats))
                        else:
                            avg_pool = (i != len(channels) - 1) or (j != len(channels_per_stage) - 1)
                            stage.add(InceptionBlock(
                                in_channels=in_channels,
                                mid1_channels_list=mid1_channels_list_i[j],
                                mid2_channels_list=mid2_channels_list_i[j],
                                avg_pool=avg_pool,
                                use_bias=use_bias,
                                use_bn=use_bn,
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


def get_bninception(model_name=None,
                    pretrained=False,
                    ctx=cpu(),
                    root=os.path.join("~", ".mxnet", "models"),
                    **kwargs):
    """
    Create BN-Inception model with specific parameters.

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
    init_block_channels_list = [64, 192]
    channels = [[256, 320], [576, 576, 576, 608, 608], [1056, 1024, 1024]]
    mid1_channels_list = [
        [[64, 64],
         [64, 64]],
        [[128, 64],  # 3c
         [64, 96],  # 4a
         [96, 96],  # 4a
         [128, 128],  # 4c
         [128, 160]],  # 4d
        [[128, 192],  # 4e
         [192, 160],  # 5a
         [192, 192]],
    ]
    mid2_channels_list = [
        [[64, 64, 96, 32],
         [64, 96, 96, 64]],
        [[0, 160, 96, 0],  # 3c
         [224, 96, 128, 128],  # 4a
         [192, 128, 128, 128],  # 4b
         [160, 160, 160, 128],  # 4c
         [96, 192, 192, 128]],  # 4d
        [[0, 192, 256, 0],  # 4e
         [352, 320, 224, 128],  # 5a
         [352, 320, 224, 128]],
    ]

    net = BNInception(
        channels=channels,
        init_block_channels_list=init_block_channels_list,
        mid1_channels_list=mid1_channels_list,
        mid2_channels_list=mid2_channels_list,
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


def bninception(**kwargs):
    """
    BN-Inception model from 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift,' https://arxiv.org/abs/1502.03167.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_bninception(model_name="bninception", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        bninception,
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
        assert (model != bninception or weight_count == 11295240)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
