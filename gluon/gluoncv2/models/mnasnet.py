"""
    MnasNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' https://arxiv.org/abs/1807.11626.
"""

__all__ = ['MnasNet', 'mnasnet_b1', 'mnasnet_a1', 'mnasnet_small']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import round_channels, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SEBlock


class DwsExpSEResUnit(HybridBlock):
    """
    Depthwise separable expanded residual unit with SE-block. Here it used as MnasNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution layer.
    use_kernel3 : bool, default True
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int, default 1
        Expansion factor for each unit.
    se_factor : int, default 0
        SE reduction factor for each unit.
    use_skip : bool, default True
        Whether to use skip connection.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    activation : str, default 'relu'
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 use_kernel3=True,
                 exp_factor=1,
                 se_factor=0,
                 use_skip=True,
                 bn_use_global_stats=False,
                 activation="relu",
                 **kwargs):
        super(DwsExpSEResUnit, self).__init__(**kwargs)
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (strides == 1) and use_skip
        self.use_exp_conv = exp_factor > 1
        self.use_se = se_factor > 0
        mid_channels = exp_factor * in_channels
        dwconv_block_fn = dwconv3x3_block if use_kernel3 else dwconv5x5_block

        with self.name_scope():
            if self.use_exp_conv:
                self.exp_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            self.dw_conv = dwconv_block_fn(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=(exp_factor * se_factor),
                    round_mid=False,
                    mid_activation=activation)
            self.pw_conv = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class MnasInitBlock(HybridBlock):
    """
    MnasNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    use_skip : bool
        Whether to use skip connection in the second block.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 use_skip,
                 bn_use_global_stats=False,
                 **kwargs):
        super(MnasInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = DwsExpSEResUnit(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_skip=use_skip,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MnasFinalBlock(HybridBlock):
    """
    MnasNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    use_skip : bool
        Whether to use skip connection in the second block.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 use_skip,
                 bn_use_global_stats=False,
                 **kwargs):
        super(MnasFinalBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = DwsExpSEResUnit(
                in_channels=in_channels,
                out_channels=mid_channels,
                exp_factor=6,
                use_skip=use_skip,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MnasNet(HybridBlock):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Number of output channels for the initial unit.
    final_block_channels : list of 2 int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
    init_block_use_skip : bool
        Whether to use skip connection in the initial unit.
    final_block_use_skip : bool
        Whether to use skip connection in the final block of the feature extractor.
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
                 kernels3,
                 exp_factors,
                 se_factors,
                 init_block_use_skip,
                 final_block_use_skip,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(MnasNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(MnasInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels[1],
                mid_channels=init_block_channels[0],
                use_skip=init_block_use_skip,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels[1]
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) else 1
                        use_kernel3 = kernels3[i][j] == 1
                        exp_factor = exp_factors[i][j]
                        se_factor = se_factors[i][j]
                        stage.add(DwsExpSEResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            use_kernel3=use_kernel3,
                            exp_factor=exp_factor,
                            se_factor=se_factor,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(MnasFinalBlock(
                in_channels=in_channels,
                out_channels=final_block_channels[1],
                mid_channels=final_block_channels[0],
                use_skip=final_block_use_skip,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels[1]
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


def get_mnasnet(version,
                width_scale,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create MnasNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('b1', 'a1' or 'small').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if version == "b1":
        init_block_channels = [32, 16]
        final_block_channels = [320, 1280]
        channels = [[24, 24, 24], [40, 40, 40], [80, 80, 80, 96, 96], [192, 192, 192, 192]]
        kernels3 = [[1, 1, 1], [0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0]]
        exp_factors = [[3, 3, 3], [3, 3, 3], [6, 6, 6, 6, 6], [6, 6, 6, 6]]
        se_factors = [[0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0]]
        init_block_use_skip = False
        final_block_use_skip = False
    elif version == "a1":
        init_block_channels = [32, 16]
        final_block_channels = [320, 1280]
        channels = [[24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        kernels3 = [[1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        exp_factors = [[6, 6], [3, 3, 3], [6, 6, 6, 6, 6, 6], [6, 6, 6]]
        se_factors = [[0, 0], [4, 4, 4], [0, 0, 0, 0, 4, 4], [4, 4, 4]]
        init_block_use_skip = False
        final_block_use_skip = True
    elif version == "small":
        init_block_channels = [8, 8]
        final_block_channels = [144, 1280]
        channels = [[16], [16, 16], [32, 32, 32, 32, 32, 32, 32], [88, 88, 88]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0]]
        exp_factors = [[3], [6, 6], [6, 6, 6, 6, 6, 6, 6], [6, 6, 6]]
        se_factors = [[0], [0, 0], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4]]
        init_block_use_skip = True
        final_block_use_skip = True
    else:
        raise ValueError("Unsupported MnasNet version {}".format(version))

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale)

    net = MnasNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        se_factors=se_factors,
        init_block_use_skip=init_block_use_skip,
        final_block_use_skip=final_block_use_skip,
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


def mnasnet_b1(**kwargs):
    """
    MnasNet-B1 model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(version="b1", width_scale=1.0, model_name="mnasnet_b1", **kwargs)


def mnasnet_a1(**kwargs):
    """
    MnasNet-A1 model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(version="a1", width_scale=1.0, model_name="mnasnet_a1", **kwargs)


def mnasnet_small(**kwargs):
    """
    MnasNet-Small model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(version="small", width_scale=1.0, model_name="mnasnet_small", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        mnasnet_b1,
        mnasnet_a1,
        mnasnet_small,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mnasnet_b1 or weight_count == 4383312)
        assert (model != mnasnet_a1 or weight_count == 3887038)
        assert (model != mnasnet_small or weight_count == 2030264)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
