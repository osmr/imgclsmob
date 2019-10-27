"""
    FBNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,'
    https://arxiv.org/abs/1812.03443.
"""

__all__ = ['FBNet', 'fbnet_cb']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block


class FBNetUnit(HybridBlock):
    """
    FBNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int
        Expansion factor for each unit.
    activation : str, default 'relu'
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_epsilon,
                 bn_use_global_stats,
                 use_kernel3,
                 exp_factor,
                 activation="relu",
                 **kwargs):
        super(FBNetUnit, self).__init__(**kwargs)
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (strides == 1)
        self.use_exp_conv = True
        mid_channels = exp_factor * in_channels

        with self.name_scope():
            if self.use_exp_conv:
                self.exp_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            if use_kernel3:
                self.conv1 = dwconv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            else:
                self.conv1 = dwconv5x5_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_epsilon=bn_epsilon,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class FBNetInitBlock(HybridBlock):
    """
    FBNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(FBNetInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = FBNetUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                use_kernel3=True,
                exp_factor=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FBNet(HybridBlock):
    """
    FBNet model from 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,'
    https://arxiv.org/abs/1812.03443.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(FBNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(FBNetInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) else 1
                        use_kernel3 = kernels3[i][j] == 1
                        exp_factor = exp_factors[i][j]
                        stage.add(FBNetUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_epsilon=bn_epsilon,
                            bn_use_global_stats=bn_use_global_stats,
                            use_kernel3=use_kernel3,
                            exp_factor=exp_factor))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels
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


def get_fbnet(version,
              bn_epsilon=1e-5,
              model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create FBNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('a', 'b' or 'c').
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if version == "c":
        init_block_channels = 16
        final_block_channels = 1984
        channels = [[24, 24, 24], [32, 32, 32, 32], [64, 64, 64, 64, 112, 112, 112, 112], [184, 184, 184, 184, 352]]
        kernels3 = [[1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        exp_factors = [[6, 1, 1], [6, 3, 6, 6], [6, 3, 6, 6, 6, 6, 6, 3], [6, 6, 6, 6, 6]]
    else:
        raise ValueError("Unsupported FBNet version {}".format(version))

    net = FBNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
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
            ctx=ctx)

    return net


def fbnet_cb(**kwargs):
    """
    FBNet-Cb model (bn_epsilon=1e-3) from 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural
    Architecture Search,' https://arxiv.org/abs/1812.03443.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_fbnet(version="c", bn_epsilon=1e-3, model_name="fbnet_cb", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        fbnet_cb,
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
        assert (model != fbnet_cb or weight_count == 5572200)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
