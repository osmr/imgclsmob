"""
    RiR for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Resnet in Resnet: Generalizing Residual Architectures,' https://arxiv.org/abs/1603.08029.
"""

__all__ = ['CIFARRiR', 'rir_cifar10', 'rir_cifar100', 'rir_svhn', 'RiRFinalBlock']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, DualPathSequential


class PostActivation(HybridBlock):
    """
    Pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(PostActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class RiRUnit(HybridBlock):
    """
    RiR unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 **kwargs):
        super(RiRUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.res_pass_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides)
            self.trans_pass_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides)
            self.res_cross_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides)
            self.trans_cross_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides)
            self.res_postactiv = PostActivation(
                in_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.trans_postactiv = PostActivation(
                in_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides)

    def hybrid_forward(self, F, x_res, x_trans):
        if self.resize_identity:
            x_res_identity = self.identity_conv(x_res)
        else:
            x_res_identity = x_res

        y_res = self.res_cross_conv(x_res)
        y_trans = self.trans_cross_conv(x_trans)
        x_res = self.res_pass_conv(x_res)
        x_trans = self.trans_pass_conv(x_trans)

        x_res = x_res + x_res_identity + y_trans
        x_trans = x_trans + y_res

        x_res = self.res_postactiv(x_res)
        x_trans = self.trans_postactiv(x_trans)

        return x_res, x_trans


class RiRInitBlock(HybridBlock):
    """
    RiR initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(RiRInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.res_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.trans_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x, _):
        x_res = self.res_conv(x)
        x_trans = self.trans_conv(x)
        return x_res, x_trans


class RiRFinalBlock(HybridBlock):
    """
    RiR final block.
    """
    def __init__(self):
        super(RiRFinalBlock, self).__init__()

    def hybrid_forward(self, F, x_res, x_trans):
        x = F.concat(x_res, x_trans, dim=1)
        return x, None


class CIFARRiR(HybridBlock):
    """
    RiR model for CIFAR from 'Resnet in Resnet: Generalizing Residual Architectures,' https://arxiv.org/abs/1603.08029.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARRiR, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=0,
                last_ordinals=0,
                prefix="")
            self.features.add(RiRInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(prefix="stage{}_".format(i + 1))
                for j, out_channels in enumerate(channels_per_stage):
                    strides = 2 if (j == 0) and (i != 0) else 1
                    stage.add(RiRUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=strides,
                        bn_use_global_stats=bn_use_global_stats))
                    in_channels = out_channels
                self.features.add(stage)
            self.features.add(RiRFinalBlock())
            in_channels = final_block_channels

            self.output = nn.HybridSequential(prefix="")
            self.output.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=classes,
                bn_use_global_stats=bn_use_global_stats,
                activation=None))
            self.output.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x, x)
        x = self.output(x)
        return x


def get_rir_cifar(classes,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create RiR model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    channels = [[48, 48, 48, 48], [96, 96, 96, 96, 96, 96], [192, 192, 192, 192, 192, 192]]
    init_block_channels = 48
    final_block_channels = 384

    net = CIFARRiR(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classes=classes,
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


def rir_cifar10(classes=10, **kwargs):
    """
    RiR model for CIFAR-10 from 'Resnet in Resnet: Generalizing Residual Architectures,'
    https://arxiv.org/abs/1603.08029.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_rir_cifar(classes=classes, model_name="rir_cifar10", **kwargs)


def rir_cifar100(classes=100, **kwargs):
    """
    RiR model for CIFAR-100 from 'Resnet in Resnet: Generalizing Residual Architectures,'
    https://arxiv.org/abs/1603.08029.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_rir_cifar(classes=classes, model_name="rir_cifar100", **kwargs)


def rir_svhn(classes=10, **kwargs):
    """
    RiR model for SVHN from 'Resnet in Resnet: Generalizing Residual Architectures,'
    https://arxiv.org/abs/1603.08029.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_rir_cifar(classes=classes, model_name="rir_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (rir_cifar10, 10),
        (rir_cifar100, 100),
        (rir_svhn, 10),
    ]

    for model, classes in models:

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
        assert (model != rir_cifar10 or weight_count == 9492980)
        assert (model != rir_cifar100 or weight_count == 9527720)
        assert (model != rir_svhn or weight_count == 9492980)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
