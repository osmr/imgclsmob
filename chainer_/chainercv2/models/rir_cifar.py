"""
    RiR for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Resnet in Resnet: Generalizing Residual Architectures,' https://arxiv.org/abs/1603.08029.
"""

__all__ = ['CIFARRiR', 'rir_cifar10', 'rir_cifar100', 'rir_svhn', 'RiRFinalBlock']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, DualPathSequential, SimpleSequential


class PostActivation(Chain):
    """
    Pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PostActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            self.activ = F.relu

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class RiRUnit(Chain):
    """
    RiR unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(RiRUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.res_pass_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            self.trans_pass_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            self.res_cross_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            self.trans_cross_conv = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            self.res_postactiv = PostActivation(in_channels=out_channels)
            self.trans_postactiv = PostActivation(in_channels=out_channels)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)

    def __call__(self, x_res, x_trans):
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


class RiRInitBlock(Chain):
    """
    RiR initial block.

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
        super(RiRInitBlock, self).__init__()
        with self.init_scope():
            self.res_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels)
            self.trans_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x, _):
        x_res = self.res_conv(x)
        x_trans = self.trans_conv(x)
        return x_res, x_trans


class RiRFinalBlock(Chain):
    """
    RiR final block.
    """
    def __init__(self):
        super(RiRFinalBlock, self).__init__()

    def __call__(self, x_res, x_trans):
        x = F.concat((x_res, x_trans), axis=1)
        return x, None


class CIFARRiR(Chain):
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
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARRiR, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=0,
                last_ordinals=0)
            with self.features.init_scope():
                setattr(self.features, "init_block", RiRInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = DualPathSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), RiRUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", RiRFinalBlock())
                in_channels = final_block_channels

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "final_conv", conv1x1_block(
                    in_channels=in_channels,
                    out_channels=classes,
                    activation=None))
                setattr(self.output, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))
                setattr(self.output, "final_flatten", partial(
                    F.reshape,
                    shape=(-1, classes)))

    def __call__(self, x):
        x = self.features(x, x)
        x = self.output(x)
        return x


def get_rir_cifar(classes,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_rir_cifar(classes=classes, model_name="rir_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (rir_cifar10, 10),
        (rir_cifar100, 100),
        (rir_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != rir_cifar10 or weight_count == 9492980)
        assert (model != rir_cifar100 or weight_count == 9527720)
        assert (model != rir_svhn or weight_count == 9492980)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
