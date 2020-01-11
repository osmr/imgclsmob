"""
    FBNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,'
    https://arxiv.org/abs/1812.03443.
"""

__all__ = ['FBNet', 'fbnet_cb']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SimpleSequential


class FBNetUnit(Chain):
    """
    FBNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the second convolution layer.
    bn_eps : float
        Small float added to variance in Batch norm.
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
                 stride,
                 bn_eps,
                 use_kernel3,
                 exp_factor,
                 activation="relu"):
        super(FBNetUnit, self).__init__()
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_exp_conv = True
        mid_channels = exp_factor * in_channels

        with self.init_scope():
            if self.use_exp_conv:
                self.exp_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_eps=bn_eps,
                    activation=activation)
            if use_kernel3:
                self.conv1 = dwconv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    stride=stride,
                    bn_eps=bn_eps,
                    activation=activation)
            else:
                self.conv1 = dwconv5x5_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    stride=stride,
                    bn_eps=bn_eps,
                    activation=activation)
            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_eps=bn_eps,
                activation=None)

    def __call__(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class FBNetInitBlock(Chain):
    """
    FBNet specific initial block.

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
                 bn_eps):
        super(FBNetInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                bn_eps=bn_eps)
            self.conv2 = FBNetUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                bn_eps=bn_eps,
                use_kernel3=True,
                exp_factor=1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FBNet(Chain):
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
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(FBNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", FBNetInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    bn_eps=bn_eps))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) else 1
                            use_kernel3 = kernels3[i][j] == 1
                            exp_factor = exp_factors[i][j]
                            setattr(stage, "unit{}".format(j + 1), FBNetUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bn_eps=bn_eps,
                                use_kernel3=use_kernel3,
                                exp_factor=exp_factor))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", conv1x1_block(
                    in_channels=in_channels,
                    out_channels=final_block_channels,
                    bn_eps=bn_eps))
                in_channels = final_block_channels
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


def get_fbnet(version,
              bn_eps=1e-5,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
              **kwargs):
    """
    Create FBNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('a', 'b' or 'c').
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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


def fbnet_cb(**kwargs):
    """
    FBNet-Cb model (bn_eps=1e-3) from 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural
    Architecture Search,' https://arxiv.org/abs/1812.03443.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_fbnet(version="c", bn_eps=1e-3, model_name="fbnet_cb", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        fbnet_cb,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fbnet_cb or weight_count == 5572200)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
