"""
    ResDrop-ResNet for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.
"""

__all__ = ['CIFARResDropResNet', 'resdropresnet20_cifar10', 'resdropresnet20_cifar100', 'resdropresnet20_svhn']

import os
import numpy as np
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block
from .resnet import ResBlock, ResBottleneck


class ResDropResUnit(HybridBlock):
    """
    ResDrop-ResNet unit with residual connection.

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
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    life_prob : float
        Residual branch life probability.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 bottleneck,
                 life_prob,
                 **kwargs):
        super(ResDropResUnit, self).__init__(**kwargs)
        self.life_prob = life_prob
        self.resize_identity = (in_channels != out_channels) or (strides != 1)
        body_class = ResBottleneck if bottleneck else ResBlock

        with self.name_scope():
            self.body = body_class(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        if mx.autograd.is_training():
            b = np.random.binomial(n=1, p=self.life_prob)
            x = float(b) / self.life_prob * x
        x = x + identity
        x = self.activ(x)
        return x


class CIFARResDropResNet(HybridBlock):
    """
    ResDrop-ResNet model for CIFAR from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    life_probs : list of float
        Residual branch life probability for each unit.
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
                 bottleneck,
                 life_probs,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARResDropResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            k = 0
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(ResDropResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            life_prob=life_probs[k]))
                        in_channels = out_channels
                        k += 1
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=8,
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


def get_resdropresnet_cifar(classes,
                            blocks,
                            bottleneck,
                            model_name=None,
                            pretrained=False,
                            ctx=cpu(),
                            root=os.path.join("~", ".mxnet", "models"),
                            **kwargs):
    """
    Create ResDrop-ResNet model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert (classes in [10, 100])

    if bottleneck:
        assert ((blocks - 2) % 9 == 0)
        layers = [(blocks - 2) // 9] * 3
    else:
        assert ((blocks - 2) % 6 == 0)
        layers = [(blocks - 2) // 6] * 3

    init_block_channels = 16
    channels_per_layers = [16, 32, 64]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if bottleneck:
        channels = [[cij * 4 for cij in ci] for ci in channels]

    total_layers = sum(layers)
    final_death_prob = 0.5
    life_probs = [1.0 - float(i + 1) / float(total_layers) * final_death_prob for i in range(total_layers)]

    net = CIFARResDropResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        life_probs=life_probs,
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


def resdropresnet20_cifar10(classes=10, **kwargs):
    """
    ResDrop-ResNet-20 model for CIFAR-10 from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

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
    return get_resdropresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resdropresnet20_cifar10",
                                   **kwargs)


def resdropresnet20_cifar100(classes=100, **kwargs):
    """
    ResDrop-ResNet-20 model for CIFAR-100 from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

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
    return get_resdropresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resdropresnet20_cifar100",
                                   **kwargs)


def resdropresnet20_svhn(classes=10, **kwargs):
    """
    ResDrop-ResNet-20 model for SVHN from 'Deep Networks with Stochastic Depth,' https://arxiv.org/abs/1603.09382.

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
    return get_resdropresnet_cifar(classes=classes, blocks=20, bottleneck=False, model_name="resdropresnet20_svhn",
                                   **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (resdropresnet20_cifar10, 10),
        (resdropresnet20_cifar100, 100),
        (resdropresnet20_svhn, 10),
    ]

    for model, classes in models:

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
        assert (model != resdropresnet20_cifar10 or weight_count == 272474)
        assert (model != resdropresnet20_cifar100 or weight_count == 278324)
        assert (model != resdropresnet20_svhn or weight_count == 272474)

        x = mx.nd.zeros((14, 3, 32, 32), ctx=ctx)
        y = net(x)
        # with mx.autograd.record():
        #     y = net(x)
        #     y.backward()
        assert (y.shape == (14, classes))


if __name__ == "__main__":
    _test()
