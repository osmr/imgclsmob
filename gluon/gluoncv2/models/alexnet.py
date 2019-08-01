"""
    AlexNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.
"""

__all__ = ['AlexNet', 'alexnet', 'alexnetb']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ConvBlock


class AlexConv(ConvBlock):
    """
    AlexNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    use_lrn : bool
        Whether to use LRN layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_lrn,
                 **kwargs):
        super(AlexConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            use_bn=False,
            **kwargs)
        self.use_lrn = use_lrn

    def hybrid_forward(self, F, x):
        x = super(AlexConv, self).hybrid_forward(F, x)
        if self.use_lrn:
            x = F.LRN(x, nsize=5)
        return x


class AlexDense(HybridBlock):
    """
    AlexNet specific dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(AlexDense, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Dense(
                units=out_channels,
                in_units=in_channels)
            self.activ = nn.Activation("relu")
            self.dropout = nn.Dropout(rate=0.5)

    def hybrid_forward(self, F, x):
        x = self.fc(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


class AlexOutputBlock(HybridBlock):
    """
    AlexNet specific output block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 classes,
                 **kwargs):
        super(AlexOutputBlock, self).__init__(**kwargs)
        mid_channels = 4096

        with self.name_scope():
            self.fc1 = AlexDense(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.fc2 = AlexDense(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.fc3 = nn.Dense(
                units=classes,
                in_units=mid_channels)

    def hybrid_forward(self, F, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNet(HybridBlock):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    kernel_sizes : list of list of int
        Convolution window sizes for each unit.
    strides : list of list of int or tuple/list of 2 int
        Strides of the convolution for each unit.
    paddings : list of list of int or tuple/list of 2 int
        Padding value for convolution layer for each unit.
    use_lrn : bool
        Whether to use LRN layer.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 use_lrn,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                use_lrn_i = use_lrn and (i in [0, 1])
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(AlexConv(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_sizes[i][j],
                            strides=strides[i][j],
                            padding=paddings[i][j],
                            use_lrn=use_lrn_i))
                        in_channels = out_channels
                    stage.add(nn.MaxPool2D(
                        pool_size=3,
                        strides=2,
                        padding=0,
                        ceil_mode=True))
                self.features.add(stage)

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            in_channels = in_channels * 6 * 6
            self.output.add(AlexOutputBlock(
                in_channels=in_channels,
                classes=classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_alexnet(version="a",
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create AlexNet model with specific parameters.

    Parameters:
    ----------
    version : str, default 'a'
        Version of AlexNet ('a' or 'b').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if version == "a":
        channels = [[96], [256], [384, 384, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[0], [2], [1, 1, 1]]
        use_lrn = True
    elif version == "b":
        channels = [[64], [192], [384, 256, 256]]
        kernel_sizes = [[11], [5], [3, 3, 3]]
        strides = [[4], [1], [1, 1, 1]]
        paddings = [[2], [2], [1, 1, 1]]
        use_lrn = False
    else:
        raise ValueError("Unsupported AlexNet version {}".format(version))

    net = AlexNet(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        use_lrn=use_lrn,
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


def alexnet(**kwargs):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(model_name="alexnet", **kwargs)


def alexnetb(**kwargs):
    """
    AlexNet-b model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997. Non-standard version.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_alexnet(version="b", model_name="alexnetb", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        alexnet,
        alexnetb,
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
        assert (model != alexnet or weight_count == 62378344)
        assert (model != alexnetb or weight_count == 61100840)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
