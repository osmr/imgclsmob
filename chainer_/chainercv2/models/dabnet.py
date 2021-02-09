"""
    DABNet for image segmentation, implemented in Chainer.
    Original paper: 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.
"""

__all__ = ['DABNet', 'dabnet_cityscapes']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv3x3, conv3x3_block, ConvBlock, NormActivation, Concurrent, InterpolationBlock,\
    DualPathSequential, SimpleSequential


class DwaConvBlock(Chain):
    """
    Depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    ksize : int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default F.relu
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: F.relu),
                 **kwargs):
        super(DwaConvBlock, self).__init__(**kwargs)
        with self.init_scope():
            self.conv1 = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                ksize=(ksize, 1),
                stride=stride,
                pad=(pad, 0),
                dilate=(dilate, 1),
                groups=channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=activation)
            self.conv2 = ConvBlock(
                in_channels=channels,
                out_channels=channels,
                ksize=(1, ksize),
                stride=stride,
                pad=(0, pad),
                dilate=(1, dilate),
                groups=channels,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_eps=bn_eps,
                activation=activation)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def dwa_conv3x3_block(channels,
                      stride=1,
                      pad=1,
                      dilate=1,
                      use_bias=False,
                      use_bn=True,
                      bn_eps=1e-5,
                      activation=(lambda: F.relu),
                      **kwargs):
    """
    3x3 version of the depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int, default 1
        Stride of the convolution.
    pad : int, default 1
        Padding value for convolution layer.
    dilate : int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return DwaConvBlock(
        channels=channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        **kwargs)


class DABBlock(Chain):
    """
    DABNet specific base block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilate : int
        Dilation value for a dilated branch in the unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 dilate,
                 bn_eps,
                 **kwargs):
        super(DABBlock, self).__init__(**kwargs)
        mid_channels = channels // 2

        with self.init_scope():
            self.norm_activ1 = NormActivation(
                in_channels=channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(channels)))
            self.conv1 = conv3x3_block(
                in_channels=channels,
                out_channels=mid_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(mid_channels)))

            self.branches = Concurrent(stack=True)
            with self.branches.init_scope():
                setattr(self.branches, "branches1", dwa_conv3x3_block(
                    channels=mid_channels,
                    bn_eps=bn_eps,
                    activation=(lambda: L.PReLU(mid_channels))))
                setattr(self.branches, "branches2", dwa_conv3x3_block(
                    channels=mid_channels,
                    pad=dilate,
                    dilate=dilate,
                    bn_eps=bn_eps,
                    activation=(lambda: L.PReLU(mid_channels))))

            self.norm_activ2 = NormActivation(
                in_channels=mid_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(mid_channels)))
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels)

    def __call__(self, x):
        identity = x

        x = self.norm_activ1(x)
        x = self.conv1(x)

        x = self.branches(x)
        x = F.sum(x, axis=1)

        x = self.norm_activ2(x)
        x = self.conv2(x)

        x = x + identity
        return x


class DownBlock(Chain):
    """
    DABNet specific downsample block for the main branch.

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
        super(DownBlock, self).__init__(**kwargs)
        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        with self.init_scope():
            self.conv = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2)
            if self.expand:
                self.pool = partial(
                    F.max_pooling_2d,
                    ksize=2,
                    stride=2,
                    cover_all=False)
            self.norm_activ = NormActivation(
                in_channels=out_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(out_channels)))

    def __call__(self, x):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = F.concat((y, z), axis=1)

        y = self.norm_activ(y)
        return y


class DABUnit(Chain):
    """
    DABNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilates : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilates,
                 bn_eps,
                 **kwargs):
        super(DABUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.init_scope():
            self.down = DownBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_eps=bn_eps)
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i, dilate in enumerate(dilates):
                    setattr(self.blocks, "block{}".format(i + 1), DABBlock(
                        channels=mid_channels,
                        dilate=dilate,
                        bn_eps=bn_eps))

    def __call__(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = F.concat((y, x), axis=1)
        return x


class DABStage(Chain):
    """
    DABNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    dilates : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 dilates,
                 bn_eps,
                 **kwargs):
        super(DABStage, self).__init__(**kwargs)
        self.use_unit = (len(dilates) > 0)

        with self.init_scope():
            self.x_down = partial(
                F.average_pooling_2d,
                ksize=3,
                stride=2,
                pad=1)

            if self.use_unit:
                self.unit = DABUnit(
                    in_channels=y_in_channels,
                    out_channels=(y_out_channels - x_channels),
                    dilates=dilates,
                    bn_eps=bn_eps)

            self.norm_activ = NormActivation(
                in_channels=y_out_channels,
                bn_eps=bn_eps,
                activation=(lambda: L.PReLU(y_out_channels)))

    def __call__(self, y, x):
        x = self.x_down(x)
        if self.use_unit:
            y = self.unit(y)
        y = F.concat((y, x), axis=1)
        y = self.norm_activ(y)
        return y, x


class DABInitBlock(Chain):
    """
    DABNet specific initial block.

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
        super(DABInitBlock, self).__init__(**kwargs)
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


class DABNet(Chain):
    """
    DABNet model from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilates : list of list of int
        Dilations for blocks.
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
                 channels,
                 init_block_channels,
                 dilates,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 **kwargs):
        super(DABNet, self).__init__(**kwargs)
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
                setattr(self.features, "init_block", DABInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    bn_eps=bn_eps))
                y_in_channels = init_block_channels

                for i, (y_out_channels, dilates_i) in enumerate(zip(channels, dilates)):
                    setattr(self.features, "stage{}".format(i + 1), DABStage(
                        x_channels=in_channels,
                        y_in_channels=y_in_channels,
                        y_out_channels=y_out_channels,
                        dilates=dilates_i,
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


def get_dabnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create DABNet model with specific parameters.

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
    channels = [35, 131, 259]
    dilates = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]
    bn_eps = 1e-3

    net = DABNet(
        channels=channels,
        init_block_channels=init_block_channels,
        dilates=dilates,
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


def dabnet_cityscapes(classes=19, **kwargs):
    """
    DABNet model for Cityscapes from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_dabnet(classes=classes, model_name="dabnet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        dabnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dabnet_cityscapes or weight_count == 756643)

        batch = 4
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        assert (y.shape == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
