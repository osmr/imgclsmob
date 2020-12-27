
__all__ = ['FastSCNN', 'fastscnn_cityscapes']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.contrib.nn import Identity
from mxnet.gluon import nn
from common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwsconv3x3_block, Concurrent


class LinearBottleneck(HybridBlock):
    """
    So-called 'Linear Bottleneck' layer. It is used as a MobileNetV2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    expansion : bool, default True
        Whether do expansion of channels.
    remove_exp_conv : bool, default False
        Whether to remove expansion convolution.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 expansion=True,
                 remove_exp_conv=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (strides == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        self.use_exp_conv = (expansion or (not remove_exp_conv))

        with self.name_scope():
            if self.use_exp_conv:
                self.conv1 = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            self.conv2 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class Steam(HybridBlock):
    def __init__(self,
                 in_channels,
                 channels):
        super(Steam, self).__init__()
        assert (len(channels) == 3)

        with self.name_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=channels[0],
                strides=2,
                padding=0)
            self.conv2 = dwsconv3x3_block(
                in_channels=channels[0],
                out_channels=channels[1],
                strides=2)
            self.conv3 = dwsconv3x3_block(
                in_channels=channels[1],
                out_channels=channels[2],
                strides=2)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class FeatureExtractor(HybridBlock):
    def __init__(self,
                 in_channels,
                 channels):
        super(FeatureExtractor, self).__init__()
        with self.name_scope():

            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != len(channels) - 1) else 1
                        stage.add(LinearBottleneck(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides))
                        in_channels = out_channels
                self.features.add(stage)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x


class PoolingBranch(HybridBlock):
    """
    Pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int or None
        Spatial size of output image for the bilinear upsampling operation.
    down_size : int
        Spatial size of downscaled image.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 down_size,
                 **kwargs):
        super(PoolingBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.down_size = down_size

        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=self.down_size)
        x = self.conv(x)
        x = F.contrib.BilinearResize2D(x, height=in_size[0], width=in_size[1])
        return x


class FastPyramidPooling(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(FastPyramidPooling, self).__init__()
        down_sizes = [1, 2, 3, 6]
        mid_channels = in_channels // 4

        with self.name_scope():
            self.branches = Concurrent()
            self.branches.add(Identity())
            for down_size in down_sizes:
                self.branches.add(PoolingBranch(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    in_size=in_size,
                    down_size=down_size))
            self.conv = conv1x1_block(
                in_channels=(in_channels * 2),
                out_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        x = self.conv(x)
        return x


class FeatureFusionModule(HybridBlock):
    def __init__(self,
                 highter_in_channels,
                 lower_in_channels,
                 out_channels,
                 height,
                 width,
                 scale_factor=4,
                 **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self._up_kwargs = {'height': height, 'width': width}

        with self.name_scope():
            self.dwconv = dwconv3x3_block(
                in_channels=lower_in_channels,
                out_channels=out_channels)
            self.conv_lower_res = conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=True,
                activation=None)
            self.conv_higher_res = conv1x1_block(
                in_channels=highter_in_channels,
                out_channels=out_channels,
                use_bias=True,
                activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, higher_res_feature, lower_res_feature):
        lower_res_feature = F.contrib.BilinearResize2D(lower_res_feature, **self._up_kwargs)

        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.activ(out)


class Head(HybridBlock):
    def __init__(self,
                 in_channels,
                 classes):
        super(Head, self).__init__()
        with self.name_scope():
            self.dsconv1 = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels)
            self.dsconv2 = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels)
            self.dp = nn.Dropout(0.1)
            self.conv = conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dp(x)
        x = self.conv(x)
        return x


class AuxHead(HybridBlock):
    def __init__(self,
                 in_channels=64,
                 mid_channels=64,
                 classes=19):
        super(AuxHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            with self.block.name_scope():
                self.block.add(conv3x3_block(
                    in_channels=in_channels,
                    out_channels=mid_channels))
                self.block.add(nn.Dropout(0.1))
                self.block.add(conv1x1(
                    in_channels=mid_channels,
                    out_channels=classes,
                    use_bias=True))

    def hybrid_forward(self, F, x):
        return self.block(x)


class FastSCNN(HybridBlock):
    def __init__(self,
                 classes,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(480, 480),
                 **kwargs):
        super(FastSCNN, self).__init__()

        height = in_size[0]
        width = in_size[1]
        self._up_kwargs = {'height': height, 'width': width}
        self.aux = aux

        with self.name_scope():
            steam_channels = (32, 48, 64)
            self.steam = Steam(
                in_channels=in_channels,
                channels=steam_channels)
            in_channels = steam_channels[-1]
            feature_channels = [[64, 64, 64], [96, 96, 96], [128, 128, 128]]
            self.features = FeatureExtractor(
                in_channels=in_channels,
                channels=feature_channels)
            in_channels = feature_channels[-1][-1]
            pool_out_size = (in_size[0] // 32, in_size[1] // 32) if fixed_size else None
            self.ppm = FastPyramidPooling(
                in_channels,
                in_channels,
                in_size=pool_out_size)
            self.ffm = FeatureFusionModule(
                highter_in_channels=64,
                lower_in_channels=128,
                out_channels=128,
                height=height//8,
                width=width//8,
                **kwargs)
            self.head = Head(
                in_channels=128,
                classes=classes)
            if self.aux:
                self.aux_head = AuxHead(
                    in_channels=64,
                    mid_channels=64,
                    classes=classes)

    def hybrid_forward(self, F, x):
        x = self.steam(x)
        y = self.features(x)
        y = self.ppm(y)
        y = self.ffm(x, y)
        y = self.head(y)
        y = F.contrib.BilinearResize2D(y, **self._up_kwargs)

        if self.aux:
            auxout = self.aux_head(x)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            return y, auxout
        return y


def get_fastscnn(dataset='citys', ctx=cpu(0), pretrained=False,
                 root='~/.mxnet/models', **kwargs):
    r"""Fast-SCNN: Fast Semantic Segmentation Network
    Parameters:
    ----------
    dataset : str, default cityscapes
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    acronyms = {
        'citys': 'citys',
    }
    from gluoncv.data import datasets

    model = FastSCNN(datasets[dataset].NUM_CLASS, ctx=ctx, **kwargs)
    model.classes = datasets[dataset].classes
    return model


def fastscnn_cityscapes(**kwargs):
    r"""Fast-SCNN: Fast Semantic Segmentation Network
        Parameters:
        ----------
        dataset : str, default cityscapes
        ctx : Context, default CPU
            The context in which to load the pretrained weights.
        """
    return get_fastscnn('citys', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (1024, 1024)
    aux = True
    pretrained = False
    fixed_size = False

    models = [
        (fastscnn_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size, aux=aux)

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
        assert (model != fastscnn_cityscapes or weight_count == 1176278)

        x = mx.nd.zeros((1, 3, in_size[0], in_size[1]), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
