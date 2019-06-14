"""
    DPN for ImageNet-1K, implemented in Gluon.
    Original paper: 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.
"""

__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn98', 'dpn107', 'dpn131']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, DualPathSequential


class GlobalAvgMaxPool2D(HybridBlock):
    """
    Global average+max pooling operation for spatial data.
    """
    def __init__(self,
                 **kwargs):
        super(GlobalAvgMaxPool2D, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.max_pool = nn.GlobalMaxPool2D()

    def hybrid_forward(self, F, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = 0.5 * (x_avg + x_max)
        return x


def dpn_batch_norm(channels):
    """
    DPN specific Batch normalization layer.

    Parameters:
    ----------
    channels : int
        Number of channels in input data.
    """
    return nn.BatchNorm(
        epsilon=0.001,
        in_channels=channels)


class PreActivation(HybridBlock):
    """
    DPN specific block, which performs the preactivation like in RreResNet.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels,
                 **kwargs):
        super(PreActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = dpn_batch_norm(channels=channels)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class DPNConv(HybridBlock):
    """
    DPN specific convolution block.

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
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 **kwargs):
        super(DPNConv, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = dpn_batch_norm(channels=in_channels)
            self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def dpn_conv1x1(in_channels,
                out_channels,
                strides=1):
    """
    1x1 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=1)


def dpn_conv3x3(in_channels,
                out_channels,
                strides,
                groups):
    """
    3x3 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int
        Number of groups.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=groups)


class DPNUnit(HybridBlock):
    """
    DPN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of intermediate channels.
    bw : int
        Number of residual channels.
    inc : int
        Incrementing step for channels.
    groups : int
        Number of groups in the units.
    has_proj : bool
        Whether to use projection.
    key_strides : int
        Key strides of the convolutions.
    b_case : bool, default False
        Whether to use B-case model.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 bw,
                 inc,
                 groups,
                 has_proj,
                 key_strides,
                 b_case=False,
                 **kwargs):
        super(DPNUnit, self).__init__(**kwargs)
        self.bw = bw
        self.has_proj = has_proj
        self.b_case = b_case

        with self.name_scope():
            if self.has_proj:
                self.conv_proj = dpn_conv1x1(
                    in_channels=in_channels,
                    out_channels=bw + 2 * inc,
                    strides=key_strides)

            self.conv1 = dpn_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = dpn_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=key_strides,
                groups=groups)

            if b_case:
                self.preactiv = PreActivation(channels=mid_channels)
                self.conv3a = conv1x1(
                    in_channels=mid_channels,
                    out_channels=bw)
                self.conv3b = conv1x1(
                    in_channels=mid_channels,
                    out_channels=inc)
            else:
                self.conv3 = dpn_conv1x1(
                    in_channels=mid_channels,
                    out_channels=bw + inc)

    def hybrid_forward(self, F, x1, x2=None):
        x_in = F.concat(x1, x2, dim=1) if x2 is not None else x1
        if self.has_proj:
            x_s = self.conv_proj(x_in)
            x_s1 = F.slice_axis(x_s, axis=1, begin=0, end=self.bw)
            x_s2 = F.slice_axis(x_s, axis=1, begin=self.bw, end=None)
        else:
            assert (x2 is not None)
            x_s1 = x1
            x_s2 = x2
        x_in = self.conv1(x_in)
        x_in = self.conv2(x_in)
        if self.b_case:
            x_in = self.preactiv(x_in)
            y1 = self.conv3a(x_in)
            y2 = self.conv3b(x_in)
        else:
            x_in = self.conv3(x_in)
            y1 = F.slice_axis(x_in, axis=1, begin=0, end=self.bw)
            y2 = F.slice_axis(x_in, axis=1, begin=self.bw, end=None)
        residual = x_s1 + y1
        dense = F.concat(x_s2, y2, dim=1)
        return residual, dense


class DPNInitBlock(HybridBlock):
    """
    DPN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 **kwargs):
        super(DPNInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=2,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = dpn_batch_norm(channels=out_channels)
            self.activ = nn.Activation("relu")
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class DPNFinalBlock(HybridBlock):
    """
    DPN final block, which performs the preactivation with cutting.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels,
                 **kwargs):
        super(DPNFinalBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.activ = PreActivation(channels=channels)

    def hybrid_forward(self, F, x1, x2):
        assert (x2 is not None)
        x = F.concat(x1, x2, dim=1)
        x = self.activ(x)
        return x, None


class DPN(HybridBlock):
    """
    DPN model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    init_block_kernel_size : int or tuple/list of 2 int
        Convolution window size for the initial unit.
    init_block_padding : int or tuple/list of 2 int
        Padding value for convolution layer in the initial unit.
    rs : list f int
        Number of intermediate channels for each unit.
    bws : list f int
        Number of residual channels for each unit.
    incs : list f int
        Incrementing step for channels for each unit.
    groups : int
        Number of groups in the units.
    b_case : bool
        Whether to use B-case model.
    for_training : bool
        Whether to use model for training.
    test_time_pool : bool
        Whether to use the avg-max pooling in the inference mode.
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
                 init_block_kernel_size,
                 init_block_padding,
                 rs,
                 bws,
                 incs,
                 groups,
                 b_case,
                 for_training,
                 test_time_pool,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(DPN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=0,
                prefix="")
            self.features.add(DPNInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                kernel_size=init_block_kernel_size,
                padding=init_block_padding))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(prefix="stage{}_".format(i + 1))
                r = rs[i]
                bw = bws[i]
                inc = incs[i]
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        has_proj = (j == 0)
                        key_strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(DPNUnit(
                            in_channels=in_channels,
                            mid_channels=r,
                            bw=bw,
                            inc=inc,
                            groups=groups,
                            has_proj=has_proj,
                            key_strides=key_strides,
                            b_case=b_case))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(DPNFinalBlock(channels=in_channels))

            self.output = nn.HybridSequential(prefix="")
            if for_training or not test_time_pool:
                self.output.add(nn.GlobalAvgPool2D())
                self.output.add(conv1x1(
                    in_channels=in_channels,
                    out_channels=classes,
                    use_bias=True))
                self.output.add(nn.Flatten())
            else:
                self.output.add(nn.AvgPool2D(
                    pool_size=7,
                    strides=1))
                self.output.add(conv1x1(
                    in_channels=in_channels,
                    out_channels=classes,
                    use_bias=True))
                self.output.add(GlobalAvgMaxPool2D())
                self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_dpn(num_layers,
            b_case=False,
            for_training=False,
            model_name=None,
            pretrained=False,
            ctx=cpu(),
            root=os.path.join("~", ".mxnet", "models"),
            **kwargs):
    """
    Create DPN model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    b_case : bool, default False
        Whether to use B-case model.
    for_training : bool
        Whether to use model for training.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    if num_layers == 68:
        init_block_channels = 10
        init_block_kernel_size = 3
        init_block_padding = 1
        bw_factor = 1
        k_r = 128
        groups = 32
        k_sec = (3, 4, 12, 3)
        incs = (16, 32, 32, 64)
        test_time_pool = True
    elif num_layers == 98:
        init_block_channels = 96
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 160
        groups = 40
        k_sec = (3, 6, 20, 3)
        incs = (16, 32, 32, 128)
        test_time_pool = True
    elif num_layers == 107:
        init_block_channels = 128
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 200
        groups = 50
        k_sec = (4, 8, 20, 3)
        incs = (20, 64, 64, 128)
        test_time_pool = True
    elif num_layers == 131:
        init_block_channels = 128
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 160
        groups = 40
        k_sec = (4, 8, 28, 3)
        incs = (16, 32, 32, 128)
        test_time_pool = True
    else:
        raise ValueError("Unsupported DPN version with number of layers {}".format(num_layers))

    channels = [[0] * li for li in k_sec]
    rs = [0 * li for li in k_sec]
    bws = [0 * li for li in k_sec]
    for i in range(len(k_sec)):
        rs[i] = (2 ** i) * k_r
        bws[i] = (2 ** i) * 64 * bw_factor
        inc = incs[i]
        channels[i][0] = bws[i] + 3 * inc
        for j in range(1, k_sec[i]):
            channels[i][j] = channels[i][j - 1] + inc

    net = DPN(
        channels=channels,
        init_block_channels=init_block_channels,
        init_block_kernel_size=init_block_kernel_size,
        init_block_padding=init_block_padding,
        rs=rs,
        bws=bws,
        incs=incs,
        groups=groups,
        b_case=b_case,
        for_training=for_training,
        test_time_pool=test_time_pool,
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


def dpn68(**kwargs):
    """
    DPN-68 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, b_case=False, model_name="dpn68", **kwargs)


def dpn68b(**kwargs):
    """
    DPN-68b model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, b_case=True, model_name="dpn68b", **kwargs)


def dpn98(**kwargs):
    """
    DPN-98 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=98, b_case=False, model_name="dpn98", **kwargs)


def dpn107(**kwargs):
    """
    DPN-107 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=107, b_case=False, model_name="dpn107", **kwargs)


def dpn131(**kwargs):
    """
    DPN-131 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=131, b_case=False, model_name="dpn131", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False
    for_training = False

    models = [
        dpn68,
        # dpn68b,
        dpn98,
        # dpn107,
        dpn131,
    ]

    for model in models:

        net = model(pretrained=pretrained, for_training=for_training)

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
        assert (model != dpn68 or weight_count == 12611602)
        assert (model != dpn68b or weight_count == 12611602)
        assert (model != dpn98 or weight_count == 61570728)
        assert (model != dpn107 or weight_count == 86917800)
        assert (model != dpn131 or weight_count == 79254504)

        # net.hybridize()
        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
