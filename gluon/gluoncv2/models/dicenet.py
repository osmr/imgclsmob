"""
    DiCENet for ImageNet-1K, implemented in Gluon.
    Original paper: 'DiCENet: Dimension-wise Convolutions for Efficient Networks,' https://arxiv.org/abs/1906.03516.
"""

__all__ = ['DiceNet', 'dicenet_wd5', 'dicenet_wd2', 'dicenet_w3d4', 'dicenet_w1', 'dicenet_w5d4', 'dicenet_w3d2',
           'dicenet_w7d8', 'dicenet_w2']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, NormActivation, ChannelShuffle, Concurrent, PReLU2


class SpatialDiceBranch(HybridBlock):
    """
    Spatial element of DiCE block for selected dimension.

    Parameters:
    ----------
    sp_size : int
        Desired size for selected spatial dimension.
    is_height : bool
        Is selected dimension height.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    """
    def __init__(self,
                 sp_size,
                 is_height,
                 fixed_size,
                 **kwargs):
        super(SpatialDiceBranch, self).__init__(**kwargs)
        self.is_height = is_height
        self.fixed_size = fixed_size
        self.index = 2 if is_height else 3
        self.base_sp_size = sp_size

        with self.name_scope():
            self.conv = conv3x3(
                in_channels=self.base_sp_size,
                out_channels=self.base_sp_size,
                groups=self.base_sp_size)

    def hybrid_forward(self, F, x):
        if not self.fixed_size:
            height, width = x.shape[2:]
            if self.is_height:
                real_sp_size = height
                real_in_size = (real_sp_size, width)
                base_in_size = (self.base_sp_size, width)
            else:
                real_sp_size = width
                real_in_size = (height, real_sp_size)
                base_in_size = (height, self.base_sp_size)
            if real_sp_size != self.base_sp_size:
                if real_sp_size < self.base_sp_size:
                    x = F.contrib.BilinearResize2D(x, height=base_in_size[0], width=base_in_size[1])
                else:
                    x = F.contrib.AdaptiveAvgPooling2D(x, output_size=base_in_size)

        x = x.swapaxes(1, self.index)
        x = self.conv(x)
        x = x.swapaxes(1, self.index)

        if not self.fixed_size:
            changed_sp_size = x.shape[self.index]
            if real_sp_size != changed_sp_size:
                if changed_sp_size < real_sp_size:
                    x = F.contrib.BilinearResize2D(x, height=real_in_size[0], width=real_in_size[1])
                else:
                    x = F.contrib.AdaptiveAvgPooling2D(x, output_size=real_in_size)

        return x


class DiceBaseBlock(HybridBlock):
    """
    Base part of DiCE block (without attention).

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 in_size,
                 fixed_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DiceBaseBlock, self).__init__(**kwargs)
        mid_channels = 3 * channels

        with self.name_scope():
            self.convs = Concurrent()
            self.convs.add(conv3x3(
                in_channels=channels,
                out_channels=channels,
                groups=channels))
            self.convs.add(SpatialDiceBranch(
                sp_size=in_size[0],
                is_height=True,
                fixed_size=fixed_size))
            self.convs.add(SpatialDiceBranch(
                sp_size=in_size[1],
                is_height=False,
                fixed_size=fixed_size))

            self.norm_activ = NormActivation(
                in_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=mid_channels)))
            self.shuffle = ChannelShuffle(
                channels=mid_channels,
                groups=3)
            self.squeeze_conv = conv1x1_block(
                in_channels=mid_channels,
                out_channels=channels,
                groups=channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=channels)))

    def hybrid_forward(self, F, x):
        x = self.convs(x)
        x = self.norm_activ(x)
        x = self.shuffle(x)
        x = self.squeeze_conv(x)
        return x


class DiceAttBlock(HybridBlock):
    """
    Pure attention part of DiCE block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int, default 4
        Squeeze reduction value.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=4,
                 **kwargs):
        super(DiceAttBlock, self).__init__(**kwargs)
        mid_channels = in_channels // reduction

        with self.name_scope():
            self.pool = nn.GlobalAvgPool2D()
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=False)
            self.activ = nn.Activation("relu")
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=False)
            self.sigmoid = nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return w


class DiceBlock(HybridBlock):
    """
    DiCE block (volume-wise separable convolutions).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 fixed_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DiceBlock, self).__init__(**kwargs)
        proj_groups = math.gcd(in_channels, out_channels)

        with self.name_scope():
            self.base_block = DiceBaseBlock(
                channels=in_channels,
                in_size=in_size,
                fixed_size=fixed_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.att = DiceAttBlock(
                in_channels=in_channels,
                out_channels=out_channels)
            # assert (in_channels == out_channels)
            self.proj_conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=proj_groups,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=out_channels)))

    def hybrid_forward(self, F, x):
        x = self.base_block(x)
        w = self.att(x)
        x = self.proj_conv(x)
        x = F.broadcast_mul(x, w)
        return x


class StridedDiceLeftBranch(HybridBlock):
    """
    Left branch of the strided DiCE block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(StridedDiceLeftBranch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=channels,
                out_channels=channels,
                strides=2,
                groups=channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=channels)))
            self.conv2 = conv1x1_block(
                in_channels=channels,
                out_channels=channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=channels)))

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StridedDiceRightBranch(HybridBlock):
    """
    Right branch of the strided DiCE block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 channels,
                 in_size,
                 fixed_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(StridedDiceRightBranch, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=3,
                strides=2,
                padding=1)
            self.dice = DiceBlock(
                in_channels=channels,
                out_channels=channels,
                in_size=(in_size[0] // 2, in_size[1] // 2),
                fixed_size=fixed_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv = conv1x1_block(
                in_channels=channels,
                out_channels=channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=channels)))

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.dice(x)
        x = self.conv(x)
        return x


class StridedDiceBlock(HybridBlock):
    """
    Strided DiCE block (strided volume-wise separable convolutions).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 fixed_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(StridedDiceBlock, self).__init__(**kwargs)
        assert (out_channels == 2 * in_channels)

        with self.name_scope():
            self.branches = Concurrent()
            self.branches.add(StridedDiceLeftBranch(
                channels=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            self.branches.add(StridedDiceRightBranch(
                channels=in_channels,
                in_size=in_size,
                fixed_size=fixed_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            self.shuffle = ChannelShuffle(
                channels=out_channels,
                groups=2)

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        x = self.shuffle(x)
        return x


class ShuffledDiceRightBranch(HybridBlock):
    """
    Right branch of the shuffled DiCE block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 fixed_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(ShuffledDiceRightBranch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=out_channels)))
            self.dice = DiceBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                in_size=in_size,
                fixed_size=fixed_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.dice(x)
        return x


class ShuffledDiceBlock(HybridBlock):
    """
    Shuffled DiCE block (shuffled volume-wise separable convolutions).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    fixed_size : bool
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 fixed_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(ShuffledDiceBlock, self).__init__(**kwargs)
        self.left_part = in_channels - in_channels // 2
        right_in_channels = in_channels - self.left_part
        right_out_channels = out_channels - self.left_part

        with self.name_scope():
            self.right_branch = ShuffledDiceRightBranch(
                in_channels=right_in_channels,
                out_channels=right_out_channels,
                in_size=in_size,
                fixed_size=fixed_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.shuffle = ChannelShuffle(
                channels=(2 * right_out_channels),
                groups=2)

    def hybrid_forward(self, F, x):
        x1, x2 = F.split(x, axis=1, num_outputs=2)
        x2 = self.right_branch(x2)
        x = F.concat(x1, x2, dim=1)
        x = self.shuffle(x)
        return x


class DiceInitBlock(HybridBlock):
    """
    DiceNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DiceInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=(lambda: PReLU2(in_channels=out_channels)))
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DiceClassifier(HybridBlock):
    """
    DiceNet specific classifier block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    classes : int, default 1000
        Number of classification classes.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 classes,
                 dropout_rate,
                 **kwargs):
        super(DiceClassifier, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                groups=4)
            self.dropout = nn.Dropout(rate=dropout_rate)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=classes,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class DiceNet(HybridBlock):
    """
    DiCENet model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,' https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    dropout_rate : float
        Parameter of Dropout layer in classifier. Faction of the input units to drop.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
                 classifier_mid_channels,
                 dropout_rate,
                 fixed_size=True,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(DiceNet, self).__init__(**kwargs)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(DiceInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = init_block_channels
            in_size = (in_size[0] // 4, in_size[1] // 4)
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        unit_class = StridedDiceBlock if j == 0 else ShuffledDiceBlock
                        stage.add(unit_class(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            in_size=in_size,
                            fixed_size=fixed_size,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off))
                        in_channels = out_channels
                        in_size = (in_size[0] // 2, in_size[1] // 2) if j == 0 else in_size
                self.features.add(stage)
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix="")
            self.output.add(DiceClassifier(
                in_channels=in_channels,
                mid_channels=classifier_mid_channels,
                classes=classes,
                dropout_rate=dropout_rate))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_dicenet(width_scale,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create DiCENet model with specific parameters.

    Parameters:
    ----------
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
    channels_per_layers_dict = {
        0.2: [32, 64, 128],
        0.5: [48, 96, 192],
        0.75: [86, 172, 344],
        1.0: [116, 232, 464],
        1.25: [144, 288, 576],
        1.5: [176, 352, 704],
        1.75: [210, 420, 840],
        2.0: [244, 488, 976],
        2.4: [278, 556, 1112],
    }

    if width_scale not in channels_per_layers_dict.keys():
        raise ValueError("Unsupported DiceNet with width scale: {}".format(width_scale))

    channels_per_layers = channels_per_layers_dict[width_scale]
    layers = [3, 7, 3]

    if width_scale > 0.2:
        init_block_channels = 24
    else:
        init_block_channels = 16

    channels = [[ci] * li for i, (ci, li) in enumerate(zip(channels_per_layers, layers))]
    for i in range(len(channels)):
        pred_channels = channels[i - 1][-1] if i != 0 else init_block_channels
        channels[i] = [pred_channels * 2] + channels[i]

    if width_scale > 2.0:
        classifier_mid_channels = 1280
    else:
        classifier_mid_channels = 1024

    if width_scale > 1.0:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.1

    net = DiceNet(
        channels=channels,
        init_block_channels=init_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        dropout_rate=dropout_rate,
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


def dicenet_wd5(**kwargs):
    """
    DiCENet x0.2 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=0.2, model_name="dicenet_wd5", **kwargs)


def dicenet_wd2(**kwargs):
    """
    DiCENet x0.5 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=0.5, model_name="dicenet_wd2", **kwargs)


def dicenet_w3d4(**kwargs):
    """
    DiCENet x0.75 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=0.75, model_name="dicenet_w3d4", **kwargs)


def dicenet_w1(**kwargs):
    """
    DiCENet x1.0 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.0, model_name="dicenet_w1", **kwargs)


def dicenet_w5d4(**kwargs):
    """
    DiCENet x1.25 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.25, model_name="dicenet_w5d4", **kwargs)


def dicenet_w3d2(**kwargs):
    """
    DiCENet x1.5 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.5, model_name="dicenet_w3d2", **kwargs)


def dicenet_w7d8(**kwargs):
    """
    DiCENet x1.75 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=1.75, model_name="dicenet_w7d8", **kwargs)


def dicenet_w2(**kwargs):
    """
    DiCENet x2.0 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=2.0, model_name="dicenet_w2", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def _test():
    import mxnet as mx

    pretrained = False
    fixed_size = True

    models = [
        dicenet_wd5,
        dicenet_wd2,
        dicenet_w3d4,
        dicenet_w1,
        dicenet_w5d4,
        dicenet_w3d2,
        dicenet_w7d8,
        dicenet_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained, fixed_size=fixed_size)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dicenet_wd5 or weight_count == 1130704)
        assert (model != dicenet_wd2 or weight_count == 1214120)
        assert (model != dicenet_w3d4 or weight_count == 1495676)
        assert (model != dicenet_w1 or weight_count == 1805604)
        assert (model != dicenet_w5d4 or weight_count == 2162888)
        assert (model != dicenet_w3d2 or weight_count == 2652200)
        assert (model != dicenet_w7d8 or weight_count == 3264932)
        assert (model != dicenet_w2 or weight_count == 3979044)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
