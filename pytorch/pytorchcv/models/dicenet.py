"""
    DiCENet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'DiCENet: Dimension-wise Convolutions for Efficient Networks,' https://arxiv.org/abs/1906.03516.
"""

__all__ = ['DiceNet', 'dicenet_wd5', 'dicenet_wd2', 'dicenet_w3d4', 'dicenet_w1', 'dicenet_w5d4', 'dicenet_w3d2',
           'dicenet_w7d8', 'dicenet_w2']

import os
import math
import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, NormActivation, ChannelShuffle, Concurrent


class SpatialDiceBranch(nn.Module):
    """
    Spatial element of DiCE block for selected dimension.

    Parameters:
    ----------
    sp_size : int
        Desired size for selected spatial dimension.
    is_height : bool
        Is selected dimension height.
    """
    def __init__(self,
                 sp_size,
                 is_height):
        super(SpatialDiceBranch, self).__init__()
        self.is_height = is_height
        self.index = 2 if is_height else 3
        self.base_sp_size = sp_size

        self.conv = conv3x3(
            in_channels=self.base_sp_size,
            out_channels=self.base_sp_size,
            groups=self.base_sp_size)

    def forward(self, x):
        height, width = x.size()[2:]
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
                x = F.interpolate(x, size=base_in_size, mode="bilinear", align_corners=True)
            else:
                x = F.adaptive_avg_pool2d(x, output_size=base_in_size)

        x = x.transpose(1, self.index).contiguous()
        x = self.conv(x)
        x = x.transpose(1, self.index).contiguous()

        changed_sp_size = x.size(self.index)
        if real_sp_size != changed_sp_size:
            if changed_sp_size < real_sp_size:
                x = F.interpolate(x, size=real_in_size, mode="bilinear", align_corners=True)
            else:
                x = F.adaptive_avg_pool2d(x, output_size=real_in_size)

        return x


class DiceBaseBlock(nn.Module):
    """
    Base part of DiCE block (without attention).

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    """
    def __init__(self,
                 channels,
                 in_size):
        super(DiceBaseBlock, self).__init__()
        mid_channels = 3 * channels

        self.convs = Concurrent()
        self.convs.add_module("ch_conv", conv3x3(
            in_channels=channels,
            out_channels=channels,
            groups=channels))
        self.convs.add_module("h_conv", SpatialDiceBranch(
            sp_size=in_size[0],
            is_height=True))
        self.convs.add_module("w_conv", SpatialDiceBranch(
            sp_size=in_size[1],
            is_height=False))

        self.norm_activ = NormActivation(
            in_channels=mid_channels,
            activation=(lambda: nn.PReLU(num_parameters=mid_channels)))
        self.shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=3)
        self.squeeze_conv = conv1x1_block(
            in_channels=mid_channels,
            out_channels=channels,
            groups=channels,
            activation=(lambda: nn.PReLU(num_parameters=channels)))

    def forward(self, x):
        x = self.convs(x)
        x = self.norm_activ(x)
        x = self.shuffle(x)
        x = self.squeeze_conv(x)
        return x


class DiceAttBlock(nn.Module):
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
                 reduction=4):
        super(DiceAttBlock, self).__init__()
        mid_channels = in_channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=False)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return w


class DiceBlock(nn.Module):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(DiceBlock, self).__init__()
        proj_groups = math.gcd(in_channels, out_channels)

        self.base_block = DiceBaseBlock(
            channels=in_channels,
            in_size=in_size)
        self.att = DiceAttBlock(
            in_channels=in_channels,
            out_channels=out_channels)
        self.proj_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=proj_groups,
            activation=(lambda: nn.PReLU(num_parameters=out_channels)))

    def forward(self, x):
        x = self.base_block(x)
        w = self.att(x)
        x = self.proj_conv(x)
        x = x * w
        return x


class StridedDiceLeftBranch(nn.Module):
    """
    Left branch of the strided DiCE block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels):
        super(StridedDiceLeftBranch, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=channels,
            out_channels=channels,
            stride=2,
            groups=channels,
            activation=(lambda: nn.PReLU(num_parameters=channels)))
        self.conv2 = conv1x1_block(
            in_channels=channels,
            out_channels=channels,
            activation=(lambda: nn.PReLU(num_parameters=channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StridedDiceRightBranch(nn.Module):
    """
    Right branch of the strided DiCE block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    """
    def __init__(self,
                 channels,
                 in_size):
        super(StridedDiceRightBranch, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            padding=1,
            stride=2)
        self.dice = DiceBlock(
            in_channels=channels,
            out_channels=channels,
            in_size=(in_size[0] // 2, in_size[1] // 2))
        self.conv = conv1x1_block(
            in_channels=channels,
            out_channels=channels,
            activation=(lambda: nn.PReLU(num_parameters=channels)))

    def forward(self, x):
        x = self.pool(x)
        x = self.dice(x)
        x = self.conv(x)
        return x


class StridedDiceBlock(nn.Module):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(StridedDiceBlock, self).__init__()
        assert (out_channels == 2 * in_channels)

        self.branches = Concurrent()
        self.branches.add_module("left_branch", StridedDiceLeftBranch(channels=in_channels))
        self.branches.add_module("right_branch", StridedDiceRightBranch(
            channels=in_channels,
            in_size=in_size))
        self.shuffle = ChannelShuffle(
            channels=out_channels,
            groups=2)

    def forward(self, x):
        x = self.branches(x)
        x = self.shuffle(x)
        return x


class ShuffledDiceRightBranch(nn.Module):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(ShuffledDiceRightBranch, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=(lambda: nn.PReLU(num_parameters=out_channels)))
        self.dice = DiceBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            in_size=in_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.dice(x)
        return x


class ShuffledDiceBlock(nn.Module):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size):
        super(ShuffledDiceBlock, self).__init__()
        self.left_part = in_channels - in_channels // 2
        right_in_channels = in_channels - self.left_part
        right_out_channels = out_channels - self.left_part

        self.right_branch = ShuffledDiceRightBranch(
            in_channels=right_in_channels,
            out_channels=right_out_channels,
            in_size=in_size)
        self.shuffle = ChannelShuffle(
            channels=(2 * right_out_channels),
            groups=2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x2 = self.right_branch(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class DiceInitBlock(nn.Module):
    """
    DiceNet specific initial block.

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
        super(DiceInitBlock, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            activation=(lambda: nn.PReLU(num_parameters=out_channels)))
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DiceClassifier(nn.Module):
    """
    DiceNet specific classifier block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    num_classes : int, default 1000
        Number of classification classes.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 num_classes,
                 dropout_rate):
        super(DiceClassifier, self).__init__()
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            groups=4)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=num_classes,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class DiceNet(nn.Module):
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
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 classifier_mid_channels,
                 dropout_rate,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DiceNet, self).__init__()
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", DiceInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        in_size = (in_size[0] // 4, in_size[1] // 4)
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                unit_class = StridedDiceBlock if j == 0 else ShuffledDiceBlock
                stage.add_module("unit{}".format(j + 1), unit_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    in_size=in_size))
                in_channels = out_channels
                in_size = (in_size[0] // 2, in_size[1] // 2) if j == 0 else in_size
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = DiceClassifier(
            in_channels=in_channels,
            mid_channels=classifier_mid_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_dicenet(width_scale,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
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
    root : str, default '~/.torch/models'
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
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def dicenet_wd5(**kwargs):
    """
    DiCENet x0.2 model from 'DiCENet: Dimension-wise Convolutions for Efficient Networks,'
    https://arxiv.org/abs/1906.03516.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
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
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dicenet(width_scale=2.0, model_name="dicenet_w2", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False

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

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
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

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
