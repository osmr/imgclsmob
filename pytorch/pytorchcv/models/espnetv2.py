"""
    ESPNetv2, implemented in PyTorch.
    Original paper: 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.
"""

__all__ = ['ESPNetv2', 'espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d8', 'espnetv2_w3d2', 'espnetv2_w2']

import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3, conv1x1_block, conv3x3_block, DualPathSequential


class PreActivation(nn.Module):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """

    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ShortcutBlock(nn.Module):
    """
    ESPNetv2 shortcut block.

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
        super(ShortcutBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=(lambda: nn.PReLU(in_channels)))
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ESPBlock(nn.Module):
    """
    ESPNetv2 block (so-called EESP block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    branches : int
        Number of parallel branches.
    rfield : int
        Maximum value of receptive field.
    use_avg : bool, default False
        Whether use average pooling.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 branches,
                 rfield,
                 use_avg=False):
        super(ESPBlock, self).__init__()
        assert (out_channels % branches == 0)
        self.stride = stride
        n = out_channels // branches
        n1 = out_channels - (branches - 1) * n
        assert n == n1

        self.proj_1x1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=n,
            groups=branches,
            activation=(lambda: nn.PReLU(n)))

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(branches):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= rfield else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(branches):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(conv3x3(
                in_channels=n,
                out_channels=n,
                stride=stride,
                padding=d_rate,
                dilation=d_rate,
                groups=n))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=branches,
            activation=None,
            activate=False)
        self.br_after_cat = PreActivation(in_channels=out_channels)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = use_avg

    def forward(self, input, input2):
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp(  # learn linear combinations using group point-wise convolutions
            self.br_after_cat(  # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1)  # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded, input2

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded), input2


class DownSampler(nn.Module):
    """
    ESPNetv2 downsample block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    branches : int
        Number of parallel branches.
    rfield : int
        Maximum value of receptive field allowed for EESP block.
    x0_in_channels : int
        Number of input channels for shortcut.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 branches,
                 rfield,
                 x0_in_channels):
        super(DownSampler, self).__init__()
        inc_channels = out_channels - in_channels

        self.avg_pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1)
        self.eesp = ESPBlock(
            in_channels=in_channels,
            out_channels=inc_channels,
            stride=2,
            branches=branches,
            rfield=rfield,
            use_avg=True)
        self.shortcut_block = ShortcutBlock(
            in_channels=x0_in_channels,
            out_channels=out_channels)
        self.activ = nn.PReLU(out_channels)

    def forward(self, x, x0):
        x1 = self.avg_pool(x)
        x2, _ = self.eesp(x, None)
        x = torch.cat((x1, x2), dim=1)
        x0 = self.avg_pool(x0)
        x3 = self.shortcut_block(x0)
        x = x + x3
        x = self.activ(x)
        return x, x0


class ESPInitBlock(nn.Module):
    """
    ESPNetv2 initial block.

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
        super(ESPInitBlock, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            activation=(lambda: nn.PReLU(out_channels)))
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x, x0):
        x = self.conv(x)
        x0 = self.pool(x0)
        return x, x0


class ESPFinalBlock(nn.Module):
    """
    ESPNetv2 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    final_groups : int
        Number of groups in the last convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 final_groups):
        super(ESPFinalBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            activation=(lambda: nn.PReLU(in_channels)))
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=final_groups,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ESPNetv2(nn.Module):
    """
    ESPNetv2 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network,'
    https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    branches : int
        Number of parallel branches.
    rfields : list of list of int
        Number of receptive field limits for each unit.
    dropout_rate : float, default 0.2
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 final_block_channels,
                 branches,
                 rfields,
                 dropout_rate=0.2,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ESPNetv2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        x0_in_channels = in_channels

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=0,
            last_ordinals=3)
        self.features.add_module("init_block", ESPInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    unit = DownSampler(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        branches=branches,
                        rfield=rfields[i][j],
                        x0_in_channels=x0_in_channels)
                else:
                    unit = ESPBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        branches=branches,
                        rfield=rfields[i][j])
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', ESPFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            final_groups=branches))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))
        self.features.add_module("final_dropout", nn.Dropout(p=dropout_rate))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x, x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_espnetv2(width_scale,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join('~', '.torch', 'models'),
                 **kwargs):
    """
    Create ESPNetv2 model with specific parameters.

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
    assert (width_scale <= 2.0)

    branches = 4
    layers = [1, 4, 8, 4]

    rfields_list = [13, 11, 9, 7, 5]
    rfields = [[rfields_list[i]] + [rfields_list[i + 1]] * (li - 1) for (i, li) in enumerate(layers)]

    base_channels = 32
    weighed_base_channels = math.ceil(float(math.floor(base_channels * width_scale)) / branches) * branches
    channels_per_layers = [weighed_base_channels * pow(2, i + 1) for i in range(len(layers))]

    init_block_channels = base_channels if weighed_base_channels > base_channels else weighed_base_channels
    final_block_channels = 1024 if width_scale <= 1.5 else 1280

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ESPNetv2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        branches=branches,
        rfields=rfields,
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


def espnetv2_wd2(**kwargs):
    """
    ESPNetv2 x0.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=0.5, model_name="espnetv2_wd2", **kwargs)


def espnetv2_w1(**kwargs):
    """
    ESPNetv2 x1.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.0, model_name="espnetv2_w1", **kwargs)


def espnetv2_w5d8(**kwargs):
    """
    ESPNetv2 x1.25 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.25, model_name="espnetv2_w5d8", **kwargs)


def espnetv2_w3d2(**kwargs):
    """
    ESPNetv2 x1.5 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=1.5, model_name="espnetv2_w3d2", **kwargs)


def espnetv2_w2(**kwargs):
    """
    ESPNetv2 x2.0 model from 'ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural
    Network,' https://arxiv.org/abs/1811.11431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnetv2(width_scale=2.0, model_name="espnetv2_w2", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        espnetv2_wd2,
        espnetv2_w1,
        espnetv2_w5d8,
        espnetv2_w3d2,
        espnetv2_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnetv2_wd2 or weight_count == 1241332)
        assert (model != espnetv2_w1 or weight_count == 1670072)
        assert (model != espnetv2_w5d8 or weight_count == 1965440)
        assert (model != espnetv2_w3d2 or weight_count == 2314856)
        assert (model != espnetv2_w2 or weight_count == 3498136)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
