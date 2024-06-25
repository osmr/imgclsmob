"""
    InceptionResNetV1 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV1', 'inceptionresnetv1', 'InceptionAUnit', 'InceptionBUnit', 'InceptionCUnit',
           'ReductionAUnit', 'ReductionBUnit']

import os
import torch.nn as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent
from .inceptionv3 import MaxPoolBranch, Conv1x1Branch, ConvSeqBranch


class InceptionAUnit(nn.Module):
    """
    InceptionResNetV1 type Inception-A unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps):
        super(InceptionAUnit, self).__init__()
        self.scale = 0.17

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:3],
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            bn_eps=bn_eps))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[3:6],
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            bn_eps=bn_eps))
        conv_in_channels = out_channels_list[0] + out_channels_list[2] + out_channels_list[5]
        self.conv = conv1x1(
            in_channels=conv_in_channels,
            out_channels=in_channels,
            bias=True)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionBUnit(nn.Module):
    """
    InceptionResNetV1 type Inception-B unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps):
        super(InceptionBUnit, self).__init__()
        self.scale = 0.10

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:4],
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            bn_eps=bn_eps))
        conv_in_channels = out_channels_list[0] + out_channels_list[3]
        self.conv = conv1x1(
            in_channels=conv_in_channels,
            out_channels=in_channels,
            bias=True)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionCUnit(nn.Module):
    """
    InceptionResNetV1 type Inception-C unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    scale : float, default 0.2
        Scale value for residual branch.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps,
                 scale=0.2,
                 activate=True):
        super(InceptionCUnit, self).__init__()
        self.activate = activate
        self.scale = scale

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:4],
            kernel_size_list=(1, (1, 3), (3, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 1), (1, 0)),
            bn_eps=bn_eps))
        conv_in_channels = out_channels_list[0] + out_channels_list[3]
        self.conv = conv1x1(
            in_channels=conv_in_channels,
            out_channels=in_channels,
            bias=True)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        if self.activate:
            x = self.activ(x)
        return x


class ReductionAUnit(nn.Module):
    """
    InceptionResNetV1 type Reduction-A unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps):
        super(ReductionAUnit, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[0:1],
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[1:4],
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            bn_eps=bn_eps))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(nn.Module):
    """
    InceptionResNetV1 type Reduction-B unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_eps):
        super(ReductionBUnit, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[0:2],
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[2:4],
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            bn_eps=bn_eps))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=out_channels_list[4:7],
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            bn_eps=bn_eps))
        self.branches.add_module("branch4", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
    """
    InceptionResNetV1 specific initial block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 bn_eps):
        super(InceptInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=32,
            stride=2,
            padding=0,
            bn_eps=bn_eps)
        self.conv2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            stride=1,
            padding=0,
            bn_eps=bn_eps)
        self.conv3 = conv3x3_block(
            in_channels=32,
            out_channels=64,
            stride=1,
            padding=1,
            bn_eps=bn_eps)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)
        self.conv4 = conv1x1_block(
            in_channels=64,
            out_channels=80,
            stride=1,
            padding=0,
            bn_eps=bn_eps)
        self.conv5 = conv3x3_block(
            in_channels=80,
            out_channels=192,
            stride=1,
            padding=0,
            bn_eps=bn_eps)
        self.conv6 = conv3x3_block(
            in_channels=192,
            out_channels=256,
            stride=2,
            padding=0,
            bn_eps=bn_eps)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class InceptHead(nn.Module):
    """
    InceptionResNetV1 specific classification block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    dropout_rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    num_classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 bn_eps,
                 dropout_rate,
                 num_classes):
        super(InceptHead, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(
            in_features=in_channels,
            out_features=512,
            bias=False)
        self.bn = nn.BatchNorm1d(
            num_features=512,
            eps=bn_eps)
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=num_classes)

    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x


class InceptionResNetV1(nn.Module):
    """
    InceptionResNetV1 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (299, 299)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_prob=0.6,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(299, 299),
                 num_classes=1000):
        super(InceptionResNetV1, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        layers = [5, 11, 7]
        in_channels_list = [256, 896, 1792]
        normal_out_channels_list = [[32, 32, 32, 32, 32, 32], [128, 128, 128, 128], [192, 192, 192, 192]]
        reduction_out_channels_list = [[384, 192, 192, 256], [256, 384, 256, 256, 256, 256, 256]]

        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("init_block", InceptInitBlock(
            in_channels=in_channels,
            bn_eps=bn_eps))
        in_channels = in_channels_list[0]
        for i, layers_per_stage in enumerate(layers):
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                    out_channels_list_per_stage = reduction_out_channels_list[i - 1]
                else:
                    unit = normal_units[i]
                    out_channels_list_per_stage = normal_out_channels_list[i]
                if (i == len(layers) - 1) and (j == layers_per_stage - 1):
                    unit_kwargs = {"scale": 1.0, "activate": False}
                else:
                    unit_kwargs = {}
                stage.add_module("unit{}".format(j + 1), unit(
                    in_channels=in_channels,
                    out_channels_list=out_channels_list_per_stage,
                    bn_eps=bn_eps,
                    **unit_kwargs))
                if (j == 0) and (i != 0):
                    in_channels = in_channels_list[i]
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=8,
            stride=1))

        self.output = InceptHead(
            in_channels=in_channels,
            bn_eps=bn_eps,
            dropout_rate=dropout_prob,
            num_classes=num_classes)

        self._init_params()

    def _init_params(self):
        for module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_inceptionresnetv1(model_name=None,
                          pretrained=False,
                          root: str = os.path.join("~", ".torch", "models"),
                          **kwargs):
    """
    Create InceptionResNetV1 model with specific parameters.

    Parameters
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = InceptionResNetV1(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def inceptionresnetv1(**kwargs):
    """
    InceptionResNetV1 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_inceptionresnetv1(model_name="inceptionresnetv1", bn_eps=1e-3, **kwargs)


def calc_net_weights(net: nn.Module) -> int:
    """
    Calculate network trainable weight count.

    Parameters
    ----------
    net : nn.Module
        Network.

    Returns
    -------
    int
        Calculated number of weights.
    """
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        inceptionresnetv1,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weights(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv1 or weight_count == 23995624)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
