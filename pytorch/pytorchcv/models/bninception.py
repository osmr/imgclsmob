"""
    BN-Inception for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,'
    https://arxiv.org/abs/1502.03167.
"""

__all__ = ['BNInception', 'bninception']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block, conv7x7_block, Concurrent


class Inception3x3Branch(nn.Module):
    """
    BN-Inception 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 stride=1,
                 bias=True,
                 use_bn=True):
        super(Inception3x3Branch, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias,
            use_bn=use_bn)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            use_bn=use_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InceptionDouble3x3Branch(nn.Module):
    """
    BN-Inception double 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 stride=1,
                 bias=True,
                 use_bn=True):
        super(InceptionDouble3x3Branch, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias,
            use_bn=use_bn)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=use_bn)
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            use_bn=use_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class InceptionPoolBranch(nn.Module):
    """
    BN-Inception avg-pool branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    avg_pool : bool
        Whether use average pooling or max pooling.
    bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 avg_pool,
                 bias,
                 use_bn):
        super(InceptionPoolBranch, self).__init__()
        if avg_pool:
            self.pool = nn.AvgPool2d(
                kernel_size=3,
                stride=1,
                padding=1,
                ceil_mode=True,
                count_include_pad=True)
        else:
            self.pool = nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1,
                ceil_mode=True)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=use_bn)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class StemBlock(nn.Module):
    """
    BN-Inception stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 bias,
                 use_bn):
        super(StemBlock, self).__init__()
        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2,
            bias=bias,
            use_bn=use_bn)
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True)
        self.conv2 = Inception3x3Branch(
            in_channels=mid_channels,
            out_channels=out_channels,
            mid_channels=mid_channels)
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x


class InceptionBlock(nn.Module):
    """
    BN-Inception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels_list : list of int
        Number of pre-middle channels for branches.
    mid2_channels_list : list of int
        Number of middle channels for branches.
    avg_pool : bool
        Whether use average pooling or max pooling.
    bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels_list,
                 mid2_channels_list,
                 avg_pool,
                 bias,
                 use_bn):
        super(InceptionBlock, self).__init__()
        assert (len(mid1_channels_list) == 2)
        assert (len(mid2_channels_list) == 4)

        self.branches = Concurrent()
        self.branches.add_module("branch1", conv1x1_block(
            in_channels=in_channels,
            out_channels=mid2_channels_list[0],
            bias=bias,
            use_bn=use_bn))
        self.branches.add_module("branch2", Inception3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[1],
            mid_channels=mid1_channels_list[0],
            bias=bias,
            use_bn=use_bn))
        self.branches.add_module("branch3", InceptionDouble3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[2],
            mid_channels=mid1_channels_list[1],
            bias=bias,
            use_bn=use_bn))
        self.branches.add_module("branch4", InceptionPoolBranch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[3],
            avg_pool=avg_pool,
            bias=bias,
            use_bn=use_bn))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBlock(nn.Module):
    """
    BN-Inception reduction block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels_list : list of int
        Number of pre-middle channels for branches.
    mid2_channels_list : list of int
        Number of middle channels for branches.
    bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels_list,
                 mid2_channels_list,
                 bias,
                 use_bn):
        super(ReductionBlock, self).__init__()
        assert (len(mid1_channels_list) == 2)
        assert (len(mid2_channels_list) == 4)

        self.branches = Concurrent()
        self.branches.add_module("branch1", Inception3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[1],
            mid_channels=mid1_channels_list[0],
            stride=2,
            bias=bias,
            use_bn=use_bn))
        self.branches.add_module("branch2", InceptionDouble3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[2],
            mid_channels=mid1_channels_list[1],
            stride=2,
            bias=bias,
            use_bn=use_bn))
        self.branches.add_module("branch3", nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True))

    def forward(self, x):
        x = self.branches(x)
        return x


class BNInception(nn.Module):
    """
    BN-Inception model from 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift,' https://arxiv.org/abs/1502.03167.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels_list : list of int
        Number of output channels for the initial unit.
    mid1_channels_list : list of list of list of int
        Number of pre-middle channels for each unit.
    mid2_channels_list : list of list of list of int
        Number of middle channels for each unit.
    bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels_list,
                 mid1_channels_list,
                 mid2_channels_list,
                 bias=True,
                 use_bn=True,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(BNInception, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", StemBlock(
            in_channels=in_channels,
            out_channels=init_block_channels_list[1],
            mid_channels=init_block_channels_list[0],
            bias=bias,
            use_bn=use_bn))
        in_channels = init_block_channels_list[-1]
        for i, channels_per_stage in enumerate(channels):
            mid1_channels_list_i = mid1_channels_list[i]
            mid2_channels_list_i = mid2_channels_list[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    stage.add_module("unit{}".format(j + 1), ReductionBlock(
                        in_channels=in_channels,
                        mid1_channels_list=mid1_channels_list_i[j],
                        mid2_channels_list=mid2_channels_list_i[j],
                        bias=bias,
                        use_bn=use_bn))
                else:
                    avg_pool = (i != len(channels) - 1) or (j != len(channels_per_stage) - 1)
                    stage.add_module("unit{}".format(j + 1), InceptionBlock(
                        in_channels=in_channels,
                        mid1_channels_list=mid1_channels_list_i[j],
                        mid2_channels_list=mid2_channels_list_i[j],
                        avg_pool=avg_pool,
                        bias=bias,
                        use_bn=use_bn))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_bninception(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".torch", "models"),
                    **kwargs):
    """
    Create BN-Inception model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels_list = [64, 192]
    channels = [[256, 320], [576, 576, 576, 608, 608], [1056, 1024, 1024]]
    mid1_channels_list = [
        [[64, 64],
         [64, 64]],
        [[128, 64],  # 3c
         [64, 96],  # 4a
         [96, 96],  # 4a
         [128, 128],  # 4c
         [128, 160]],  # 4d
        [[128, 192],  # 4e
         [192, 160],  # 5a
         [192, 192]],
    ]
    mid2_channels_list = [
        [[64, 64, 96, 32],
         [64, 96, 96, 64]],
        [[0, 160, 96, 0],  # 3c
         [224, 96, 128, 128],  # 4a
         [192, 128, 128, 128],  # 4b
         [160, 160, 160, 128],  # 4c
         [96, 192, 192, 128]],  # 4d
        [[0, 192, 256, 0],  # 4e
         [352, 320, 224, 128],  # 5a
         [352, 320, 224, 128]],
    ]

    net = BNInception(
        channels=channels,
        init_block_channels_list=init_block_channels_list,
        mid1_channels_list=mid1_channels_list,
        mid2_channels_list=mid2_channels_list,
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


def bninception(**kwargs):
    """
    BN-Inception model from 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift,' https://arxiv.org/abs/1502.03167.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_bninception(model_name="bninception", **kwargs)


def _calc_width(net):
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
        bninception,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bninception or weight_count == 11295240)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
