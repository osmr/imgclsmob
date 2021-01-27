"""
    CGNet for image segmentation, implemented in PyTorch.
    Original paper: 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.
"""

__all__ = ['CGNet', 'cgnet_cityscapes']

import os
import torch
import torch.nn as nn
from common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock, Concurrent,\
    InterpolationBlock


class CGDownBlock(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)

    args:
       nIn: the channel of input feature map
       nOut: the channel of output feature map, and nOut=2*nIn
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 se_reduction,
                 bn_eps):
        super(CGDownBlock, self).__init__()
        mid_channels = 2 * out_channels

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

        self.branches = Concurrent()
        self.branches.add_module("branches1", depthwise_conv3x3(channels=out_channels))
        self.branches.add_module("branches2", depthwise_conv3x3(
            channels=out_channels,
            padding=dilation,
            dilation=dilation))

        self.norm_activ = NormActivation(
            in_channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels)))

        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels)

        self.se = SEBlock(
            channels=out_channels,
            reduction=se_reduction,
            use_conv=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.branches(x)
        x = self.norm_activ(x)
        x = self.conv2(x)
        x = self.se(x)
        return x


class CGBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 se_reduction,
                 bn_eps):
        super(CGBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels)))

        self.branches = Concurrent()
        self.branches.add_module("branches1", depthwise_conv3x3(channels=mid_channels))
        self.branches.add_module("branches2", depthwise_conv3x3(
            channels=mid_channels,
            padding=dilation,
            dilation=dilation))

        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

        self.se = SEBlock(
            channels=out_channels,
            reduction=se_reduction,
            use_conv=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.branches(x)
        x = self.norm_activ(x)
        x = self.se(x)
        x += identity
        return x


class InputInjection(nn.Module):
    def __init__(self,
                 downsampling_ratio):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsampling_ratio):
            self.pool.append(nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class CGUnit(nn.Module):
    """
    CGNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_eps):
        super(CGUnit, self).__init__()
        mid_channels = out_channels // 2

        self.down = CGDownBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            dilation=dilation,
            se_reduction=se_reduction,
            bn_eps=bn_eps)
        self.blocks = nn.Sequential()
        for i in range(layers - 1):
            self.blocks.add_module("block{}".format(i + 1), CGBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                dilation=dilation,
                se_reduction=se_reduction,
                bn_eps=bn_eps))

    def forward(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = torch.cat((y, x), dim=1)
        return x


class CGStage(nn.Module):
    """
    CGNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    dilations : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_eps):
        super(CGStage, self).__init__()
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        if self.use_x:
            self.x_down = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1)

        if self.use_unit:
            self.unit = CGUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                layers=layers,
                dilation=dilation,
                se_reduction=se_reduction,
                bn_eps=bn_eps)

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(y_out_channels)))

    def forward(self, y, x=None):
        if self.use_unit:
            y = self.unit(y)
        if self.use_x:
            x = self.x_down(x)
            y = torch.cat((y, x), dim=1)
        y = self.norm_activ(y)
        return y, x


class CGInitBlock(nn.Module):
    """
    CGNet specific initial block.

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
                 bn_eps):
        super(CGInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CGNet(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.

    args:
      classes: number of classes in the dataset. Default is 19 for the cityscapes
      M: the number of blocks in stage 2
      N: the number of blocks in stage 3
    """
    def __init__(self,
                 init_block_channels,
                 layers,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(CGNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.init_block = CGInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps)

        # self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        # self.sample2 = InputInjection(2)  # down-sample for Input Injiection, factor=4

        # channels1 = 32 + 3
        # self.b1 = NormActivation(
        #     in_channels=channels1,
        #     bn_eps=bn_eps,
        #     activation=(lambda: nn.PReLU(channels1)))
        y_in_channels = 32
        y_out_channels = 32 + 3
        self.stage1 = CGStage(
            x_channels=in_channels,
            y_in_channels=y_in_channels,
            y_out_channels=y_out_channels,
            layers=layers[0],
            dilation=0,
            se_reduction=0,
            bn_eps=bn_eps)
        y_in_channels = y_out_channels

        # # stage 2
        # self.level2_0 = CGDownBlock(32 + 3, 64, dilation_rate=2, reduction=8)
        # self.level2 = nn.ModuleList()
        # for i in range(layers[0] - 1):
        #     self.level2.append(CGBlock(64, 64, dilation_rate=2, reduction=8))  # CG block
        #
        # channels1 = 128 + 3
        # self.bn_prelu_2 = NormActivation(
        #     in_channels=channels1,
        #     bn_eps=bn_eps,
        #     activation=(lambda: nn.PReLU(channels1)))
        y_out_channels = 128 + 3
        self.stage2 = CGStage(
            x_channels=in_channels,
            y_in_channels=y_in_channels,
            y_out_channels=y_out_channels,
            layers=layers[1],
            dilation=2,
            se_reduction=8,
            bn_eps=bn_eps)
        y_in_channels = y_out_channels

        # # stage 3
        # self.level3_0 = CGDownBlock(128 + 3, 128, dilation_rate=4, reduction=16)
        # self.level3 = nn.ModuleList()
        # for i in range(layers[1] - 1):
        #     self.level3.append(CGBlock(128, 128, dilation_rate=4, reduction=16))  # CG block
        #
        # channels1 = 256
        # self.bn_prelu_3 = NormActivation(
        #     in_channels=channels1,
        #     bn_eps=bn_eps,
        #     activation=(lambda: nn.PReLU(channels1)))

        y_out_channels = 256
        self.stage3 = CGStage(
            x_channels=0,
            y_in_channels=y_in_channels,
            y_out_channels=y_out_channels,
            layers=layers[2],
            dilation=4,
            se_reduction=16,
            bn_eps=bn_eps)
        y_in_channels = y_out_channels

        self.classifier = conv1x1(
            in_channels=y_in_channels,
            out_channels=num_classes)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]

        # stage 1
        output0 = self.init_block(x)

        # inp1 = self.sample1(x)
        # output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output0_cat, inp1 = self.stage1(output0, x)

        # inp2 = self.sample2(x)
        # # stage 2
        # output1_0 = self.level2_0(output0_cat)  # down-sampled
        # for i, layer in enumerate(self.level2):
        #     if i == 0:
        #         output1 = layer(output1_0)
        #     else:
        #         output1 = layer(output1)
        # output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))
        output1_cat, inp2 = self.stage2(output0_cat, inp1)

        # stage 3
        # output2_0 = self.level3_0(output1_cat)  # down-sampled
        # for i, layer in enumerate(self.level3):
        #     if i == 0:
        #         output2 = layer(output2_0)
        #     else:
        #         output2 = layer(output2)
        # output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        output2_cat, _ = self.stage3(output1_cat, inp2)

        y = self.classifier(output2_cat)
        y = self.up(y, size=in_size)
        return y


def get_cgnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create CGNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    # channels = [35, 131, 259]
    # dilations = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]
    layers = [0, 3, 21]
    bn_eps = 1e-3

    net = CGNet(
        # channels=channels,
        init_block_channels=init_block_channels,
        layers=layers,
        # dilations=dilations,
        bn_eps=bn_eps,
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


def cgnet_cityscapes(num_classes=19, **kwargs):
    """
    CGNet model for Cityscapes from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_cgnet(num_classes=num_classes, model_name="cgnet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        cgnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cgnet_cityscapes or weight_count == 496306)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
