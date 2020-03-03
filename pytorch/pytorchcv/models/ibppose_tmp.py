"""
    IBPPose for COCO Keypoint, implemented in PyTorch.
    Original paper: 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.
"""

__all__ = ['ibppose_coco']

import torch
from torch import nn
from common import conv1x1_block, conv3x3_block, conv7x7_block, SEBlock


class IbpResBottleneck(nn.Module):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bias=False,
                 bottleneck_factor=2,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(IbpResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias,
            activation=activation)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            bias=bias,
            activation=activation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IbpResUnit(nn.Module):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 bias=False,
                 bottleneck_factor=2,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(IbpResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = IbpResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
            bottleneck_factor=bottleneck_factor,
            activation=activation)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class IbpBackbone(nn.Module):
    """
    IBPPose backbone.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation):
        super(IbpBackbone, self).__init__()
        dilations = (3, 3, 4, 4, 5, 5)
        mid1_channels = out_channels // 4
        mid2_channels = out_channels // 2

        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid1_channels,
            stride=2,
            activation=activation)
        self.res1 = IbpResUnit(
            in_channels=mid1_channels,
            out_channels=mid2_channels)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.res2 = IbpResUnit(
            in_channels=mid2_channels,
            out_channels=mid2_channels)
        self.dilation_branch = nn.Sequential()
        for i, dilation in enumerate(dilations):
            self.dilation_branch.add_module("block{}".format(i + 1), conv3x3_block(
                in_channels=mid2_channels,
                out_channels=mid2_channels,
                padding=dilation,
                dilation=dilation,
                activation=activation))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        y = self.dilation_branch(x)
        x = torch.cat((x, y), dim=1)
        return x


class Hourglass(nn.Module):
    """
    Instantiate an n order Hourglass Network block using recursive trick.
    """
    def __init__(self,
                 channels,  # input and output channels
                 depth,  # oder number
                 increase,  # increased channels while the depth grows
                 use_bn,
                 activation):
        super(Hourglass, self).__init__()
        self.channels = channels
        self.depth = depth
        self.increase = increase
        self.use_bn = use_bn
        self.activation = activation

        self.hg = self._make_hour_glass()

        self.down = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest")

    def _make_lower_residual(self,
                             depth_id):
        pack_layers = [
            IbpResUnit(
                in_channels=self.channels + self.increase * depth_id,
                out_channels=self.channels + self.increase * depth_id,
                activation=self.activation),
            IbpResUnit(
                in_channels=self.channels + self.increase * depth_id,
                out_channels=self.channels + self.increase * (depth_id + 1),
                activation=self.activation),
            IbpResUnit(
                in_channels=self.channels + self.increase * (depth_id + 1),
                out_channels=self.channels + self.increase * depth_id,
                activation=self.activation),
            conv3x3_block(
                in_channels=self.channels + self.increase * depth_id,
                out_channels=self.channels + self.increase * depth_id,
                bias=(not self.use_bn),
                use_bn=self.use_bn,
                activation=self.activation),
        ]
        return pack_layers

    def _make_single_residual(self, depth_id):
        # the innermost conve layer, return as a layer item
        # ###########  Index: 4
        return IbpResUnit(
            in_channels=self.channels + self.increase * (depth_id + 1),
            out_channels=self.channels + self.increase * (depth_id + 1),
            activation=self.activation)

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            res = self._make_lower_residual(i)
            if i == (self.depth - 1):
                res.append(self._make_single_residual(i))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self,
                            depth_id,
                            x,
                            up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.down(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):  # except for the highest-order hourglass block
            low2 = self.hg[depth_id][4](low1)
        else:
            # call the lower-order hourglass block recursively
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        # ######################## # if we don't consider 8*8 scale
        # if depth_id < self.depth - 1:
        #     self.up_fms.append(low2)
        up2 = self.up(low3)
        deconv1 = self.hg[depth_id][3](up2)
        # deconv2 = self.hg[depth_id][4](deconv1)
        # up1 += deconv2
        # out = self.hg[depth_id][5](up1)  # relu after residual add
        return up1 + deconv1

    def forward(self, x):
        """
        :param: x a input tensor warpped wrapped as a list
        :return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
        """
        up_fms = []  # collect feature maps produced by low2 at every scale
        feature_map = self._hour_glass_forward(0, x, up_fms)
        return [feature_map] + up_fms[::-1]


class Merge(nn.Module):
    """
    Change the channel dimension of the input tensor
    """
    def __init__(self,
                 x_dim,
                 y_dim,
                 use_bn):
        super(Merge, self).__init__()
        self.conv = conv1x1_block(
            in_channels=x_dim,
            out_channels=y_dim,
            bias=(not use_bn),
            use_bn=use_bn,
            activation=None)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """
    Input: feature maps produced by hourglass block
    Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
    """
    def __init__(self,
                 out_channels,
                 use_bn,
                 increase,
                 scales):
        super(Features, self).__init__()
        self.scales = scales
        # Regress 5 different scales of heatmaps per stack
        self.before_regress = nn.ModuleList(
            [nn.Sequential(
                conv3x3_block(
                    in_channels=(out_channels + i * increase),
                    out_channels=out_channels,
                    bias=(not use_bn),
                    use_bn=use_bn),
                conv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    bias=(not use_bn),
                    use_bn=use_bn),
                SEBlock(channels=out_channels),
            ) for i in range(scales)])

    def forward(self, fms):
        return [self.before_regress[i](fms[i]) for i in range(self.scales)]


class IbpPose(nn.Module):
    def __init__(self,
                 nstack,
                 backbone_out_channels,
                 oup_dim,
                 bn=False,
                 increase=128,
                 init_weights=True,
                 in_channels=3,
                 **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param backbone_out_channels: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(IbpPose, self).__init__()
        activation = (lambda: nn.LeakyReLU(inplace=True))
        self.scales = 5
        self.nstack = nstack

        self.backbone = IbpBackbone(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            activation=activation)

        self.hourglass = nn.ModuleList([Hourglass(
            channels=backbone_out_channels,
            depth=4,
            increase=increase,
            use_bn=bn,
            activation=activation) for _ in range(nstack)])

        self.features = nn.ModuleList([Features(
            out_channels=backbone_out_channels,
            use_bn=bn,
            increase=increase,
            scales=self.scales) for _ in range(nstack)])

        self.outs = nn.ModuleList(
            [nn.ModuleList([conv1x1_block(
                in_channels=backbone_out_channels,
                out_channels=oup_dim,
                bias=True,
                use_bn=False,
                activation=None) for j in range(self.scales)]) for i in
             range(nstack)])

        self.merge_features = nn.ModuleList(
            [nn.ModuleList([Merge(
                backbone_out_channels,
                backbone_out_channels + j * increase,
                use_bn=bn) for j in range(self.scales)]) for i in range(nstack - 1)])

        self.merge_preds = nn.ModuleList(
            [nn.ModuleList([Merge(
                oup_dim,
                backbone_out_channels + j * increase,
                use_bn=bn) for j in range(self.scales)]) for i in range(nstack - 1)])

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, imgs):
        x = imgs
        x = self.backbone(x)
        pred = []
        for i in range(self.nstack):
            preds_instack = []
            hourglass_feature = self.hourglass[i](x)

            if i == 0:
                features_cache = [torch.zeros_like(hourglass_feature[scale]) for scale in range(self.scales)]
            else:
                hourglass_feature = [hourglass_feature[scale] + features_cache[scale] for scale in range(self.scales)]

            features_instack = self.features[i](hourglass_feature)

            for j in range(self.scales):
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]) +\
                            self.merge_features[i][j](features_instack[j])
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) +\
                                            self.merge_features[i][j](features_instack[j])

                    else:
                        # reset the res caches
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) +\
                                            self.merge_features[i][j](features_instack[j])
            pred.append(preds_instack)

        y = pred[-1][0]
        return y


def ibppose_coco(pretrained=False, num_classes=3, in_channels=3, **kwargs):
    model = IbpPose(4, 256, 54, bn=True, **kwargs)
    return model


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
        ibppose_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibppose_coco or weight_count == 129050040)

        x = torch.randn(14, 3, 256, 256)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 54, 64, 64))


if __name__ == "__main__":
    _test()
