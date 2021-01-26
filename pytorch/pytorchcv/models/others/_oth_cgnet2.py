
__all__ = ["oth_cgnet_cityscapes"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super(ChannelWiseDilatedConv, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=(padding, padding),
            groups=in_channels,
            bias=False,
            dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self,
                 channel,
                 reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlockDown(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rate=2,
                 reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super(ContextGuidedBlockDown, self).__init__()
        bn_eps = 1e-3

        self.conv1x1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))  #  size/2, channel: nIn--->nOut
        
        self.F_loc = depthwise_conv3x3(
            channels=out_channels,
            stride=1)
        self.F_sur = ChannelWiseDilatedConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation_rate)
        
        self.bn = nn.BatchNorm2d(2 * out_channels, eps=1e-3)
        self.act = nn.PReLU(2 * out_channels)
        self.reduce = conv1x1(
            in_channels=(2 * out_channels),
            out_channels=out_channels)  #reduce dimension: 2*nOut--->nOut
        
        self.F_glo = FGlo(out_channels, reduction)

    def forward(self, x):
        x = self.conv1x1(x)
        loc = self.F_loc(x)
        sur = self.F_sur(x)

        joi_feat = torch.cat([loc, sur], 1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut
        
        x = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        return x


class ContextGuidedBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rate=2,
                 reduction=16,
                 add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super(ContextGuidedBlock, self).__init__()
        bn_eps = 1e-3

        n = int(out_channels / 2)
        self.conv1x1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=n,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(n)))  # 1x1 Conv is employed to reduce the computation
        self.F_loc = depthwise_conv3x3(
            channels=n,
            stride=1) # local feature
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate) # surrounding context
        self.bn_prelu = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.add = add
        self.F_glo = FGlo(out_channels, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output


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


class CGNet(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """
    def __init__(self,
                 num_classes=19,
                 M=3,
                 N=21,
                 dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super(CGNet, self).__init__()
        bn_eps = 1e-3

        self.level1_0 = conv3x3_block(
            in_channels=3,
            out_channels=32,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(32)))      # feature map size divided 2, 1/2
        self.level1_1 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(32)))
        self.level1_2 = conv3x3_block(
            in_channels=32,
            out_channels=32,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(32)))

        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4

        channels1 = 32 + 3
        self.b1 = NormActivation(
            in_channels=channels1,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(channels1)))
        
        #stage 2
        self.level2_0 = ContextGuidedBlockDown(32 + 3, 64, dilation_rate=2, reduction=8)
        self.level2 = nn.ModuleList()
        for i in range(0, M-1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))  # CG block

        channels1 = 128 + 3
        self.bn_prelu_2 = NormActivation(
            in_channels=channels1,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(channels1)))
        
        #stage 3
        self.level3_0 = ContextGuidedBlockDown(128 + 3, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList()
        for i in range(0, N-1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16))  # CG block

        channels1 = 256
        self.bn_prelu_3 = NormActivation(
            in_channels=channels1,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(channels1)))

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False),
                conv1x1(
                    in_channels=256,
                    out_channels=num_classes))
        else:
            self.classifier = nn.Sequential(
                conv1x1(
                    in_channels=256,
                    out_channels=num_classes))

        #init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d')!= -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1,  output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.upsample(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return out


def oth_cgnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    net = CGNet(num_classes=num_classes)
    return net


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False

    in_size = (1024, 2048)

    models = [
        oth_cgnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_cgnet_cityscapes or weight_count == 496306)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
