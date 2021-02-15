"""
    ESPNet for image segmentation, implemented in PyTorch.
    Original paper: 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NormActivation, conv1x1, conv3x3, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock,\
    Concurrent, DualPathSequential, InterpolationBlock


class DownSamplerB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps):
        super().__init__()
        n = int(out_channels / 5)
        n1 = out_channels - 4 * n
        self.c1 = conv3x3(
            in_channels=in_channels,
            out_channels=n,
            stride=2)
        self.d1 = conv3x3(
            in_channels=n,
            out_channels=n1,
            padding=1,
            dilation=1)
        self.d2 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=2,
            dilation=2)
        self.d4 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=4,
            dilation=4)
        self.d8 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=8,
            dilation=8)
        self.d16 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=16,
            dilation=16)
        self.morm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.c1(x)
        d1 = self.d1(x)
        d2 = self.d2(x)
        d4 = self.d4(x)
        d8 = self.d8(x)
        d16 = self.d16(x)
         
        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced 
        # by the large effective receptive filed of the ESP module 
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        x = torch.cat([d1, add1, add2, add3, add4], 1)
        x = self.morm_activ(x)
        return x


class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 add,
                 bn_eps):
        super().__init__()
        self.add = add
        n = int(out_channels / 5)  #K=5,
        n1 = out_channels - 4 * n  #(N-(K-1)INT(N/K)) for dilation rate of 2^0, for producing an output feature map of channel=nOut

        self.c1 = conv1x1(
            in_channels=in_channels,
            out_channels=n)  #the point-wise convolutions with 1x1 help in reducing the computation, channel=c

        #K=5, dilation rate: 2^{k-1},k={1,2,3,...,K}
        self.d1 = conv3x3(
            in_channels=n,
            out_channels=n1,
            padding=1,
            dilation=1)  # dilation rate of 2^0
        self.d2 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=2,
            dilation=2)  # dilation rate of 2^1
        self.d4 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=4,
            dilation=4)  # dilation rate of 2^2
        self.d8 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=8,
            dilation=8)  # dilation rate of 2^3
        self.d16 = conv3x3(
            in_channels=n,
            out_channels=n,
            padding=16,
            dilation=16)  # dilation rate of 2^4
        self.morm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        identity = x

        # reduce
        x = self.c1(x)
        # split and transform
        d1 = self.d1(x)
        d2 = self.d2(x)
        d4 = self.d4(x)
        d8 = self.d8(x)
        d16 = self.d16(x)

        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced 
        # by the large effective receptive filed of the ESP module 
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        #merge
        x = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            x = x + identity
        x = self.morm_activ(x)
        return x


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3, for input reinforcement, which establishes a direct link between 
    the input image and encoding stage, improving the flow of information.    
    '''
    def __init__(self,
                 samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(
                3,
                stride=2,
                padding=1))

    def forward(self, x):
        '''
        :param x: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            x = pool(x)
        return x


class ESPNetC(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
    '''
    def __init__(self,
                 num_classes=19,
                 p=5,
                 q=3,
                 bn_eps=1e-03):
        super().__init__()
        self.level1 = conv3x3_block(
            in_channels=3,
            out_channels=16,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(16)))      # feature map size divided 2,                         1/2
        self.sample1 = InputProjectionA(samplingTimes=1)  #down-sample for input reinforcement, factor=2
        self.sample2 = InputProjectionA(samplingTimes=2)  #down-sample for input reinforcement, factor=4

        self.b1 = NormActivation(
            in_channels=(16 + 3),
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(16 + 3)))
        self.level2_0 = DownSamplerB(
            in_channels=(16 + 3),
            out_channels=64,
            bn_eps=bn_eps)  # Downsample Block, feature map size divided 2,    1/4

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(
                in_channels=64,
                out_channels=64,
                add=True,
                bn_eps=bn_eps))  # ESP block
        self.b2 = NormActivation(
            in_channels=(128 + 3),
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(128 + 3)))

        self.level3_0 = DownSamplerB(
            in_channels=128 + 3,
            out_channels=128,
            bn_eps=bn_eps)  # Downsample Block, feature map size divided 2,   1/8
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(
                in_channels=128,
                out_channels=128,
                add=True,
                bn_eps=bn_eps))  # ESPblock

        self.b3 = NormActivation(
            in_channels=256,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(256)))

        self.classifier = conv1x1(
            in_channels=256,
            out_channels=num_classes)

    def forward(self, x):
        output0 = self.level1(x)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        out = F.upsample(classifier, x.size()[2:], mode='bilinear')   #Upsample score map, factor=8
        return out


class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
    '''

    def __init__(self,
                 num_classes=19,
                 p=2,
                 q=3,
                 bn_eps=1e-03):
        super().__init__()
        encoder = ESPNetC(
            num_classes=num_classes,
            p=p,
            q=q,
            bn_eps=bn_eps)
        # load the encoder modules
        self.en_modules = []
        for i, m in enumerate(encoder.children()):
            self.en_modules.append(m)

        # light-weight decoder
        self.level3_C = conv1x1(
            in_channels=(128 + 3),
            out_channels=num_classes)
        self.br = nn.BatchNorm2d(num_classes, eps=1e-03)
        self.conv = conv3x3_block(
            in_channels=(19 + num_classes),
            out_channels=num_classes,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(num_classes)))

        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(
            num_classes,
            num_classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False))
        self.combine_l2_l3 = nn.Sequential(
            NormActivation(
                in_channels=(2 * num_classes),
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(2 * num_classes))),
            DilatedParllelResidualBlockB(
                in_channels=(2 * num_classes),
                out_channels=num_classes,
                add=False,
                bn_eps=bn_eps))

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(
                num_classes,
                num_classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False),
            NormActivation(
                in_channels=num_classes,
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(num_classes))))

        self.head = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False)

    def forward(self, x):
        output0 = self.en_modules[0](x)
        inp1 = self.en_modules[1](x)
        inp2 = self.en_modules[2](x)

        output0_cat = self.en_modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.en_modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.en_modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.en_modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.en_modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.en_modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.en_modules[9](torch.cat([output2_0, output2], 1)) # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.en_modules[10](output2_cat))) #RUM

        output1_C = self.level3_C(output1_cat) # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1))) #RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.head(concat_features)

        return classifier


def espnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ESPNet(num_classes=num_classes, **kwargs)


def espnetc_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ESPNetC(num_classes=num_classes, **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    # fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        # espnet_cityscapes,
        espnetc_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != espnet_cityscapes or weight_count == 201542)
        assert (model != espnetc_cityscapes or weight_count == 210889)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
