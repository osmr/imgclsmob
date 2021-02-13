"""
    ESPNet for image segmentation, implemented in PyTorch.
    Original paper: 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock, Concurrent,\
    DualPathSequential, InterpolationBlock


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self,
                 out_channels):
        '''
        :param out_channels: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution, which can maintain feature map size
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 d=1):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        n = int(out_channels / 5)
        n1 = out_channels - 4 * n
        self.c1 = C(in_channels, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
         
        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced 
        # by the large effective receptive filed of the ESP module 
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine  #shotcut path
        output = self.bn(combine)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 add=True,
                 bn_eps=1e-3):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(out_channels / 5)  #K=5,
        n1 = out_channels - 4 * n  #(N-(K-1)INT(N/K)) for dilation rate of 2^0, for producing an output feature map of channel=nOut
        self.c1 = C(in_channels, n, 1, 1)  #the point-wise convolutions with 1x1 help in reducing the computation, channel=c

        #K=5, dilation rate: 2^{k-1},k={1,2,3,...,K}
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        
        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced 
        # by the large effective receptive filed of the ESP module 
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


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
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
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
        self.sample1 = InputProjectionA(1)  #down-sample for input reinforcement, factor=2
        self.sample2 = InputProjectionA(2)  #down-sample for input reinforcement, factor=4

        self.b1 = NormActivation(
            in_channels=(16 + 3),
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(16 + 3)))
        self.level2_0 = DownSamplerB(16 + 3, 64)  # Downsample Block, feature map size divided 2,    1/4

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))  #ESP block
        self.b2 = BR(out_channels=(128 + 3))

        self.level3_0 = DownSamplerB(128 + 3, 128) #Downsample Block, feature map size divided 2,   1/8
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128)) # ESPblock
        self.b3 = BR(out_channels=256)

        self.classifier = C(256, num_classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

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

        #return classifier
        out = F.upsample(classifier, input.size()[2:], mode='bilinear')   #Upsample score map, factor=8
        return out


class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
    '''

    def __init__(self,
                 num_classes=19,
                 p=2,
                 q=3,
                 encoderFile=None,
                 bn_eps=1e-03):
        super().__init__()
        self.encoder = ESPNet_Encoder(num_classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')
        # load the encoder modules
        self.en_modules = []
        for i, m in enumerate(self.encoder.children()):
            self.en_modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 3, num_classes, 1, 1)
        self.br = nn.BatchNorm2d(num_classes, eps=1e-03)
        self.conv = conv3x3_block(
            in_channels=(19 + num_classes),
            out_channels=num_classes,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(num_classes)))

        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(
            BR(out_channels=2 * num_classes),
            DilatedParllelResidualBlockB(2 * num_classes, num_classes, add=False))

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias=False),
            BR(out_channels=num_classes))

        self.classifier = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output0 = self.en_modules[0](input)
        inp1 = self.en_modules[1](input)
        inp2 = self.en_modules[2](input)

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

        classifier = self.classifier(concat_features)

        return classifier


def espnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ESPNet(num_classes=num_classes, **kwargs)


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
        espnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnet_cityscapes or weight_count == 201542)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
