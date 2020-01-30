'''
C3SINet
Copyright (c) 2019-present NAVER Corp.
MIT licenses
'''

__all__ = ['oth_sinet_cityscapes']

import torch
import torch.nn as nn
BN_moment = 0.1
# Ours

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class separableCBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False),
            nn.Conv2d(nIn, nOut,  kernel_size=1, stride=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4.0):
        super(SqueezeBlock, self).__init__()

        if divide > 1:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, int(exp_size / divide)),
                nn.PReLU(int(exp_size / divide)),
                nn.Linear(int(exp_size / divide), exp_size),
                nn.PReLU(exp_size),
            )
        else:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, exp_size),
                nn.PReLU(exp_size)
            )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = torch.nn.functional.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x

class SEseparableCBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, divide=2.0):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False),
            SqueezeBlock(nIn,divide=divide),
            nn.Conv2d(nIn, nOut,  kernel_size=1, stride=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)

        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

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

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum= BN_moment)

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

    def __init__(self, nIn, nOut, kSize, stride=1,group=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
                              padding=(padding, padding), bias=False, groups=group)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output



class SBblock(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, config):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        kSize = config[0]
        avgsize = config[1]
        self.SB = True if avgsize>0 else False

        if avgsize == 0:
            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, kernel_size=3, stride=1, padding=1, bias=False,groups=nIn),
                nn.BatchNorm2d(nIn, eps=1e-03, momentum=BN_moment),
            )
        else:
            self.resolution_down = False
            if avgsize >1:
                self.resolution_down = True
                self.down_res = nn.AvgPool2d(avgsize, avgsize)
                self.up_res = nn.UpsamplingBilinear2d(scale_factor=avgsize)

            padding = int((kSize - 1) / 2 )
            self.vertical = nn.Conv2d(nIn, nIn, kernel_size=(kSize, 1), stride=1,
                                      padding=(padding, 0), groups=nIn, bias=False)
            self.horizontal = nn.Conv2d(nIn, nIn, kernel_size=(1, kSize), stride=1,
                                        padding=(0, padding), groups = nIn, bias = False)
            self.B_v = nn.BatchNorm2d(nIn, eps=1e-03, momentum=BN_moment)
            self.B_h = nn.BatchNorm2d(nIn, eps=1e-03, momentum=BN_moment)

        self.act_conv1x1 = nn.Sequential(
            nn.PReLU(nIn),
            nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)



    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        if self.SB:
            if self.resolution_down:
                input = self.down_res(input)
            output_v = self.B_v(self.vertical(input))
            output_h = self.B_h(self.horizontal(input))
            output = output_v + output_h

        else:
            output = self.conv(input)

        output = self.act_conv1x1(output)
        if self.SB and self.resolution_down:
            output = self.up_res(output)
        return self.bn(output)


class SBmodule(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, config= [[3,1],[5,1]]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        print("This module has " + str(config))

        group_n = len(config)
        n = int(nOut / group_n)
        n1 = nOut - group_n * n

        self.c1 = C(nIn, n, 1, 1, group=group_n)

        for i in range(group_n):
            var_name = 'd{}'.format(i + 1)
            if i == 0:
                self.__dict__["_modules"][var_name] = SBblock(n, n + n1, config[i])
            else:
                self.__dict__["_modules"][var_name] = SBblock(n, n,  config[i])

        self.BR = BR(nOut)
        self.add = add
        self.group_n = group_n

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        output1= channel_shuffle(output1, self.group_n)

        for i in range(self.group_n):
            var_name = 'd{}'.format(i + 1)
            result_d = self.__dict__["_modules"][var_name](output1)
            if i == 0:
                combine = result_d
            else:
                combine = torch.cat([combine, result_d], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.BR(combine)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(2, stride=2))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input



class SBNet_Encoder(nn.Module):

    def __init__(self, config,classes=20, p=5, q=3,  chnn=1.0):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        print("SB Net Enc bracnch num :  " + str(len(config[0])))
        print("SB Net Enc chnn num:  " + str(chnn))
        dim1 = 24
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 72 + 4 * (chnn - 1)
        dim4 = 96 + 4 * (chnn - 1)

        self.level1 = CBR(3, 16, 3, 2)

        self.level2_0 = SEseparableCBR(16,classes, 3,2, divide=1)
        self.level3_0 = SEseparableCBR(classes,dim1, 3,2, divide=1)

        self.level3 = nn.ModuleList()
        for i in range(0, p):
            if i ==0:
                self.level3.append(SBmodule(dim1, dim2, config=config[i], add=False))
            else:
                self.level3.append(SBmodule(dim2, dim2,config=config[i]))
        self.BR3 = BR(dim2+dim1)

        self.level4_0 =SEseparableCBR(dim2+dim1,dim2, 3,2, divide=2)
        self.level4 = nn.ModuleList()
        for i in range(0, q//2):
            if i==0:
                self.level4.append(SBmodule(dim2, dim3, config=config[p + i], add=False))
            else:
                self.level4.append(SBmodule(dim3, dim3,config=config[p+i]))

        for i in range(q//2,q):
            if  i == q//2:
                self.level4.append(SBmodule(dim3, dim4, config=config[p + i], add=False))
            else:
                self.level4.append(SBmodule(dim4, dim4,config=config[p+i]))

        self.BR4 = BR(dim4+dim2)

        self.classifier = C(dim4+dim2, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output1 = self.level1(input) #8h 8w


        output2_0 = self.level2_0(output1)  # 4h 4w
        output3_0 = self.level3_0(output2_0)  # 4h 4w

        # print(str(output1_0.size()))
        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3) # 2h 2w


        output4_0 = self.level4_0(self.BR3(torch.cat([output3_0, output3],1)))  # h w
        # print(str(output2_0.size()))

        for i, layer in enumerate(self.level4):
            if i == 0:
                output4 = layer(output4_0)
            else:
                output4 = layer(output4)

        output4_cat = self.BR4(torch.cat([output4_0, output4], 1))

        classifier = self.classifier(output4_cat)

        return classifier


class SBNet_aux(nn.Module):

    def __init__(self, config, num_classes=20, p=2, q=3, chnn=1.0, encoderFile=None, in_channels=3, in_size=(480, 480)):
        '''
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.num_classes = num_classes

        print("SB Net Enc bracnch num :  " + str(len(config[0])))
        print("SB Net Enc chnn num:  " + str(chnn))
        dim1 = 24
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 72 + 4 * (chnn - 1)
        dim4 = 96 + 4 * (chnn - 1)

        self.encoder = SBNet_Encoder(config, num_classes, p, q, chnn)
        # # load the encoder modules
        if encoderFile != None:
            if torch.cuda.device_count() ==0:
                self.encoder.load_state_dict(torch.load(encoderFile,map_location="cpu"))
            else:
                self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')

        self.up = nn.functional.interpolate
        self.bn_4 = nn.BatchNorm2d(num_classes, eps=1e-03)

        self.level3_C = CBR(dim2, num_classes, 1, 1)
        self.bn_3 = nn.BatchNorm2d(num_classes, eps=1e-03)

        self.classifier = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, padding=0, output_padding=0, bias=False)


    def forward(self, input, train=False):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output1 = self.encoder.level1(input)  # 8h 8w

        output2_0 = self.encoder.level2_0(output1)  # 4h 4w
        output3_0 = self.encoder.level3_0(output2_0)  # 2h 2w

        # print(str(output1_0.size()))
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)  # 2h 2w

        output4_0 = self.encoder.level4_0(self.encoder.BR3(torch.cat([output3_0, output3], 1)))  # h w
        # print(str(output2_0.size()))

        for i, layer in enumerate(self.encoder.level4):
            if i == 0:
                output4 = layer(output4_0)
            else:
                output4 = layer(output4)

        output4_cat = self.encoder.BR4(torch.cat([output4_0, output4], 1))
        Enc_final = self.encoder.classifier(output4_cat)

        Dnc_stage1 = self.bn_4(self.up(Enc_final, scale_factor=2, mode="bilinear"))  # 2h 2w
        stage1_confidence = nn.Softmax2d()(Dnc_stage1)
        b, c, h, w = Dnc_stage1.size()
        # Coarse_att = ((torch.max(Coarse_confidence,dim=1)[0]).unsqueeze(1)).expand(b,c,h,w)
        stage1_blocking = (torch.max(stage1_confidence, dim=1)[0]).unsqueeze(1).expand(b, c, h, w)

        Dnc_stage2_0 = self.level3_C(output3)  # 2h 2w
        Dnc_stage2 = self.bn_3(
            self.up(Dnc_stage2_0 * (1 - stage1_blocking) + (Dnc_stage1), scale_factor=2, mode="bilinear"))  # 4h 4w

        stage2_confidence = nn.Softmax2d()(Dnc_stage2)  # 4h 4w
        b, c, h, w = Dnc_stage2.size()

        stage2_blocking = (torch.max(stage2_confidence, dim=1)[0]).unsqueeze(1).expand(b, c, h, w)
        Dnc_stage3 = output2_0 * (1 - stage2_blocking) + (Dnc_stage2)

        classifier = self.classifier(Dnc_stage3)

        import torch.nn.functional as F
        classifier = F.interpolate(
            classifier,
            scale_factor=2,
            mode="bilinear",
            align_corners=True)


        if train:
            return Enc_final, classifier
        else :
            return classifier



def Enc_SIN(classes, p, q, chnn):
    # k, avg
    config = [[[3, 1], [5, 1]], [[3, 0], [3, 1]], [[3, 0], [3, 1]],
              [[3, 1], [5, 1]], [[3, 0], [3, 1]], [[5, 1], [5, 4]], [[3, 2], [5, 8]], [[3, 1], [5, 1]],
              [[3, 1], [5, 1]], [[3, 0], [3, 1]], [[5, 1], [5, 8]], [[3, 2], [5, 4]], [[3, 0], [5, 2]]]


    model = SBNet_Encoder(config, classes=classes, p=p, q=q, chnn=chnn)
    return model



def oth_sinet_cityscapes(num_classes=19, p=3, q=10, chnn=4, encoderFile=None, pretrained=False, aux=False,
                         fixed_size=False, **kwargs):
    #
    config = [[[3, 1], [5, 1]], [[3, 0], [3, 1]], [[3, 0], [3, 1]],
              [[3, 1], [5, 1]], [[3, 0], [3, 1]], [[5, 1], [5, 4]], [[3, 2], [5, 8]], [[3, 1], [5, 1]],
              [[3, 1], [5, 1]], [[3, 0], [3, 1]], [[5, 1], [5, 8]], [[3, 2], [5, 4]], [[3, 0], [5, 2]]]


    print("SINet with auxillary loss")
    model = SBNet_aux(config, num_classes=num_classes, p=p, q=q, chnn=chnn, encoderFile=encoderFile, **kwargs)
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
        oth_sinet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_sinet_cityscapes or weight_count == 119418)

        x = torch.randn(14, 3, 1024, 2048)
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (14, 19, 1024, 2048))


if __name__ == "__main__":
    _test()
