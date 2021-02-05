import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2


def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize,groups,
               channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    # flatten
    x = x.view(batchsize,-1,height,width)
    
    return x
    

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock (nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock,self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        oup_inc = chann//2
        
        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)       
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)       
        
    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
    
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if self.dropout.p != 0:
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1,output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,32)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))
        
        self.layers.append(DownsamplerBlock(32,64))

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))
  
        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            
        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))
                    
        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class Interpolate(nn.Module):
    def __init__(self,size,mode):
        super(Interpolate,self).__init__()
        
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self,x):
        x = self.interp(x,size=self.size,mode=self.mode,align_corners=True)
        return x
        

class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]
        
        b1 = self.branch1(x)
        # b1 = Interpolate(size=(h, w), mode="bilinear")(b1)
        b1 = interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x3 = Interpolate(size=(h // 4, w // 4), mode="bilinear")(x3)
        x3= interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)	
        x2 = self.conv2(x2)
        x = x2 + x3
        # x = Interpolate(size=(h // 2, w // 2), mode="bilinear")(x)
        x= interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
       		
        x1 = self.conv1(x1)
        x = x + x1
        # x = Interpolate(size=(h, w), mode="bilinear")(x)
        x= interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        		
        x = torch.mul(x, mid)

        x = x + b1

        return x


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.apn = APN_Module(in_ch=128, out_ch=19)

    def forward(self, input):
        output = self.apn(input)
        out = interpolate(output, size=(1024, 2048), mode="bilinear", align_corners=True)
        return out


class LEDNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    
            return self.decoder.forward(output)


def oth_lednet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    net = LEDNet(num_classes=num_classes)
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
        oth_lednet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_lednet_cityscapes or weight_count == 925272)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
