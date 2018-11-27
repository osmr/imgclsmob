import torch.nn as nn
import math


def conv_bn(inp, oup, stride ):
    return nn.Sequential(
        nn.Conv2d(inp, oup,kernel_size= 3, stride= stride, padding= 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size = 1, stride= 1, padding= 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return output


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio,kernel_size = 1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            #permutation
            PermutationBlock(groups=2),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size =3, stride= stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size =1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(oup),
            # permutation
            PermutationBlock(groups= int(round((oup/2)))),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class IGCV3(nn.Module):
    def __init__(self, args):
        super(IGCV3, self).__init__()
        s1, s2 = 2, 2
        if args.downsampling == 16:
            s1, s2 = 2, 1
        elif args.downsampling == 8:
            s1, s2 = 1, 1

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 4, s2],
            [6, 32, 6, 2],
            [6, 64, 8, 2],
            [6, 96, 6, 1],
            [6, 160, 6, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert args.img_height % 32 == 0
        input_channel = int(32 * args.width_multiplier)
        self.last_channel = int(1280 * args.width_multiplier) if args.width_multiplier > 1.0 else 1280
        self.features = [conv_bn(inp =3, oup =input_channel, stride = s1)]
        #  Layers from 1 to 7
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * args.width_multiplier)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(kernel_size = (args.img_height // args.downsampling, args.img_width // args.downsampling)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel,args.num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def oth_igcv3(**kwargs):
    args = {
        'downsampling': 16
    }

    return IGCV3(args)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_igcv3,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_igcv3 or weight_count == 4308816)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
