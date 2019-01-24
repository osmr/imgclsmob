import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ShakeDrop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, drop_factor, beta=None):
        ctx.save_for_backward(input, alpha, drop_factor, beta)
        out = (drop_factor + alpha + alpha * (1 - drop_factor)) * input
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha, drop_factor, beta = ctx.saved_tensors
        grad_input = (drop_factor + beta + beta * (1 - drop_factor)) * grad_output
        return grad_input, None, None, None


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, drop_prob=1.0, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob
        self.shake_drop = ShakeDrop.apply

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        
        batch_size = x.size(0)
        if self.training:
            alpha = torch.Tensor(batch_size).uniform_(-1, 1).to(device)
            beta = torch.rand(batch_size).to(device)
            drop_prob = torch.Tensor(batch_size).fill_(self.drop_prob).to(device)
            m = torch.distributions.Bernoulli(drop_prob)
            drop_factor = m.sample().to(device)
            # check whether drop probability and drop factor act correctly
            # print(self.drop_prob, drop_factor)

            alpha = alpha.view(batch_size, 1, 1, 1)
            beta = beta.view(batch_size, 1, 1, 1)
            drop_factor = drop_factor.view(batch_size, 1, 1, 1)

            out = self.shake_drop(out, alpha, drop_factor, beta)
        else:
            alpha = torch.Tensor(batch_size).fill_(0.0).to(device)
            drop_factor = torch.Tensor(batch_size).fill_(self.drop_prob).to(device)
            alpha = alpha.view(batch_size, 1, 1, 1)
            drop_factor = drop_factor.view(batch_size, 1, 1, 1)
            out = self.shake_shake(out, alpha, drop_factor)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.FloatTensor(batch_size, residual_channel - shortcut_channel, 
                                        featuremap_size[0], featuremap_size[1]).fill_(0).to(device) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class ShakeDropNet(nn.Module):
    def __init__(self, depth, alpha, num_classes):
        super(ShakeDropNet, self).__init__()      
        self.inplanes = 16

        n = int((depth - 2) / 6)
        block = BasicBlock

        self.addrate = alpha / (3*n*1.0)
        self.drop_prob = 1.0
        self.addprob = - 0.5 / (3*n*1.0)
        # print(self.addprob)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        self.drop_prob = self.drop_prob + self.addprob
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, self.drop_prob, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            self.drop_prob = self.drop_prob + self.addprob
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1, self.drop_prob))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def oth_shakedrop_cifar10(pretrained=False, **kwargs):
    model = ShakeDropNet(depth=110, alpha=84, num_classes=10, **kwargs)
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
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_shakedrop_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_shakedrop_cifar10 or weight_count == 3904446)

        x = Variable(torch.randn(1, 3, 32, 32))
        y = net(x)
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
