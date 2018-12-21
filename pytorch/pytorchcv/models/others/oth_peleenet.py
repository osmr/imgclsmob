import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['oth_peleenet']


class Conv_bn_relu(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=3,
                 stride=1,
                 pad=1,
                 use_relu = True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        out = self.convs(x)
        return out


class StemBlock(nn.Module):
    def __init__(self, inp=3,num_init_features=32):
        super(StemBlock, self).__init__()



        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)

        self.stem_2a = Conv_bn_relu(num_init_features,int(num_init_features/2),1,1,0)

        self.stem_2b = Conv_bn_relu(int(num_init_features/2), num_init_features, 3, 2, 1)

        self.stem_2p = nn.MaxPool2d(kernel_size=2,stride=2)

        self.stem_3 = Conv_bn_relu(num_init_features*2,num_init_features,1,1,0)



    def forward(self, x):
        stem_1_out  = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(torch.cat((stem_2b_out,stem_2p_out),1))

        return out


class DenseBlock(nn.Module):
    def __init__(self, inp,inter_channel,growth_rate):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp,inter_channel,1,1,0)
        self.cb1_b = Conv_bn_relu(inter_channel,growth_rate,3,1,1)

        self.cb2_a = Conv_bn_relu(inp,inter_channel,1,1,0)
        self.cb2_b = Conv_bn_relu(inter_channel,growth_rate,3,1,1)
        self.cb2_c = Conv_bn_relu(growth_rate,growth_rate,3,1,1)


    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x,cb1_b_out,cb2_c_out),1)

        return out


class TransitionBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 with_pooling= True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp,oup,1,1,0),
                                    nn.AvgPool2d(kernel_size=2,stride=2))
        else:
            self.tb = Conv_bn_relu(inp,oup,1,1,0)

    def forward(self, x):
        out = self.tb(x)
        return out


class PeleeNet(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 num_init_features=32,
                 growthRate=32,
                 nDenseBlocks = [3,4,8,6],
                 bottleneck_width=[1,2,4,4]):
        super(PeleeNet, self).__init__()


        self.stage = nn.Sequential()
        self.num_classes = num_classes
        self.num_init_features = num_init_features

        inter_channel =list()
        total_filter =list()
        dense_inp = list()

        self.half_growth_rate = int(growthRate / 2)
        
        # building stemblock
        self.stage.add_module('stage_0', StemBlock(3,num_init_features))

        #
        for i, b_w in enumerate(bottleneck_width):

            # print("i={}, b_w={}".format(i, b_w))

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                dense_inp.append(self.num_init_features)
            else:
                total_filter.append(total_filter[i-1] + growthRate * nDenseBlocks[i])
                dense_inp.append(total_filter[i-1])

            if i == len(nDenseBlocks)-1:
                with_pooling = False
            else:
                with_pooling = True

        # building middle stageblock
            self.stage.add_module('stage_{}'.format(i+1),self._make_dense_transition(
                dense_inp[i],
                total_filter[i],
                inter_channel[i],
                nDenseBlocks[i],
                with_pooling=with_pooling))
            
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(total_filter[len(nDenseBlocks)-1], self.num_classes)
        )

        self._initialize_weights()


    def _make_dense_transition(self, dense_inp,total_filter, inter_channel, ndenseblocks,with_pooling= True):
        layers = []

        for i in range(ndenseblocks):
            # print("_make: i={}, dense_inp={}, inter_channel={}".format(i, dense_inp, inter_channel))
            layers.append(DenseBlock(dense_inp, inter_channel,self.half_growth_rate))
            dense_inp += self.half_growth_rate * 2

        # print("_make-trans: dense_inp={}, total_filter={}".format(dense_inp, total_filter))
        #Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp,total_filter,with_pooling))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stage(x)

        # global average pooling layer
        x = F.avg_pool2d(x,kernel_size=7)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # out = F.log_softmax(x,dim=1)

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



# if __name__ == '__main__':
#     p = PeleeNet(num_classes=1000)
#     input = torch.autograd.Variable(torch.ones(1, 3, 224, 224))
#     output = p(input)
#
#     print(output.size())
#
#     # torch.save(p.state_dict(), 'peleenet.pth.tar')

def oth_peleenet(pretrained=False, **kwargs):
    return PeleeNet(**kwargs)


def load_model(net,
               file_path,
               ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


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
        oth_peleenet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_peleenet or weight_count == 2802248)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()















