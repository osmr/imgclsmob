__all__ = ['oth_ibppose']

import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    """Residual Block modified by us"""

    def __init__(self, ins, outs, bn=True, relu=True):
        super(Residual, self).__init__()
        self.relu_flag = relu
        self.convBlock = nn.Sequential(
            nn.Conv2d(ins, outs//2, 1, bias=False),
            nn.BatchNorm2d(outs//2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(outs // 2, outs // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outs // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(outs // 2, outs, 1, bias=False),
            nn.BatchNorm2d(outs),
        )
        if ins != outs:
            self.skipConv = nn.Sequential(
                nn.Conv2d(ins, outs, 1, bias=False),
                nn.BatchNorm2d(outs)
            )
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual  # Bn layer is in the middle, so we can do in-plcae += here

        if self.relu_flag:
            x = self.relu(x)
            return x
        else:
            return x


class Conv(nn.Module):
    # conv block used in hourglass
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True, dropout=False, dialated=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 换成 Leak Relu减缓神经元死亡现象
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False, dilation=1)
            # Different form TF, momentum default in Pytorch is 0.1, which means the decay rate of old running value
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True, dilation=1)

    def forward(self, x):
        # examine the input channel equals the conve kernel channel
        assert x.size()[1] == self.inp_dim, "input channel {} dese not fit kernel channel {}".format(x.size()[1],
                                                                                                     self.inp_dim)
        if self.dropout:  # comment these two lines if we do not want to use Dropout layers
            # p: probability of an element to be zeroed
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)  # 直接注释掉这一行，如果我们不想使用Dropout

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DilatedConv(nn.Module):
    """
    Dilated convolutional layer of stride=1 only!
    """
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True, dropout=False, dialation=3):
        super(DilatedConv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 换成 Leak Relu减缓神经元死亡现象
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=dialation, bias=False, dilation=dialation)
            # Different form TF, momentum default in Pytorch is 0.1, which means the decay rate of old running value
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=dialation, bias=True, dilation=dialation)

    def forward(self, x):
        # examine the input channel equals the conve kernel channel
        assert x.size()[1] == self.inp_dim, "input channel {} dese not fit kernel channel {}".format(x.size()[1],
                                                                                                     self.inp_dim)
        if self.dropout:  # comment these two lines if we do not want to use Dropout layers
            # p: probability of an element to be zeroed
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)  # 直接注释掉这一行，如果我们不想使用Dropout

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Backbone(nn.Module):
    """
    Input Tensor: a batch of images with shape (N, C, H, W)
    """
    def __init__(self, nFeat=256, inplanes=3, resBlock=Residual, dilatedBlock=DilatedConv):
        super(Backbone, self).__init__()
        self.nFeat = nFeat
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.dilation = nn.Sequential(
            dilatedBlock(128, 128, dialation=3),
            dilatedBlock(128, 128, dialation=3),
            dilatedBlock(128, 128, dialation=4),
            dilatedBlock(128, 128, dialation=4),
            dilatedBlock(128, 128, dialation=5),
            dilatedBlock(128, 128, dialation=5),
        )

    def forward(self, x):
        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x1 = self.dilation(x)
        concat_merge = torch.cat([x, x1], dim=1)  # (N, C1+C2, H, W)

        return concat_merge


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""
    def __init__(self, depth, nFeat, increase=128, bn=False, resBlock=Residual, convBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth  # oder number
        self.nFeat = nFeat  # input and output channels
        self.increase = increase  # increased channels while the depth grows
        self.bn = bn
        self.resBlock = resBlock
        self.convBlock = convBlock
        # will execute when instantiate the Hourglass object, prepare network's parameters
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)  # no learning parameters, can be used any times repeatedly
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # no learning parameters  # FIXME: 改成反卷积？

    def _make_single_residual(self, depth_id):
        # the innermost conve layer, return as a layer item
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * (depth_id + 1),
                             bn=self.bn)                            # ###########  Index: 4

    def _make_lower_residual(self, depth_id):
        # return as a list
        pack_layers = [self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                     bn=self.bn),                                     # ######### Index: 0
                       self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * (depth_id + 1),
                                                                                                  # ######### Index: 1
                                     bn=self.bn),
                       self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * depth_id,
                                                                                                   # ######### Index: 2
                                     bn=self.bn),
                       self.convBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                     # ######### Index: 3
                                     bn=self.bn),  # 添加一个Conv精细化上采样的特征图?
                       ]
        return pack_layers

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            #  skip path; up_residual_block; down_residual_block_path,
            # 0 ~ n-2 (except the outermost n-1 order) need 3 residual blocks
            res = self._make_lower_residual(i)  # type:list
            if i == (self.depth - 1):  # the deepest path (i.e. the longest path) need 4 residual blocks
                res.append(self._make_single_residual(i))  # list append an element
            hg.append(nn.ModuleList(res))  # pack conve layers of  every oder of hourglass block
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, depth_id, x, up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
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
        up2 = self.upsample(low3)
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


class SELayer(nn.Module):
    def __init__(self, inp_dim, reduction=16):
        """
        Squeeze and Excitation
        :param inp_dim: the channel of input tensor
        :param reduction: channel compression ratio
        :return output the tensor with the same shape of input
        """
        # assert inp_dim > reduction, f"Make sure your input channel bigger than reduction which equals to {reduction}"
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(inp_dim, inp_dim // reduction),
                nn.LeakyReLU(inplace=True),  # Relu
                nn.Linear(inp_dim // reduction, inp_dim),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    # def forward(self, x):  # 去掉Selayer
    #     return x


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        # Regress 5 different scales of heatmaps per stack
        self.before_regress = nn.ModuleList(
            [nn.Sequential(Conv(inp_dim + i * increase, inp_dim, 3, bn=bn, dropout=False),
                           Conv(inp_dim, inp_dim, 3, bn=bn, dropout=False),
                           # ##################### Channel Attention layer  #####################
                           SELayer(inp_dim),
                           ) for i in range(5)])

    def forward(self, fms):
        assert len(fms) == 5, "hourglass output {} tensors,but 5 scale heatmaps are supervised".format(len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        # self.pre = nn.Sequential(
        #     Conv(3, 64, 7, 2, bn=bn),
        #     Conv(64, 128, bn=bn),
        #     nn.MaxPool2d(2, 2),
        #     Conv(128, 128, bn=bn),
        #     Conv(128, inp_dim, bn=bn)
        # )
        self.pre = Backbone(nFeat=inp_dim)  # It doesn't affect the results regardless of which self.pre is used
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase, bn=bn) for _ in range(nstack)])
        # predict 5 different scales of heatmpas per stack, keep in mind to pack the list using ModuleList.
        # Notice: nn.ModuleList can only identify Module subclass! Thus, we must pack the inner layers in ModuleList.
        # TODO: change the outs layers, Conv(inp_dim + j * increase, oup_dim, 1, relu=False, bn=False)
        self.outs = nn.ModuleList(
            [nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for i in
             range(nstack)])

        # TODO: change the merge layers, Merge(inp_dim + j * increase, inp_dim + j * increase)
        self.merge_features = nn.ModuleList(
            [nn.ModuleList([Merge(inp_dim, inp_dim + j * increase, bn=bn) for j in range(5)]) for i in
             range(nstack - 1)])
        self.merge_preds = nn.ModuleList(
            [nn.ModuleList([Merge(oup_dim, inp_dim + j * increase, bn=bn) for j in range(5)]) for i in range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        # Input Tensor: a batch of images within [0,1], shape=(N, H, W, C). Pre-processing was done in data generator
        # x = imgs.permute(0, 3, 1, 2)  # Permute the dimensions of images to (N, C, H, W)
        x = imgs
        x = self.pre(x)
        pred = []
        # loop over stack
        for i in range(self.nstack):
            preds_instack = []
            # return 5 scales of feature maps
            hourglass_feature = self.hourglass[i](x)

            if i == 0:  # cache for smaller feature maps produced by hourglass block
                features_cache = [torch.zeros_like(hourglass_feature[scale]) for scale in range(5)]

            else:  # residual connection across stacks
                #  python里面的+=, ，*=也是in-place operation,需要注意
                hourglass_feature = [hourglass_feature[scale] + features_cache[scale] for scale in range(5)]
            # feature maps before heatmap regression
            features_instack = self.features[i](hourglass_feature)

            for j in range(5):  # handle 5 scales of heatmaps
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])  # input tensor for next stack
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])

                    else:
                        # reset the res caches
                        features_cache[j] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        # returned list shape: [nstack * [batch*128*128, batch*64*64, batch*32*32, batch*16*16, batch*8*8]]z
        # return pred
        return pred[-1][0]

    def _initialize_weights(self):
        for m in self.modules():
            # 卷积的初始化方法
            if isinstance(m, nn.Conv2d):
                # TODO: 使用正态分布进行初始化（0, 0.01) 网络权重看看
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He kaiming 初始化, 方差为2/n. math.sqrt(2. / n) 或者直接使用现成的nn.init中的函数。在这里会梯度爆炸
                m.weight.data.normal_(0, 0.001)    # # math.sqrt(2. / n)
                # torch.nn.init.kaiming_normal_(m.weight)
                # bias都初始化为0
                if m.bias is not None:  # 当有BN层时，卷积层Con不加bias！
                    m.bias.data.zero_()
            # batchnorm使用全1初始化 bias全0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # todo: 0.001?
                # m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def oth_ibppose(pretrained=False, num_classes=3, in_channels=3, **kwargs):
    model = PoseNet(4, 256, 50, bn=True, **kwargs)
    return model


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


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


def _test():
    import torch

    pretrained = False

    models = [
        oth_ibppose,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_ibppose or weight_count == 128998760)

        x = torch.randn(14, 3, 256, 256)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, 50, 64, 64))


if __name__ == "__main__":
    _test()
