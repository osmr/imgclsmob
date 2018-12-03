import torch.nn as nn
import torch.nn.functional as F


class Hourglass(nn.Module):
    """
    A hourglass block.

    Parameters:
    ----------
    down_seq : nn.Sequential
        Down modules as sequential.
    up_seq : nn.Sequential
        Up modules as sequential.
    skip_seq : nn.Sequential
        Skip modules as sequential.
    merge_type : str, default 'add'
        Type of concatenation of up and skip outputs.
    return_first_skip : bool, default False
        Whether return the first skip connection output.
    """
    def __init__(self,
                 down_seq,
                 up_seq,
                 skip_seq,
                 merge_type="add",
                 return_first_skip=False):
        super(Hourglass, self).__init__()
        assert (len(up_seq) == len(down_seq))
        assert (len(skip_seq) == len(down_seq))
        assert (merge_type in ["add"])
        self.down_seq = down_seq
        self.up_seq = up_seq
        self.skip_seq = skip_seq
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.depth = len(down_seq)

    def forward(self, x, **kwargs):
        y = None
        down_outs = [x]
        for down_module in self.down_seq._modules.values():
            x = down_module(x)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module = self.skip_seq[self.depth - i]
                y = skip_module(y)
                if self.merge_type == "add":
                    x = x + y
            if i != len(down_outs) - 1:
                up_module = self.up_seq[self.depth - 1 - i]
                x = up_module(x)
        if self.return_first_skip:
            return x, y
        else:
            return x


def conv1x1(in_channels,
            out_channels,
            stride=1,
            bias=False):
    """
    Convolution 1x1 layer.

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
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias)


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act_type="relu",
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            if act_type == "relu":
                self.activ = nn.ReLU(inplace=True)
            elif act_type == "relu6":
                self.activ = nn.ReLU6(inplace=True)
            else:
                raise NotImplementedError()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=3,
                  bias=False,
                  act_type="relu",
                  activate=True):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        act_type=act_type,
        activate=activate)


class PreConvBlock(nn.Module):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 return_preact=False):
        super(PreConvBlock, self).__init__()
        self.return_preact = return_preact

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv1x1_block(in_channels,
                      out_channels,
                      stride=1,
                      return_preact=False):
    """
    1x1 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        return_preact=return_preact)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      stride,
                      return_preact=False):
    """
    3x3 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        return_preact=return_preact)


class PreResBottleneck(nn.Module):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            return_preact=True)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = pre_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x, x_pre_activ


class ResBlock(nn.Module):
    """
    Residual block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResBlock, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = PreResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


def interpolate(x, size):
    return F.interpolate(
        input=x,
        size=size,
        mode='bilinear',
        align_corners=True)


class InterpolationBlock(nn.Module):
    """
    Interpolation block.

    Parameters:
    ----------
    size : tuple of 2 int
        Output spatial size.
    """
    def __init__(self,
                 size):
        super(InterpolationBlock, self).__init__()
        self.size = size

    def forward(self, x, **kwargs):
        return F.interpolate(
            input=x,
            size=self.size,
            mode='bilinear',
            align_corners=True)


class InterpolationBlock2(nn.Module):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    """
    def __init__(self,
                 scale_factor):
        super(InterpolationBlock2, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x, **kwargs):
        return F.interpolate(
            input=x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True)


class DoubleSkipBlock(nn.Module):
    """
    Double skip connection block.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DoubleSkipBlock, self).__init__()
        self.skip1 = ResBlock(in_channels, out_channels)

    def forward(self, x, **kwargs):
        x = x + self.skip1(x)
        return x


class AttentionModule_stage1(nn.Module):
    # input size is 56*56
    def __init__(self,
                 in_channels,
                 out_channels):
        super(AttentionModule_stage1, self).__init__()
        scale_factor = 2

        self.first_residual_blocks = ResBlock(in_channels, out_channels)

        down_seq = nn.Sequential()
        down_seq.add_module('down1', nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(in_channels, out_channels)
        ))
        down_seq.add_module('down2', nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(in_channels, out_channels)
        ))
        down_seq.add_module('down3', nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Sequential(
                ResBlock(in_channels, out_channels),
                ResBlock(in_channels, out_channels)
            )
        ))
        up_seq = nn.Sequential()
        up_seq.add_module('up1', nn.Sequential(
            ResBlock(in_channels, out_channels),
            InterpolationBlock2(scale_factor)))
        up_seq.add_module('up2', nn.Sequential(
            ResBlock(in_channels, out_channels),
            InterpolationBlock2(scale_factor)))
        up_seq.add_module('up3', InterpolationBlock2(scale_factor))
        skip_seq = nn.Sequential()
        skip_seq.add_module('skip1', nn.Sequential(
            ResBlock(in_channels, out_channels),
            ResBlock(in_channels, out_channels)))
        skip_seq.add_module('skip2', DoubleSkipBlock(in_channels, out_channels))
        skip_seq.add_module('skip3', DoubleSkipBlock(in_channels, out_channels))
        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            return_first_skip=True)

        self.softmax6_blocks = nn.Sequential(
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            nn.Sigmoid()
        )

        self.last_blocks = ResBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)

        out_interp1, out_trunk = self.hg(x)

        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk

        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage2(nn.Module):
    # input image size is 28*28
    def __init__(self,
                 in_channels,
                 out_channels,
                 size1=(28, 28),
                 size2=(14, 14)):
        super(AttentionModule_stage2, self).__init__()
        self.size1 = size1
        self.size2 = size2

        self.first_residual_blocks = ResBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResBlock(in_channels, out_channels),
            ResBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = nn.Sequential(
            ResBlock(in_channels, out_channels),
            ResBlock(in_channels, out_channels)
        )

        # self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax3_blocks = ResBlock(in_channels, out_channels)

        # self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax4_blocks = nn.Sequential(
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            nn.Sigmoid()
        )

        self.last_blocks = ResBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = interpolate(out_softmax2, size=self.size2) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = interpolate(out_softmax3, size=self.size1) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage3(nn.Module):
    # input image size is 14*14
    def __init__(self,
                 in_channels,
                 out_channels,
                 size1=(14, 14)):
        super(AttentionModule_stage3, self).__init__()
        self.size1 = size1

        self.first_residual_blocks = ResBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResBlock(in_channels, out_channels),
            ResBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResBlock(in_channels, out_channels),
            ResBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            nn.Sigmoid()
        )

        self.last_blocks = ResBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = interpolate(out_softmax1, size=self.size1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class ResAttInitBlock(nn.Module):
    """
    ResAttNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResAttInitBlock, self).__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResidualAttentionModel_56(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_56, self).__init__()
        self.init_block = ResAttInitBlock(
            in_channels=3,
            out_channels=64)
        self.residual_block1 = ResBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResBlock(1024, 2048, 2)
        self.residual_block5 = ResBlock(2048, 2048)
        self.residual_block6 = ResBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,1000)

    def forward(self, x):
        out = self.init_block(x)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_92(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_92, self).__init__()
        self.init_block = ResAttInitBlock(
            in_channels=3,
            out_channels=64)
        self.residual_block1 = ResBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResBlock(1024, 2048, 2)
        self.residual_block5 = ResBlock(2048, 2048)
        self.residual_block6 = ResBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,1000)

    def forward(self, x):
        out = self.init_block(x)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def oth_resattnet56(pretrained=False, **kwargs):
    return ResidualAttentionModel_56(**kwargs)


def oth_resattnet92(pretrained=False, **kwargs):
    return ResidualAttentionModel_92(**kwargs)


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
        oth_resattnet56,
        oth_resattnet92,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net = AttentionModule_stage1(256, 256)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_resattnet56 or weight_count == 31810728)
        assert (model != oth_resattnet92 or weight_count == 52466344)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()


