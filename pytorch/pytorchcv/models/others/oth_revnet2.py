import torch
import torch.nn as nn
import math
import copy
import warnings
from torch.autograd import Variable
from contextlib import contextmanager

__all__ = ['oth_revnet38', 'oth_revnet110', 'oth_revnet164']


use_context_mans = int(torch.__version__[0]) * 100 + int(torch.__version__[2]) - \
                   (1 if 'a' in torch.__version__ else 0) > 3


@contextmanager
def set_grad_enabled(grad_mode):
    if not use_context_mans:
        yield
    else:
        with torch.set_grad_enabled(grad_mode) as c:
            yield [c]


class ReversibleBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0)  # assert if proper split is possible

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        with set_grad_enabled(False):
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            with warnings.catch_warnings():
                x2var = Variable(x2, requires_grad=False, volatile=True)
            fmr = Fm.forward(x2var).data

            y1 = x1 + fmr
            x1.set_()
            del x1
            with warnings.catch_warnings():
                y1var = Variable(y1, requires_grad=False, volatile=True)
            gmr = Gm.forward(y1var).data
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)
            y1.set_()
            y2.set_()
            del y1, y2

        # save the (empty) input and (non-empty) output variables
        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve weight references
        Fm, Gm = ctx.Fm, ctx.Gm

        # retrieve input and output references
        x, output = ctx.saved_variables
        y1, y2 = torch.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.data.shape[1] % 2 == 0)

        with set_grad_enabled(False):
            # recompute x
            z1_stop = Variable(y1.data, requires_grad=True)
            GWeights = [p for p in Gm.parameters()]
            x2 = y2 - Gm.forward(z1_stop)
            x1 = y1 - Fm.forward(x2)


        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1_ = Variable(x1.data, requires_grad=True)
            x2_ = Variable(x2.data, requires_grad=True)
            x1_.requires_grad = True
            x2_.requires_grad = True

            y1_ = x1_ + Fm.forward(x2_)
            y2_ = x2_ + Gm.forward(y1_)
            y = torch.cat([y1_, y2_], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(y, (x1_, x2_ ) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output)

            GWgrads = dd[2:2+len(GWeights)]
            FWgrads = dd[2+len(GWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            y1_.detach_()
            y2_.detach_()
            del y1_, y2_

        # restore input
        x.data.set_(torch.cat([x1, x2], dim=1).data.contiguous())

        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlockFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0) # assert if possible

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        with set_grad_enabled(False):
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            with warnings.catch_warnings():
                x2var = Variable(x2, requires_grad=False, volatile=True)
            fmr = Fm.forward(x2var).data

            y1 = x1 + fmr
            x1.set_()
            del x1
            with warnings.catch_warnings():
                y1var = Variable(y1, requires_grad=False, volatile=True)
            gmr = Gm.forward(y1var).data
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)
            y1.set_()
            del y1
            y2.set_()
            del y2

        # save the input and output variables
        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        x, output = ctx.saved_variables

        with set_grad_enabled(False):
            y1, y2 = torch.chunk(output, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.data.shape[1] % 2 == 0)
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)
            y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        with set_grad_enabled(True):
            z1_stop = Variable(y1.data, requires_grad=True)

            G_z1 = Gm.forward(z1_stop)
            x2 = y2 - G_z1
            x2_stop = Variable(x2.data, requires_grad=True)

            F_x2 = Fm.forward(x2_stop)
            x1 = y1 - F_x2
            x1_stop = Variable(x1.data, requires_grad=True)

            # restore input
            x.data.set_(torch.cat([x1.data, x2.data], dim=1).contiguous())

            # compute outputs building a sub-graph
            z1 = x1_stop + F_x2
            y2_ = x2_stop + G_z1
            y1_ = z1

            # calculate the final gradients for the weights and inputs
            dd = torch.autograd.grad(y2_, (z1_stop,) + tuple(Gm.parameters()), y2_grad) #, retain_graph=False)
            z1_grad = dd[0] + y1_grad
            GWgrads = dd[1:]

            dd = torch.autograd.grad(y1_, (x1_stop, x2_stop) + tuple(Fm.parameters()), z1_grad, retain_graph=False)

            FWgrads = dd[2:]
            x2_grad = dd[1] + y2_grad
            x1_grad = dd[0]
            grad_input = torch.cat([x1_grad, x2_grad], dim=1)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_

        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, implementation=1, keep_input=False):
        """The ReversibleBlock

        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function

            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Gm is used as a Module)

            implementation : int
                Switch between different Reversible Operation implementations. Default = 1

            keep_input : bool
                Retain the input information, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        """
        super(ReversibleBlock, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.implementation = implementation
        self.keep_input = keep_input

    def forward(self, x):
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation == 0:
            out = ReversibleBlockFunction.apply(*args)
        elif self.implementation == 1:
            out = ReversibleBlockFunction2.apply(*args)
        else:
            raise NotImplementedError("Selected implementation ({}) not implemented..."
                                      .format(self.implementation))

        # clears the input data as it can be reversed on the backward pass
        if not self.keep_input:
            x.data.set_()

        return out

    def inverse(self, y):
        if self.implementation == 0 or self.implementation == 1:
            assert (y.shape[1] % 2 == 0)  # assert if possible

            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute inputs from outputs
            x2 = y2 - self.Gm.forward(y1)
            x1 = y1 - self.Fm.forward(x2)

            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError("Inverse for selected implementation ({}) not implemented..."
                                      .format(self.implementation))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def batch_norm(input):
    """match Tensorflow batch norm settings"""
    return nn.BatchNorm2d(input, momentum=0.99, eps=0.001)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False, *args, **kwargs):
        super(BasicBlock, self).__init__()
        self.basicblock_sub = BasicBlockSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.basicblock_sub(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False, *args, **kwargs):
        super(Bottleneck, self).__init__()
        self.bottleneck_sub = BottleneckSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bottleneck_sub(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(RevBasicBlock, self).__init__()
        if downsample is None and stride == 1:
            Gm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation)
            Fm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation)
            self.revblock = ReversibleBlock(Gm, Fm)
        else:
            self.basicblock_sub = BasicBlockSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            out = self.basicblock_sub(x)
            residual = self.downsample(x)
            out += residual
        else:
            out = self.revblock(x)
        return out


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(RevBottleneck, self).__init__()
        if downsample is None and stride == 1:
            Gm = BottleneckSub(inplanes // 2, planes // 2, stride, noactivation)
            Fm = BottleneckSub(inplanes // 2, planes // 2, stride, noactivation)
            self.revblock = ReversibleBlock(Gm, Fm)
        else:
            self.bottleneck_sub = BottleneckSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            out = self.bottleneck_sub(x)
            residual = self.downsample(x)
            out += residual
        else:
            out = self.revblock(x)
        return out


class BottleneckSub(nn.Module):
    def __init__(self, inplanes, planes, stride=1, noactivation=False):
        super(BottleneckSub, self).__init__()
        self.noactivation = noactivation
        if not self.noactivation:
            self.bn1 = batch_norm(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = batch_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.noactivation:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class BasicBlockSub(nn.Module):
    def __init__(self, inplanes, planes, stride=1, noactivation=False):
        super(BasicBlockSub, self).__init__()
        self.noactivation = noactivation
        if not self.noactivation:
            self.bn1 = batch_norm(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = batch_norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.noactivation:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class RevNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 channels_per_layer=None,
                 strides=None,
                 implementation=0):
        super(RevNet, self).__init__()
        assert(len(channels_per_layer) == len(layers) + 1)
        self.channels_per_layer = channels_per_layer
        self.strides = strides
        self.implementation = implementation
        self.inplanes = channels_per_layer[0]  # 64 by default

        init_kernel_size = 3
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=init_kernel_size,
            stride=strides[0],
            padding=(init_kernel_size - 1) // 2,
            bias=False)
        self.bn1 = batch_norm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, channels_per_layer[1], layers[0], stride=strides[1], noactivation=True)
        self.layer2 = self._make_layer(block, channels_per_layer[2], layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, channels_per_layer[3], layers[2], stride=strides[3])
        self.has_4_layers = len(layers) >= 4
        if self.has_4_layers:
            self.layer4 = self._make_layer(block, channels_per_layer[4], layers[3], stride=strides[4])
        self.bn_final = batch_norm(self.inplanes)  # channels_per_layer[-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels_per_layer[-1] * block.expansion, num_classes)

        self.configure()
        self.init_weights()

    def init_weights(self):
        """Initialization using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

    def configure(self):
        """Initialization specific configuration settings"""
        for m in self.modules():
            if isinstance(m, ReversibleBlock):
                m.implementation = self.implementation
            elif isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1
                m.eps = 1e-05

    def _make_layer(self, block, planes, blocks, stride=1, noactivation=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                batch_norm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, noactivation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.has_4_layers:
            x = self.layer4(x)
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def oth_revnet38(pretrained=False, **kwargs):
    net = RevNet(
        block=RevBasicBlock,
        layers=[3, 3, 3],
        num_classes=1000,
        channels_per_layer=[32, 32, 64, 112],
        strides=[1, 1, 2, 2],
        implementation=0)
    return net


def oth_revnet110(pretrained=False, **kwargs):
    net = RevNet(
        block=RevBasicBlock,
        layers=[9, 9, 9],
        num_classes=1000,
        channels_per_layer=[32, 32, 64, 128],
        strides=[1, 1, 2, 2],
        implementation=0)
    return net


def oth_revnet164(pretrained=False, **kwargs):
    net = RevNet(
        block=RevBottleneck,
        layers=[9, 9, 9],
        num_classes=1000,
        channels_per_layer=[32, 32, 64, 128],
        strides=[1, 1, 2, 2],
        implementation=0)
    return net


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
        oth_revnet38,
        oth_revnet110,
        oth_revnet164,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_revnet38 or weight_count == 685864)
        assert (model != oth_revnet110 or weight_count == 1982600)
        assert (model != oth_revnet164 or weight_count == 2491656)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
