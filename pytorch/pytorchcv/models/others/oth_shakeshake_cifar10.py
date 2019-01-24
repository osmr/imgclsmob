import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# see https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
# see https://discuss.pytorch.org/t/why-input-is-tensor-in-the-forward-function-when-extending-torch-autograd/9039
class ShakeShake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, alpha, beta=None):
        ctx.save_for_backward(input1, input2, alpha, beta)
        out = alpha * input1 + (1 - alpha) * input2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, alpha, beta = ctx.saved_tensors
        grad_input1 = beta * grad_output
        grad_input2 = (1 - beta) * grad_output
        return grad_input1, grad_input2, None, None


class SkippingBranch(nn.Module):
    def __init__(self, inplanes, stride=2):
        super(SkippingBranch, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, 
                              padding=0, bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, 
                              padding=0, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=stride, padding=0)    

    def forward(self, x):
        out1 = self.conv1(self.avg_pool(x))
        shift_x = x[:, :, 1:, 1:]
        shift_x= F.pad(shift_x, (0, 1, 0, 1))
        out2 = self.conv2(self.avg_pool(shift_x))
        out = torch.cat([out1, out2], dim=1)
        return out


class ResidualBranch(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBranch, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.bn1(self.conv1(F.relu(x, inplace=False)))
        out = self.bn2(self.conv2(F.relu(out, inplace=False)))
        return out


class ShakeBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ShakeBlock, self).__init__()
        self.residual_branch1 = ResidualBranch(inplanes, planes, stride)
        self.residual_branch2 = ResidualBranch(inplanes, planes, stride)

        if inplanes != planes:
            self.skipping_branch = SkippingBranch(inplanes, stride)
        else:
            self.skipping_branch = nn.Sequential()

        self.shake_shake = ShakeShake.apply

    def forward(self, x):
        residual = x
        out1 = self.residual_branch1(x)
        out2 = self.residual_branch2(x)
        
        batch_size = out1.size(0)
        if self.training:        
            alpha = torch.rand(batch_size).to(device)
            beta = torch.rand(batch_size).to(device)
            beta = beta.view(batch_size, 1, 1, 1)
            alpha = alpha.view(batch_size, 1, 1, 1)
            out = self.shake_shake(out1, out2, alpha, beta)
        else:
            alpha = torch.Tensor([0.5]).to(device)
            out = self.shake_shake(out1, out2, alpha)

        skip = self.skipping_branch(residual)
        return out + skip


class ShakeResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ShakeResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage(block, 32, 4, stride=1)
        self.stage2 = self._make_stage(block, 64, 4, stride=2) 
        self.stage3 = self._make_stage(block, 128, 4, stride=2)  
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x


def oth_shakeshake_cifar10(pretrained=False, **kwargs):
    model = ShakeResNet(ShakeBlock, **kwargs) 
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
        oth_shakeshake_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_shakeshake_cifar10 or weight_count == 2922714)

        x = Variable(torch.randn(1, 3, 32, 32))
        y = net(x)
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
