import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import model_zoo


def define_model(params):
    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'],
                        params[base + '.bias'], stride, pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    def f(input, params):
        o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f


def _test():
    params = torch.load(
        '../imgclsmob_data/pt-wrn50_2-oth/wide-resnet-50-2-export-5ae25d50.pth',
        map_location='cpu')

    # convert numpy arrays to torch Variables
    for k, v in sorted(params.items()):
        print(k, tuple(v.shape))
        params[k] = Variable(v, requires_grad=True)

    print('\nTotal parameters:', sum(v.numel() for v in params.values()))

    f = define_model(params)

    inputs = torch.randn(1, 3, 224, 224)
    y = f(Variable(inputs), params)
    print(y)


if __name__ == "__main__":
    _test()
