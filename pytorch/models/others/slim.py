#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

try:
    import caffe
    from caffe import layers as L
    from caffe import params as P
except ImportError:
    pass


def g_name(g_name, m):
    m.g_name = g_name
    return m


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = x.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.ShuffleChannel(layer, group=self.groups)
        caffe_net[self.g_name] = layer
        return layer


def channel_shuffle(name, groups):
    return g_name(name, ChannelShuffle(groups))


class Permute(nn.Module):
    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def forward(self, x):
        x = x.permute(*self.order).contiguous()
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.Permute(layer, order=list(self.order))
        caffe_net[self.g_name] = layer
        return layer


def permute(name, order):
    return g_name(name, Permute(order))


class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        assert self.axis == 1
        x = x.reshape(x.shape[0], -1)
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.Flatten(layer, axis=self.axis)
        caffe_net[self.g_name] = layer
        return layer


def flatten(name, axis):
    return g_name(name, Flatten(axis))


def generate_caffe_prototxt(m, caffe_net, layer):
    if hasattr(m, 'generate_caffe_prototxt'):
        return m.generate_caffe_prototxt(caffe_net, layer)

    if isinstance(m, nn.Sequential):
        for module in m:
            layer = generate_caffe_prototxt(module, caffe_net, layer)
        return layer

    if isinstance(m, nn.Conv2d):
        if m.bias is None:
            param=[dict(lr_mult=1, decay_mult=1)]
        else:
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)]
        assert m.dilation[0] == m.dilation[1]
        convolution_param=dict(
            num_output=m.out_channels,
            group=m.groups, bias_term=(m.bias is not None),
            weight_filler=dict(type='msra'),
            dilation=m.dilation[0],
        )
        if m.kernel_size[0] == m.kernel_size[1]:
            convolution_param['kernel_size'] = m.kernel_size[0]
        else:
            convolution_param['kernel_h'] = m.kernel_size[0]
            convolution_param['kernel_w'] = m.kernel_size[1]
        if m.stride[0] == m.stride[1]:
            convolution_param['stride'] = m.stride[0]
        else:
            convolution_param['stride_h'] = m.stride[0]
            convolution_param['stride_w'] = m.stride[1]
        if m.padding[0] == m.padding[1]:
            convolution_param['pad'] = m.padding[0]
        else:
            convolution_param['pad_h'] = m.padding[0]
            convolution_param['pad_w'] = m.padding[1]
        layer = L.Convolution(
            layer,
            param=param,
            convolution_param=convolution_param,
        )
        caffe_net.tops[m.g_name] = layer
        return layer

    if isinstance(m, nn.ConvTranspose2d):
        if m.bias is None:
            param=[dict(lr_mult=1, decay_mult=1)]
        else:
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)]
        assert m.dilation[0] == m.dilation[1]
        convolution_param=dict(
            num_output=m.out_channels,
            group=m.groups, bias_term=(m.bias is not None),
            weight_filler=dict(type='msra'),
            dilation=m.dilation[0],
        )
        if m.kernel_size[0] == m.kernel_size[1]:
            convolution_param['kernel_size'] = m.kernel_size[0]
        else:
            convolution_param['kernel_h'] = m.kernel_size[0]
            convolution_param['kernel_w'] = m.kernel_size[1]
        if m.stride[0] == m.stride[1]:
            convolution_param['stride'] = m.stride[0]
        else:
            convolution_param['stride_h'] = m.stride[0]
            convolution_param['stride_w'] = m.stride[1]
        if m.padding[0] == m.padding[1]:
            convolution_param['pad'] = m.padding[0]
        else:
            convolution_param['pad_h'] = m.padding[0]
            convolution_param['pad_w'] = m.padding[1]
        layer = L.Deconvolution(
            layer,
            param=param,
            convolution_param=convolution_param,
        )
        caffe_net.tops[m.g_name] = layer
        return layer

    if isinstance(m, nn.BatchNorm2d):
        layer = L.BatchNorm(
            layer, in_place=True,
            param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        )
        caffe_net[m.g_name] = layer
        if m.affine:
            layer = L.Scale(
                layer, in_place=True, bias_term=True,
                filler=dict(type='constant', value=1), bias_filler=dict(type='constant', value=0),
                param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
            )
            caffe_net[m.g_name + '/scale'] = layer
        return layer

    if isinstance(m, nn.ReLU):
        layer = L.ReLU(layer, in_place=True)
        caffe_net.tops[m.g_name] = layer
        return layer

    if isinstance(m, nn.PReLU):
        layer = L.PReLU(layer)
        caffe_net.tops[m.g_name] = layer
        return layer

    if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
        if isinstance(m, nn.AvgPool2d):
            pooling_param = dict(pool=P.Pooling.AVE)
        else:
            pooling_param = dict(pool=P.Pooling.MAX)
        if isinstance(m.kernel_size, tuple) or isinstance(m.kernel_size, list):
            pooling_param['kernel_h'] = m.kernel_size[0]
            pooling_param['kernel_w'] = m.kernel_size[1]
        else:
            pooling_param['kernel_size'] = m.kernel_size
        if isinstance(m.stride, tuple) or isinstance(m.stride, list):
            pooling_param['stride_h'] = m.stride[0]
            pooling_param['stride_w'] = m.stride[1]
        else:
            pooling_param['stride'] = m.stride
        if isinstance(m.padding, tuple) or isinstance(m.padding, list):
            pooling_param['pad_h'] = m.padding[0]
            pooling_param['pad_w'] = m.padding[1]
        else:
            pooling_param['pad'] = m.padding
        layer = L.Pooling(layer, pooling_param=pooling_param)
        caffe_net.tops[m.g_name] = layer
        return layer
    raise Exception("Unknow module '%s' to generate caffe prototxt." % m)


def convert_pytorch_to_caffe(torch_net, caffe_net):
    for name, m in torch_net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            print('convert conv:', name, m.g_name, m)
            caffe_net.params[m.g_name][0].data[...] = m.weight.data.cpu().numpy()
            if m.bias is not None:
                caffe_net.params[m.g_name][1].data[...] = m.bias.data.cpu().numpy()
        if isinstance(m, nn.BatchNorm2d):
            print('convert bn:', name, m.g_name, m)
            caffe_net.params[m.g_name][0].data[...] = m.running_mean.cpu().numpy()
            caffe_net.params[m.g_name][1].data[...] = m.running_var.cpu().numpy()
            caffe_net.params[m.g_name][2].data[...] = 1
            if m.affine:
                caffe_net.params[m.g_name + '/scale'][0].data[...] = m.weight.data.cpu().numpy()
                caffe_net.params[m.g_name + '/scale'][1].data[...] = m.bias.data.cpu().numpy()


def conv_bn_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        g_name(name + '/relu', nn.ReLU(inplace=True)),
    )


def conv_bn(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
    )


def conv(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True))


def conv_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)),
        g_name(name + '/relu', nn.ReLU()),
    )

def conv_prelu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)),
        g_name(name + '/prelu', nn.PReLU()),
    )
    

if __name__ == '__main__':

    class BasicBlock(nn.Module):

        def __init__(self, name, in_channels, middle_channels, out_channels, stride, residual):
            super(BasicBlock, self).__init__()
            self.g_name = name
            self.residual = residual
            self.conv = [
                conv_bn(name + '/conv1', 
                    in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                conv_bn_relu(name + '/conv2', in_channels, middle_channels, 1),
                conv_bn(name + '/conv3', middle_channels, out_channels, 1),
            ]
            self.conv = nn.Sequential(*self.conv)
            # self.relu = g_name(name + '/relu', nn.ReLU(inplace=True))

        def forward(self, x):
            x = x + self.conv(x) if self.residual else self.conv(x)
            # x = self.relu(x)
            return x

        def generate_caffe_prototxt(self, caffe_net, layer):
            residual_layer = layer
            layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
            if self.residual:
                layer = L.Eltwise(residual_layer, layer, operation=P.Eltwise.SUM)
                caffe_net[self.g_name + '/sum'] = layer
            # layer = generate_caffe_prototxt(self.relu, caffe_net, layer)
            return layer


    class Network(nn.Module):

        def __init__(self, num_outputs, width_multiplier=32):
            super(Network, self).__init__()

            assert width_multiplier >= 0 and width_multiplier <= 256
            # assert width_multiplier % 2 == 0

            self.network = [
                g_name('data/bn', nn.BatchNorm2d(3)),
                conv_bn_relu('stage1/conv', 3, 32, 3, 2, 1),
                # g_name('stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)),
            ]
            channel = lambda i: (2**i) * int(width_multiplier)
            network_parameters = [
                (32,         channel(2) * 4, channel(2), 2, 2),
                (channel(2), channel(2) * 4, channel(2), 2, 4),
                (channel(2), channel(3) * 4, channel(3), 2, 8),
                (channel(3), channel(4) * 4, channel(4), 2, 4),
            ]
            for i, parameters in enumerate(network_parameters):
                in_channels, middle_channels, out_channels, stride, num_blocks = parameters
                self.network += [self._generate_stage('stage_{}'.format(i + 2), 
                    in_channels, middle_channels, out_channels, stride, num_blocks)]
            self.network += [
                conv_bn_relu('unsqueeze', out_channels, out_channels * 4, 1),
                g_name('pool_fc', nn.AvgPool2d(7)),
                g_name('fc', nn.Conv2d(out_channels * 4, num_outputs, 1)),
            ]
            self.network = nn.Sequential(*self.network)

            for name, m in self.named_modules():
                if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
                    nn.init.kaiming_normal(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant(m.bias, 0)

        def _generate_stage(self, name, in_channels, middle_channels, out_channels, stride, num_blocks):
            blocks = [BasicBlock(name + '_1', in_channels, middle_channels, out_channels, 2, False)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(name + '_{}'.format(i + 1), 
                    out_channels, middle_channels, out_channels, 1, True))
            return nn.Sequential(*blocks)

        def forward(self, x):
            return self.network(x).view(x.size(0), -1)
        
        def generate_caffe_prototxt(self, caffe_net, layer):
            return generate_caffe_prototxt(self.network, caffe_net, layer)

        def convert_to_caffe(self, name):
            caffe_net = caffe.NetSpec()
            layer = L.Input(shape=dict(dim=[1, 3, 224, 224]))
            caffe_net.tops['data'] = layer
            generate_caffe_prototxt(self, caffe_net, layer)
            print(caffe_net.to_proto())
            with open(name + '.prototxt', 'wb') as f:
                f.write(str(caffe_net.to_proto()))
            caffe_net = caffe.Net(name + '.prototxt', caffe.TEST)
            convert_pytorch_to_caffe(self, caffe_net)
            caffe_net.save(name + '.caffemodel')


    network = Network(1000, 8)
    print(network)
    network.convert_to_caffe('net')