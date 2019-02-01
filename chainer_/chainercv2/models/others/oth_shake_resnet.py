#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import configuration
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import cupy


class ShakeShake(chainer.function.Function):

    def __init__(self):
        return

    def forward(self, inputs):
        x1, x2 = inputs
        xp = cuda.get_array_module(x1)  # Get numpy(x=n) or cupy(x=c) array module
        alpha = xp.ones(x1.shape, dtype=x1.dtype) * 0.5

        if configuration.config.train:
            for i in range(len(alpha)):
                alpha[i] = xp.random.rand()

        return x1 * alpha + x2 * (xp.ones(x1.shape, dtype=x1.dtype) - alpha),

    def backward(self, inputs, grad_outputs):
        gx, = grad_outputs
        xp = cuda.get_array_module(gx)
        beta = xp.empty(gx.shape, dtype=gx.dtype)

        for i in range(len(beta)):
            beta[i] = xp.random.rand()

        return gx * beta, gx * (xp.ones(gx.shape, dtype=gx.dtype) - beta)


def shake_shake(x1, x2):
    return ShakeShake()(x1, x2)


class ShortCut(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride, initialW):
        super(ShortCut, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, out_channels // 2,
                ksize=1, stride=1, pad=0, initialW=initialW, nobias=True
            )
            self.conv2 = L.Convolution2D(
                in_channels, out_channels // 2,
                ksize=1, stride=1, pad=0, initialW=initialW, nobias=True
            )
            self.bn = L.BatchNormalization(out_channels)

        self._stride = stride

    def __call__(self, x):
        h0 = F.relu(x)

        h1 = h0
        h1 = F.average_pooling_2d(h1, ksize=1, stride=self._stride, pad=0)
        h1 = self.conv1(h1)

        h2 = self._zero_pads(self._zero_pads(h0, pad=1, axis=2), pad=1, axis=3)
        h2 = F.average_pooling_2d(h2, ksize=1, stride=self._stride, pad=0)[:, :, 1:, 1:]
        h2 = self.conv2(h2)

        h = F.concat((h1, h2), axis=1)
        h = self.bn(h)
        return h

    def _zero_pads(self, x, pad, axis):
        sizes = list(x.data.shape)
        sizes[axis] = pad

        xp = cuda.get_array_module(x)  # Get numpy(x=n) or cupy(x=c) array module
        dtype = x.dtype
        pad_tensor = xp.zeros(sizes, dtype=dtype)
        if isinstance(x.data, cupy.core.core.ndarray):
            device = cuda.get_device_from_array(x.data)
            pad_tensor = cuda.to_gpu(pad_tensor, device=device)

        pad_tensor = Variable(pad_tensor)
        return F.concat((pad_tensor, x), axis=axis)


class BuildingShakeBlocks(chainer.link.Chain):

    """Build a stage that consists of several residual/shake blocks.
    Args:
        n_block (int): Number of residual/shake blocks used in the stage.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_block, in_channels, out_channels, stride, initialW=None):
        super(BuildingShakeBlocks, self).__init__()
        with self.init_scope():
            # Down-sample res-block
            self.a = ShakeBlock(in_channels, out_channels, stride=stride, initialW=initialW)
            self._blocks = ['a']
            # Basic res-blocks
            for i in range(n_block - 1):
                name = 'b{}'.format(i + 1)
                basic = ShakeBlock(out_channels, out_channels, stride=1, initialW=initialW)
                setattr(self, name, basic)
                self._blocks.append(name)
    
    def __call__(self, x):
        for name in self._blocks:
            block = getattr(self, name)
            x = block(x)
        return x


class ShakeBlock(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=1, initialW=None):
        super(ShakeBlock, self).__init__()
        
        with self.init_scope():
            self.branch1 = RCBRCB(in_channels, out_channels, stride=stride, initialW=initialW)
            self.branch2 = RCBRCB(in_channels, out_channels, stride=stride, initialW=initialW)

            equal_io = (in_channels == out_channels)
            self.short_cut = None if equal_io else ShortCut(in_channels, out_channels, stride, initialW)
    
    def __call__(self, x):
        h0 = x if (self.short_cut is None) else self.short_cut(x)
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        return h0 + shake_shake(h1, h2)


class RCBRCB(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=1, initialW=None):
        super(RCBRCB, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, 
                ksize=3, stride=stride, pad=1, initialW=initialW, nobias=True
            )
            self.bn1 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels,
                ksize=3, stride=1, pad=1, initialW=initialW, nobias=True
            )
            self.bn2 = L.BatchNormalization(out_channels)
    
    def __call__(self, x):
        h = self.bn1(self.conv1(F.relu(x)))
        h = self.bn2(self.conv2(F.relu(h)))
        return h


class ShakeResNet(chainer.Chain):

    def __init__(self, n_out=10, n_layer=26, base_width=64):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if (n_layer - 2) % 6 == 0:
            n_block = (n_layer - 2) // 6
            # n_layer = n_stage(=3) * n_block * n_conv(=2) + 2(conv1 + fc5)
        else:
            raise ValueError(
                'The n_layer argument should be mod({} - 2, 6) == 0, \
                 but {} was given.'.format(n_layer, n_layer))
        
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 16,
                ksize=3, stride=1, pad=1, nobias=True, **kwargs
            )
            self.bn1 = L.BatchNormalization(16)

            k = base_width
            self.stage2 = BuildingShakeBlocks(
                n_block, 16, k, 1, **kwargs)
            self.stage3 = BuildingShakeBlocks(
                n_block, k, 2 * k, 2, **kwargs)
            self.stage4 = BuildingShakeBlocks(
                n_block, 2 * k, 4 * k, 2, **kwargs)
            
            self.fc5 = L.Linear(4 * k, n_out)

    def __call__(self, x):
        h = x # [b, 3, 32, 32]
        h = self.bn1(self.conv1(h)) # [b, 16, 32, 32]
        h = self.stage2(h) # [b, k, 32, 32]
        h = self.stage3(h) # [b, 2*k, 16, 16]
        h = self.stage4(h) # [b, 4*k, 8, 8]
        B, _, H, W = h.data.shape
        h = F.average_pooling_2d(F.relu(h), ksize=(H, W)).reshape(B, -1) # [b, 4*k]
        h = self.fc5(h) # [b, n_out]
        return h