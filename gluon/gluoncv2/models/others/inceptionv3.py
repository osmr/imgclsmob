"""
    InceptionV3, implemented in Gluon.
    Original paper: 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.
"""

# __all__ = ['ResNet', 'resnet10']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import SEBlock


class InceptConv(HybridBlock):
    """
    InceptionV3 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptConv, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                epsilon=1e-3,
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class InceptBranch(HybridBlock):
    """
    Simple InceptionV3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    pool_type : str, default 'none'
        Type of poling ('none', 'avg', 'max').
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bn_use_global_stats,
                 pool_type='none',
                 **kwargs):
        super(InceptBranch, self).__init__(**kwargs)
        self.use_pool = (pool_type != 'none')
        with self.name_scope():
            if pool_type == 'avg':
                self.pool = nn.AvgPool2D(
                    pool_size=3,
                    strides=1,
                    padding=1)
            elif pool_type == 'max':
                self.pool = nn.MaxPool2D(
                    pool_size=3,
                    strides=2,
                    padding=0)
            self.conv = InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        if self.use_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x


# class InceptBlockA(HybridBlock):
#     """
#     InceptionV3 type A block.
#
#     Parameters:
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     strides : int or tuple/list of 2 int
#         Strides of the convolution.
#     bn_use_global_stats : bool
#         Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
#     conv1_stride : bool
#         Whether to use stride in the first or the second convolution layer of the block.
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  strides,
#                  bn_use_global_stats,
#                  conv1_stride,
#                  **kwargs):
#         super(InceptBlockA, self).__init__(**kwargs)
#         mid_channels = out_channels // 4
#
#         with self.name_scope():
#             self.conv1 = res_conv1x1(
#                 in_channels=in_channels,
#                 out_channels=mid_channels,
#                 strides=(strides if conv1_stride else 1),
#                 bn_use_global_stats=bn_use_global_stats,
#                 activate=True)
#             self.conv2 = res_conv3x3(
#                 in_channels=mid_channels,
#                 out_channels=mid_channels,
#                 strides=(1 if conv1_stride else strides),
#                 bn_use_global_stats=bn_use_global_stats,
#                 activate=True)
#             self.conv3 = res_conv1x1(
#                 in_channels=mid_channels,
#                 out_channels=out_channels,
#                 strides=1,
#                 bn_use_global_stats=bn_use_global_stats,
#                 activate=False)
#
#     def hybrid_forward(self, F, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x
#
#
# class ResUnit(HybridBlock):
#     """
#     ResNet unit with residual connection.
#
#     Parameters:
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     strides : int or tuple/list of 2 int
#         Strides of the convolution.
#     bn_use_global_stats : bool
#         Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
#     bottleneck : bool
#         Whether to use a bottleneck or simple block in units.
#     conv1_stride : bool
#         Whether to use stride in the first or the second convolution layer of the block.
#     use_se : bool
#         Whether to use SE block.
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  strides,
#                  bn_use_global_stats,
#                  bottleneck,
#                  conv1_stride,
#                  use_se,
#                  **kwargs):
#         super(ResUnit, self).__init__(**kwargs)
#         self.use_se = use_se
#         self.resize_identity = (in_channels != out_channels) or (strides != 1)
#
#         with self.name_scope():
#             if bottleneck:
#                 self.body = ResBottleneck(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     strides=strides,
#                     bn_use_global_stats=bn_use_global_stats,
#                     conv1_stride=conv1_stride)
#             else:
#                 self.body = InceptBranch(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     strides=strides,
#                     bn_use_global_stats=bn_use_global_stats)
#             if self.use_se:
#                 self.se = SEBlock(channels=out_channels)
#             if self.resize_identity:
#                 self.identity_conv = res_conv1x1(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     strides=strides,
#                     bn_use_global_stats=bn_use_global_stats,
#                     activate=False)
#             self.activ = nn.Activation('relu')
#
#     def hybrid_forward(self, F, x):
#         if self.resize_identity:
#             identity = self.identity_conv(x)
#         else:
#             identity = x
#         x = self.body(x)
#         if self.use_se:
#             x = self.se(x)
#         x = x + identity
#         x = self.activ(x)
#         return x
#
#
# class ResInitBlock(HybridBlock):
#     """
#     ResNet specific initial block.
#
#     Parameters:
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     bn_use_global_stats : bool
#         Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
#     """
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  bn_use_global_stats,
#                  **kwargs):
#         super(ResInitBlock, self).__init__(**kwargs)
#         with self.name_scope():
#             self.conv = InceptConv(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=7,
#                 strides=2,
#                 padding=3,
#                 bn_use_global_stats=bn_use_global_stats,
#                 activate=True)
#             self.pool = nn.MaxPool2D(
#                 pool_size=3,
#                 strides=2,
#                 padding=1)
#
#     def hybrid_forward(self, F, x):
#         x = self.conv(x)
#         x = self.pool(x)
#         return x
#
#
# class ResNet(HybridBlock):
#     """
#     InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
#     https://arxiv.org/abs/1512.00567.
#
#     Parameters:
#     ----------
#     channels : list of list of int
#         Number of output channels for each unit.
#     init_block_channels : int
#         Number of output channels for the initial unit.
#     bottleneck : bool
#         Whether to use a bottleneck or simple block in units.
#     conv1_stride : bool
#         Whether to use stride in the first or the second convolution layer in units.
#     use_se : bool
#         Whether to use SE block.
#     bn_use_global_stats : bool, default False
#         Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
#         Useful for fine-tuning.
#     in_channels : int, default 3
#         Number of input channels.
#     in_size : tuple of two ints, default (224, 224)
#         Spatial size of the expected input image.
#     classes : int, default 1000
#         Number of classification classes.
#     """
#     def __init__(self,
#                  channels,
#                  init_block_channels,
#                  bottleneck,
#                  conv1_stride,
#                  use_se,
#                  bn_use_global_stats=False,
#                  in_channels=3,
#                  in_size=(224, 224),
#                  classes=1000,
#                  **kwargs):
#         super(ResNet, self).__init__(**kwargs)
#         self.in_size = in_size
#         self.classes = classes
#
#         with self.name_scope():
#             self.features = nn.HybridSequential(prefix='')
#             self.features.add(ResInitBlock(
#                 in_channels=in_channels,
#                 out_channels=init_block_channels,
#                 bn_use_global_stats=bn_use_global_stats))
#             in_channels = init_block_channels
#             for i, channels_per_stage in enumerate(channels):
#                 stage = nn.HybridSequential(prefix='stage{}_'.format(i + 1))
#                 with stage.name_scope():
#                     for j, out_channels in enumerate(channels_per_stage):
#                         strides = 2 if (j == 0) and (i != 0) else 1
#                         stage.add(ResUnit(
#                             in_channels=in_channels,
#                             out_channels=out_channels,
#                             strides=strides,
#                             bn_use_global_stats=bn_use_global_stats,
#                             bottleneck=bottleneck,
#                             conv1_stride=conv1_stride,
#                             use_se=use_se))
#                         in_channels = out_channels
#                 self.features.add(stage)
#             self.features.add(nn.AvgPool2D(
#                 pool_size=7,
#                 strides=1))
#
#             self.output = nn.HybridSequential(prefix='')
#             self.output.add(nn.Flatten())
#             self.output.add(nn.Dense(
#                 units=classes,
#                 in_units=in_channels))
#
#     def hybrid_forward(self, F, x):
#         x = self.features(x)
#         x = self.output(x)
#         return x
#
#
# def get_resnet(blocks,
#                conv1_stride=True,
#                use_se=False,
#                width_scale=1.0,
#                model_name=None,
#                pretrained=False,
#                ctx=cpu(),
#                root=os.path.join('~', '.mxnet', 'models'),
#                **kwargs):
#     """
#     Create InceptionV3 model with specific parameters.
#
#     Parameters:
#     ----------
#     blocks : int
#         Number of blocks.
#     conv1_stride : bool
#         Whether to use stride in the first or the second convolution layer in units.
#     use_se : bool
#         Whether to use SE block.
#     width_scale : float
#         Scale factor for width of layers.
#     model_name : str or None, default None
#         Model name for loading pretrained model.
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     ctx : Context, default CPU
#         The context in which to load the pretrained weights.
#     root : str, default '~/.mxnet/models'
#         Location for keeping the model parameters.
#     """
#
#     if blocks == 10:
#         layers = [1, 1, 1, 1]
#     elif blocks == 12:
#         layers = [2, 1, 1, 1]
#     elif blocks == 14:
#         layers = [2, 2, 1, 1]
#     elif blocks == 16:
#         layers = [2, 2, 2, 1]
#     elif blocks == 18:
#         layers = [2, 2, 2, 2]
#     elif blocks == 34:
#         layers = [3, 4, 6, 3]
#     elif blocks == 50:
#         layers = [3, 4, 6, 3]
#     elif blocks == 101:
#         layers = [3, 4, 23, 3]
#     elif blocks == 152:
#         layers = [3, 8, 36, 3]
#     elif blocks == 200:
#         layers = [3, 24, 36, 3]
#     else:
#         raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))
#
#     init_block_channels = 64
#
#     if blocks < 50:
#         channels_per_layers = [64, 128, 256, 512]
#         bottleneck = False
#     else:
#         channels_per_layers = [256, 512, 1024, 2048]
#         bottleneck = True
#
#     channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
#
#     if width_scale != 1.0:
#         channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
#         init_block_channels = int(init_block_channels * width_scale)
#
#     net = ResNet(
#         channels=channels,
#         init_block_channels=init_block_channels,
#         bottleneck=bottleneck,
#         conv1_stride=conv1_stride,
#         use_se=use_se,
#         **kwargs)
#
#     if pretrained:
#         if (model_name is None) or (not model_name):
#             raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
#         from .model_store import get_model_file
#         net.load_parameters(
#             filename=get_model_file(
#                 model_name=model_name,
#                 local_model_store_dir_path=root),
#             ctx=ctx)
#
#     return net
#
#
# def resnet10(**kwargs):
#     """
#     InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
#     https://arxiv.org/abs/1512.00567.
#
#     Parameters:
#     ----------
#     pretrained : bool, default False
#         Whether to load the pretrained weights for model.
#     ctx : Context, default CPU
#         The context in which to load the pretrained weights.
#     root : str, default '~/.mxnet/models'
#         Location for keeping the model parameters.
#     """
#     return get_resnet(blocks=10, model_name="resnet10", **kwargs)
#
#
# def _test():
#     import numpy as np
#     import mxnet as mx
#
#     pretrained = False
#
#     models = [
#         resnet10,
#     ]
#
#     for model in models:
#
#         net = model(pretrained=pretrained)
#
#         ctx = mx.cpu()
#         if not pretrained:
#             net.initialize(ctx=ctx)
#
#         net_params = net.collect_params()
#         weight_count = 0
#         for param in net_params.values():
#             if (param.shape is None) or (not param._differentiable):
#                 continue
#             weight_count += np.prod(param.shape)
#         print("m={}, {}".format(model.__name__, weight_count))
#         assert (model != resnet10 or weight_count == 5418792)
#
#         x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
#         y = net(x)
#         assert (y.shape == (1, 1000))
#
#
# if __name__ == "__main__":
#     _test()
