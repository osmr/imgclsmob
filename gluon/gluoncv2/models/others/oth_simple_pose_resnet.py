from __future__ import division

__all__ = ['oth_simple_pose_resnet18_v1b', 'oth_simple_pose_resnet50_v1b', 'oth_simple_pose_resnet101_v1b',
           'oth_simple_pose_resnet152_v1b', 'oth_simple_pose_resnet50_v1d', 'oth_simple_pose_resnet101_v1d',
           'oth_simple_pose_resnet152_v1d', 'oth_resnet50_v1d', 'oth_resnet101_v1d',
           'oth_resnet152_v1d', 'oth_mobilenet_v2_1_0', 'oth_mobilenet_v2_0_75', 'oth_mobilenet_v2_0_5',
           'oth_mobilenet_v2_0_25']

import numpy as np
import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import initializer
import gluoncv as gcv


def get_max_pred(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = mx.nd.argmax(heatmaps_reshaped, 2)
    maxvals = mx.nd.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = mx.nd.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = mx.nd.floor((preds[:, :, 1]) / width)

    pred_mask = mx.nd.tile(mx.nd.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def _get_final_preds(batch_heatmaps):
    coords, maxvals = get_max_pred(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(mx.nd.floor(coords[n][p][0] + 0.5).asscalar())
            py = int(mx.nd.floor(coords[n][p][1] + 0.5).asscalar())
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = mx.nd.concat(hm[py][px+1] - hm[py][px-1],
                                 hm[py+1][px] - hm[py-1][px],
                                 dim=0)
                coords[n][p] += mx.nd.sign(diff) * .25

    return coords, maxvals


class SimplePoseResNet(HybridBlock):

    def __init__(self,
                 base_name='resnet50_v1b',
                 pretrained_base=False,
                 pretrained_ctx=cpu(),
                 num_joints=17,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 deconv_with_bias=False,
                 in_channels=3,
                 in_size=(256, 192),
                 **kwargs):
        super(SimplePoseResNet, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size

        from gluoncv.model_zoo import get_model
        base_network = get_model(
            base_name,
            pretrained=pretrained_base,
            ctx=pretrained_ctx,
            norm_layer=gcv.nn.BatchNormCudnnOff)

        self.resnet = nn.HybridSequential()
        if base_name.endswith('v1'):
            for layer in ['features']:
                self.resnet.add(getattr(base_network, layer))
        else:
            for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']:
                self.resnet.add(getattr(base_network, layer))

        self.deconv_with_bias = deconv_with_bias

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2D(
            channels=num_joints,
            kernel_size=final_conv_kernel,
            strides=1,
            padding=1 if final_conv_kernel == 3 else 0,
            weight_initializer=initializer.Normal(0.001),
            bias_initializer=initializer.Zero()
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self,
                           num_layers,
                           num_filters,
                           num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'

        layer = nn.HybridSequential(prefix='')
        with layer.name_scope():
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i])

                planes = num_filters[i]
                layer.add(
                    nn.Conv2DTranspose(
                        channels=planes,
                        kernel_size=kernel,
                        strides=2,
                        padding=padding,
                        output_padding=output_padding,
                        use_bias=self.deconv_with_bias,
                        weight_initializer=initializer.Normal(0.001),
                        bias_initializer=initializer.Zero()))
                layer.add(gcv.nn.BatchNormCudnnOff(gamma_initializer=initializer.One(),
                                                   beta_initializer=initializer.Zero()))
                layer.add(nn.Activation('relu'))
                self.inplanes = planes

        return layer

    def hybrid_forward(self, F, x, center=None, scale=None):
        x = self.resnet(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        batch_heatmaps = x.as_in_context(mx.cpu())
        y, maxvals = _get_final_preds(batch_heatmaps=batch_heatmaps)
        keypoints = F.concat(y, maxvals, dim=2)
        return keypoints


def get_simple_pose_resnet(base_name,
                           pretrained=False,
                           ctx=cpu(),
                           root='~/.mxnet/models',
                           **kwargs):
    if "fixed_size" in kwargs:
        del kwargs["fixed_size"]
    net = SimplePoseResNet(base_name, **kwargs)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(
            get_model_file(
                'simple_pose_%s'%(base_name),
                tag=pretrained,
                root=root),
            ctx=ctx)

    return net


def oth_simple_pose_resnet18_v1b(**kwargs):
    r"""ResNet-18 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet18_v1b', **kwargs)

def oth_simple_pose_resnet50_v1b(**kwargs):
    r"""ResNet-50 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet50_v1b', **kwargs)

def oth_simple_pose_resnet101_v1b(**kwargs):
    r"""ResNet-101 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet101_v1b', **kwargs)

def oth_simple_pose_resnet152_v1b(**kwargs):
    r"""ResNet-152 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet152_v1b', **kwargs)

def oth_simple_pose_resnet50_v1d(**kwargs):
    r"""ResNet-50-d backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet50_v1d', **kwargs)

def oth_simple_pose_resnet101_v1d(**kwargs):
    r"""ResNet-101-d backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet101_v1d', **kwargs)

def oth_simple_pose_resnet152_v1d(**kwargs):
    r"""ResNet-152-d backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet152_v1d', **kwargs)


def oth_resnet50_v1d(pretrained=False, **kwargs):
    from gluoncv.model_zoo import get_model
    net = get_model(
        'resnet50_v1d',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def oth_resnet101_v1d(pretrained=False, **kwargs):
    from gluoncv.model_zoo import get_model
    net = get_model(
        'resnet101_v1d',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def oth_resnet152_v1d(pretrained=False, **kwargs):
    from gluoncv.model_zoo import get_model
    net = get_model(
        'resnet152_v1d',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def oth_mobilenet_v2_1_0(pretrained=False, **kwargs):
    if "in_channels" in kwargs:
        del kwargs["in_channels"]
    from gluoncv.model_zoo import get_model
    net = get_model(
        'mobilenetv2_1.0',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def oth_mobilenet_v2_0_75(pretrained=False, **kwargs):
    if "in_channels" in kwargs:
        del kwargs["in_channels"]
    from gluoncv.model_zoo import get_model
    net = get_model(
        'mobilenetv2_0.75',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def oth_mobilenet_v2_0_5(pretrained=False, **kwargs):
    if "in_channels" in kwargs:
        del kwargs["in_channels"]
    from gluoncv.model_zoo import get_model
    net = get_model(
        'mobilenetv2_0.5',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def oth_mobilenet_v2_0_25(pretrained=False, **kwargs):
    if "in_channels" in kwargs:
        del kwargs["in_channels"]
    from gluoncv.model_zoo import get_model
    net = get_model(
        'mobilenetv2_0.25',
        pretrained=pretrained,
        **kwargs)
    net.in_size = (224, 224)
    return net


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = True

    models = [
        oth_simple_pose_resnet18_v1b,
        oth_simple_pose_resnet50_v1b,
        oth_simple_pose_resnet101_v1b,
        oth_simple_pose_resnet152_v1b,
        oth_simple_pose_resnet50_v1d,
        oth_simple_pose_resnet101_v1d,
        oth_simple_pose_resnet152_v1d,
        oth_resnet50_v1d,
        oth_resnet101_v1d,
        oth_resnet152_v1d,

        oth_mobilenet_v2_1_0,
        oth_mobilenet_v2_0_75,
        oth_mobilenet_v2_0_5,
        oth_mobilenet_v2_0_25,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        x = mx.nd.zeros((1, 3, 256, 192), ctx=ctx)
        y = net(x)
        # assert (y.shape == (1, 17, 64, 48))

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_simple_pose_resnet18_v1b or weight_count == 15376721)
        assert (model != oth_simple_pose_resnet50_v1b or weight_count == 33999697)
        assert (model != oth_simple_pose_resnet101_v1b or weight_count == 52991825)
        assert (model != oth_simple_pose_resnet152_v1b or weight_count == 68635473)
        assert (model != oth_simple_pose_resnet50_v1d or weight_count == 34018929)
        assert (model != oth_simple_pose_resnet101_v1d or weight_count == 53011057)
        assert (model != oth_simple_pose_resnet152_v1d or weight_count == 68654705)


if __name__ == "__main__":
    _test()
