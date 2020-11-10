"""
    DNN distillation routines.
"""

__all__ = ['MealDiscriminator', 'MealAdvLoss']

from mxnet.gluon import nn, HybridBlock
from .gluoncv2.models.common import conv1x1, conv1x1_block
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss


class MealDiscriminator(HybridBlock):
    """
    MEALv2 discriminator.

    Parameters:
    ----------
    classes : int, default 1000
        Number of classification classes.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 classes=1000,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(MealDiscriminator, self).__init__(**kwargs)
        in_channels = classes
        channels = [200, 40, 8]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for out_channels in channels:
                self.features.add(conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                in_channels = out_channels

            self.output = nn.HybridSequential(prefix="")
            self.output.add(conv1x1(
                in_channels=in_channels,
                out_channels=1,
                use_bias=True))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = x.expand_dims(-1).expand_dims(-1)
        x = self.features(x)
        x = self.output(x)
        x = x.squeeze(1)
        return x


class MealAdvLoss(SigmoidBinaryCrossEntropyLoss):
    """
    MEALv2 adversarial loss.

    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and BCE together, which is more numerically
        stable through log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self,
                 **kwargs):
        super(MealAdvLoss, self).__init__(**kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None, pos_weight=None):
        z_pred = F.zeros_like(pred)
        loss_pred = super(MealAdvLoss, self).hybrid_forward(F, pred, z_pred)

        z_label = F.ones_like(label)
        loss_label = super(MealAdvLoss, self).hybrid_forward(F, label, z_label)

        return loss_pred + loss_label


def _test():
    import numpy as np
    import mxnet as mx

    model = MealDiscriminator
    net = model()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    # net.hybridize()
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    print("m={}, {}".format(model.__name__, weight_count))
    # assert (model != MealDiscriminator or weight_count == 208834)

    batch = 14
    classes = 1000
    x = mx.nd.random.normal(shape=(batch, classes), ctx=ctx)
    y = net(x)
    assert (y.shape == (batch,))

    loss = MealAdvLoss()
    z = loss(y, 1 - y)
    print(z)
    pass


if __name__ == "__main__":
    _test()
