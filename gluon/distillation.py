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
    down_factor : int, default 10
        Channel down factor.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 classes=1000,
                 down_factor=10,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(MealDiscriminator, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            in_channels = classes
            out_channels = in_channels // down_factor
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = out_channels
            out_channels = in_channels // down_factor
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(conv1x1(
                in_channels=out_channels,
                out_channels=2,
                use_bias=True))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = x.expand_dims(-1).expand_dims(-1)
        x = self.features(x)
        x = self.output(x)
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
        z_pred = F.zeros_like(pred)[:, 0].one_hot(depth=2)
        loss_pred = super(MealAdvLoss, self).hybrid_forward(F, pred, z_pred)

        z_label = F.ones_like(label)[:, 0].one_hot(depth=2)
        loss_label = super(MealAdvLoss, self).hybrid_forward(F, label, z_label)

        return loss_pred + loss_label