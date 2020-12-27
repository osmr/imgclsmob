"""
    FractalNet for CIFAR, implemented in Gluon.
    Original paper: 'FractalNet: Ultra-Deep Neural Networks without Residuals,' https://arxiv.org/abs/1605.07648.
"""

__all__ = ['CIFARFractalNet', 'fractalnet_cifar10', 'fractalnet_cifar100']

import os
import numpy as np
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ParametricSequential


class DropConvBlock(HybridBlock):
    """
    Convolution block with Batch normalization, ReLU activation, and Dropout layer.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 bn_use_global_stats=False,
                 dropout_prob=0.0,
                 **kwargs):
        super(DropConvBlock, self).__init__(**kwargs)
        self.use_dropout = (dropout_prob != 0.0)

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_prob)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


def drop_conv3x3_block(in_channels,
                       out_channels,
                       strides=1,
                       padding=1,
                       use_bias=False,
                       bn_use_global_stats=False,
                       dropout_prob=0.0,
                       **kwargs):
    """
    3x3 version of the convolution block with dropout.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    return DropConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        dropout_prob=dropout_prob,
        **kwargs)


class FractalBlock(HybridBlock):
    """
    FractalNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_columns : int
        Number of columns in each block.
    loc_drop_prob : float
        Local drop path probability.
    dropout_prob : float
        Probability of dropout.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_columns,
                 loc_drop_prob,
                 dropout_prob,
                 **kwargs):
        super(FractalBlock, self).__init__(**kwargs)
        assert (num_columns >= 1)
        self.num_columns = num_columns
        self.loc_drop_prob = loc_drop_prob

        with self.name_scope():
            self.blocks = nn.HybridSequential(prefix="")
            depth = 2 ** (num_columns - 1)
            for i in range(depth):
                level_block_i = nn.HybridSequential(prefix='block{}_'.format(i + 1))
                for j in range(self.num_columns):
                    column_step_j = 2 ** j
                    if (i + 1) % column_step_j == 0:
                        in_channels_ij = in_channels if (i + 1 == column_step_j) else out_channels
                        level_block_i.add(drop_conv3x3_block(
                            in_channels=in_channels_ij,
                            out_channels=out_channels,
                            dropout_prob=dropout_prob))
                self.blocks.add(level_block_i)

    @staticmethod
    def calc_drop_mask(batch_size,
                       glob_num_columns,
                       curr_num_columns,
                       max_num_columns,
                       loc_drop_prob):
        """
        Calculate drop path mask.

        Parameters:
        ----------
        batch_size : int
            Size of batch.
        glob_num_columns : int
            Number of columns in global drop path mask.
        curr_num_columns : int
            Number of active columns in the current level of block.
        max_num_columns : int
            Number of columns for all network.
        loc_drop_prob : float
            Local drop path probability.

        Returns:
        -------
        np.array
            Resulted mask.
        """
        glob_batch_size = glob_num_columns.shape[0]
        glob_drop_mask = np.zeros((curr_num_columns, glob_batch_size), dtype=np.float32)
        glob_drop_num_columns = glob_num_columns - (max_num_columns - curr_num_columns)
        glob_drop_indices = np.where(glob_drop_num_columns >= 0)[0]
        glob_drop_mask[glob_drop_num_columns[glob_drop_indices], glob_drop_indices] = 1.0

        loc_batch_size = batch_size - glob_batch_size
        loc_drop_mask = np.random.binomial(
            n=1,
            p=(1.0 - loc_drop_prob),
            size=(curr_num_columns, loc_batch_size)).astype(np.float32)
        alive_count = loc_drop_mask.sum(axis=0)
        dead_indices = np.where(alive_count == 0.0)[0]
        loc_drop_mask[np.random.randint(0, curr_num_columns, size=dead_indices.shape), dead_indices] = 1.0

        drop_mask = np.concatenate((glob_drop_mask, loc_drop_mask), axis=1)
        return drop_mask

    @staticmethod
    def join_outs(F,
                  raw_outs,
                  glob_num_columns,
                  num_columns,
                  loc_drop_prob,
                  training):
        """
        Join outputs for current level of block.

        Parameters:
        ----------
        F : namespace
            Symbol or NDArray namespace.
        raw_outs : list of Tensor
            Current outputs from active columns.
        glob_num_columns : int
            Number of columns in global drop path mask.
        num_columns : int
            Number of columns for all network.
        loc_drop_prob : float
            Local drop path probability.
        training : bool
            Whether training mode for network.

        Returns:
        -------
        NDArray
            Joined output.
        """
        curr_num_columns = len(raw_outs)
        out = F.stack(*raw_outs, axis=0)
        assert (out.shape[0] == curr_num_columns)

        if training:
            batch_size = out.shape[1]
            batch_mask = FractalBlock.calc_drop_mask(
                batch_size=batch_size,
                glob_num_columns=glob_num_columns,
                curr_num_columns=curr_num_columns,
                max_num_columns=num_columns,
                loc_drop_prob=loc_drop_prob)
            batch_mask = mx.nd.array(batch_mask, ctx=out.context)
            assert (batch_mask.shape[0] == curr_num_columns)
            assert (batch_mask.shape[1] == batch_size)
            batch_mask = batch_mask.expand_dims(2).expand_dims(3).expand_dims(4)
            masked_out = out * batch_mask
            num_alive = batch_mask.sum(axis=0).asnumpy()
            num_alive[num_alive == 0.0] = 1.0
            num_alive = mx.nd.array(num_alive, ctx=out.context)
            out = masked_out.sum(axis=0) / num_alive
        else:
            out = out.mean(axis=0)

        return out

    def hybrid_forward(self, F, x, glob_num_columns):
        outs = [x] * self.num_columns

        for level_block_i in self.blocks._children.values():
            outs_i = []

            for j, block_ij in enumerate(level_block_i._children.values()):
                input_i = outs[j]
                outs_i.append(block_ij(input_i))

            joined_out = FractalBlock.join_outs(
                F=F,
                raw_outs=outs_i[::-1],
                glob_num_columns=glob_num_columns,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                training=mx.autograd.is_training())

            len_level_block_i = len(level_block_i._children.values())
            for j in range(len_level_block_i):
                outs[j] = joined_out

        return outs[0]


class FractalUnit(HybridBlock):
    """
    FractalNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_columns : int
        Number of columns in each block.
    loc_drop_prob : float
        Local drop path probability.
    dropout_prob : float
        Probability of dropout.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_columns,
                 loc_drop_prob,
                 dropout_prob,
                 **kwargs):
        super(FractalUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.block = FractalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                num_columns=num_columns,
                loc_drop_prob=loc_drop_prob,
                dropout_prob=dropout_prob)
            self.pool = nn.MaxPool2D(
                pool_size=2,
                strides=2)

    def hybrid_forward(self, F, x, glob_num_columns):
        x = self.block(x, glob_num_columns)
        x = self.pool(x)
        return x


class CIFARFractalNet(HybridBlock):
    """
    FractalNet model for CIFAR from 'FractalNet: Ultra-Deep Neural Networks without Residuals,'
    https://arxiv.org/abs/1605.07648.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    num_columns : int
        Number of columns in each block.
    dropout_probs : list of float
        Probability of dropout in each block.
    loc_drop_prob : float
        Local drop path probability.
    glob_drop_ratio : float
        Global drop part fraction.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 num_columns,
                 dropout_probs,
                 loc_drop_prob,
                 glob_drop_ratio,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARFractalNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.glob_drop_ratio = glob_drop_ratio
        self.num_columns = num_columns

        with self.name_scope():
            self.features = ParametricSequential(prefix="")
            for i, out_channels in enumerate(channels):
                dropout_prob = dropout_probs[i]
                self.features.add(FractalUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_columns=num_columns,
                    loc_drop_prob=loc_drop_prob,
                    dropout_prob=dropout_prob))
                in_channels = out_channels

            self.output = nn.Dense(
                units=classes,
                in_units=in_channels)

    def hybrid_forward(self, F, x):
        glob_batch_size = int(x.shape[0] * self.glob_drop_ratio)
        glob_num_columns = np.random.randint(0, self.num_columns, size=(glob_batch_size,))

        x = self.features(x, glob_num_columns)
        x = self.output(x)
        return x


def get_fractalnet_cifar(num_classes,
                         model_name=None,
                         pretrained=False,
                         ctx=cpu(),
                         root=os.path.join("~", ".mxnet", "models"),
                         **kwargs):
    """
    Create WRN model for CIFAR with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    dropout_probs = (0.0, 0.1, 0.2, 0.3, 0.4)
    channels = [64 * (2 ** (i if i != len(dropout_probs) - 1 else i - 1)) for i in range(len(dropout_probs))]
    num_columns = 3
    loc_drop_prob = 0.15
    glob_drop_ratio = 0.5

    net = CIFARFractalNet(
        channels=channels,
        num_columns=num_columns,
        dropout_probs=dropout_probs,
        loc_drop_prob=loc_drop_prob,
        glob_drop_ratio=glob_drop_ratio,
        classes=num_classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def fractalnet_cifar10(num_classes=10, **kwargs):
    """
    FractalNet model for CIFAR-10 from 'FractalNet: Ultra-Deep Neural Networks without Residuals,'
    https://arxiv.org/abs/1605.07648.

    Parameters:
    ----------
    num_classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_fractalnet_cifar(num_classes=num_classes, model_name="fractalnet_cifar10", **kwargs)


def fractalnet_cifar100(num_classes=100, **kwargs):
    """
    FractalNet model for CIFAR-100 from 'FractalNet: Ultra-Deep Neural Networks without Residuals,'
    https://arxiv.org/abs/1605.07648.

    Parameters:
    ----------
    num_classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_fractalnet_cifar(num_classes=num_classes, model_name="fractalnet_cifar100", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (fractalnet_cifar10, 10),
        (fractalnet_cifar100, 100),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fractalnet_cifar10 or weight_count == 33724618)
        assert (model != fractalnet_cifar100 or weight_count == 33770788)

        x = mx.nd.zeros((14, 3, 32, 32), ctx=ctx)
        y = net(x)
        # with mx.autograd.record():
        #     y = net(x)
        #     y.backward()
        assert (y.shape == (14, classes))


if __name__ == "__main__":
    _test()
