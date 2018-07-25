"""
    NASNet-A, implemented in Gluon.
    Original paper: 'Learning Transferable Architectures for Scalable Image Recognition'
"""


from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


TESTING = False


def process_with_padding(x,
                         F,
                         process=(lambda x: x),
                         z_padding=1):
    x = F.pad(x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, z_padding, 0, z_padding, 0))
    x = process(x)
    x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
    return x


def nasnet_batch_norm(in_channels):
    return nn.BatchNorm(
        momentum=0.1,
        epsilon=0.001,
        in_channels=in_channels)


class MaxPoolPad(HybridBlock):

    def __init__(self,
                 **kwargs):
        super(MaxPoolPad, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = process_with_padding(x, F, self.pool)
        return x


class AvgPoolPad(HybridBlock):

    def __init__(self,
                 strides=2,
                 padding=1,
                 **kwargs):
        super(AvgPoolPad, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=3,
                strides=strides,
                padding=padding,
                count_include_pad=False)

    def hybrid_forward(self, F, x):
        x = process_with_padding(x, F, self.pool)
        return x


class DwsConv(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 **kwargs):
        super(DwsConv, self).__init__(**kwargs)

        with self.name_scope():
            self.dw_conv = nn.Conv2D(
                channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=in_channels,
                use_bias=use_bias,
                in_channels=in_channels)
            self.pw_conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=1,
                use_bias=use_bias,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DwsConvBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 specific=False,
                 z_padding=1,
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.specific = specific
        self.z_padding = z_padding

        with self.name_scope():
            self.activ = nn.Activation(activation='relu')
            self.dws_conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias)
            self.bn = nasnet_batch_norm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        if self.specific:
            x = process_with_padding(x, F, self.dws_conv, self.z_padding)
        else:
            x = self.dws_conv(x)
        x = self.bn(x)
        return x


class BranchSeparables(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 specific=False,
                 z_padding=1,
                 **kwargs):
        super(BranchSeparables, self).__init__(**kwargs)

        with self.name_scope():
            self.dws_conv_block1 = DwsConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                specific=specific,
                z_padding=z_padding)
            self.dws_conv_block2 = DwsConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                use_bias=use_bias)

    def hybrid_forward(self, F, x):
        x = self.dws_conv_block1(x)
        x = self.dws_conv_block2(x)
        return x


class BranchSeparablesStem(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 **kwargs):
        super(BranchSeparablesStem, self).__init__(**kwargs)

        with self.name_scope():
            self.dws_conv_block1 = DwsConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias)
            self.dws_conv_block2 = DwsConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                use_bias=use_bias)

    def hybrid_forward(self, F, x):
        x = self.dws_conv_block1(x)
        x = self.dws_conv_block2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 use_bias=False,
                 z_padding=1,
                 **kwargs):
        super(BranchSeparablesReduction, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            specific=True,
            z_padding=z_padding,
            **kwargs)


class ConvBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding=0,
                 use_bias=False,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.activ = nn.Activation(activation='relu')
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                in_channels=in_channels)
            self.bn = nasnet_batch_norm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class CellStem0(HybridBlock):

    def __init__(self,
                 stem_filters,
                 num_filters=42,
                 **kwargs):
        super(CellStem0, self).__init__(**kwargs)

        with self.name_scope():
            self.conv_1x1 = ConvBlock(
                in_channels=stem_filters,
                out_channels=num_filters,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.comb_iter_0_left = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=5,
                strides=2,
                padding=2)
            self.comb_iter_0_right = BranchSeparablesStem(
                in_channels=stem_filters,
                out_channels=num_filters,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=False)

            self.comb_iter_1_left = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)
            self.comb_iter_1_right = BranchSeparablesStem(
                in_channels=stem_filters,
                out_channels=num_filters,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=False)

            self.comb_iter_2_left = nn.AvgPool2D(
                pool_size=3,
                strides=2,
                padding=1)
            self.comb_iter_2_right = BranchSeparablesStem(
                in_channels=stem_filters,
                out_channels=num_filters,
                kernel_size=5,
                strides=2,
                padding=2,
                use_bias=False)

            self.comb_iter_3_right = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_4_left = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                strides=1,
                padding=1)
            self.comb_iter_4_right = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        assert ((not TESTING) or x.shape == (1, 32, 111, 111))

        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out


class CellStemPathBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(CellStemPathBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.avg_pool = nn.AvgPool2D(
                pool_size=1,
                strides=2,
                count_include_pad=False)
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=1,
                strides=1,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        return x


class CellStem1(HybridBlock):

    def __init__(self,
                 stem_filters,
                 num_filters,
                 **kwargs):
        super(CellStem1, self).__init__(**kwargs)

        with self.name_scope():
            self.conv_1x1 = ConvBlock(
                in_channels=(2 * num_filters),
                out_channels=num_filters,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.relu = nn.Activation('relu')

            self.path_1 = CellStemPathBlock(
                in_channels=stem_filters,
                out_channels=(num_filters // 2))

            self.path_2 = CellStemPathBlock(
                in_channels=stem_filters,
                out_channels=(num_filters // 2))

            self.final_path_bn = nasnet_batch_norm(in_channels=num_filters)

            self.comb_iter_0_left = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=5,
                strides=2,
                padding=2,
                specific=True)
            self.comb_iter_0_right = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=7,
                strides=2,
                padding=3,
                specific=True)

            self.comb_iter_1_left = MaxPoolPad()
            self.comb_iter_1_right = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=7,
                strides=2,
                padding=3,
                specific=True)

            self.comb_iter_2_left = AvgPoolPad()
            self.comb_iter_2_right = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=5,
                strides=2,
                padding=2,
                specific=True)

            self.comb_iter_3_right = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_4_left = BranchSeparables(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                strides=1,
                padding=1,
                specific=True)
            self.comb_iter_4_right = MaxPoolPad()

    def hybrid_forward(self, F, x_conv0, x_stem_0):
        assert ((not TESTING) or x_conv0.shape == (1, 32, 111, 111))
        assert ((not TESTING) or x_stem_0.shape == (1, 44, 56, 56))

        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)

        # path 2
        x_path2 = process_with_padding(x_relu, F)
        x_path2 = self.path_2(x_path2)

        # final path
        x_right = self.final_path_bn(F.concat(*[x_path1, x_path2], dim=1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out


class FirstCell(HybridBlock):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right,
                 **kwargs):
        super(FirstCell, self).__init__(**kwargs)

        with self.name_scope():
            self.conv_1x1 = ConvBlock(
                in_channels=in_channels_right,
                out_channels=out_channels_right,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.relu = nn.Activation(activation='relu')

            self.path_1 = CellStemPathBlock(
                in_channels=in_channels_left,
                out_channels=out_channels_left)

            self.path_2 = CellStemPathBlock(
                in_channels=in_channels_left,
                out_channels=out_channels_left)

            self.final_path_bn = nasnet_batch_norm(in_channels=(2 * out_channels_left))

            self.comb_iter_0_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=1,
                padding=2)
            self.comb_iter_0_right = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=3,
                strides=1,
                padding=1)

            self.comb_iter_1_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=1,
                padding=2)
            self.comb_iter_1_right = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=3,
                strides=1,
                padding=1)

            self.comb_iter_2_left = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_3_left = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)
            self.comb_iter_3_right = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_4_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=3,
                strides=1,
                padding=1)

    def hybrid_forward(self, F, x, x_prev):
        assert ((not TESTING) or x.shape == (1, 88, 28, 28) or x.shape == (1, 352, 14, 14) or x.shape == (1, 704, 7, 7))
        assert ((not TESTING) or x_prev.shape == (1, 44, 56, 56) or x_prev.shape == (1, 264, 28, 28) or x_prev.shape == (1, 528, 14, 14))

        x_relu = self.relu(x_prev)

        # path 1
        x_path1 = self.path_1(x_relu)

        # path 2
        x_path2 = process_with_padding(x_relu, F)
        x_path2 = self.path_2(x_path2)

        # final path
        x_left = self.final_path_bn(F.concat(*[x_path1, x_path2], dim=1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = F.concat(*[x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out


class NormalCell(HybridBlock):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right,
                 **kwargs):
        super(NormalCell, self).__init__(**kwargs)

        with self.name_scope():
            self.conv_prev_1x1 = ConvBlock(
                in_channels=in_channels_left,
                out_channels=out_channels_left,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.conv_1x1 = ConvBlock(
                in_channels=in_channels_right,
                out_channels=out_channels_right,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.comb_iter_0_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=1,
                padding=2)
            self.comb_iter_0_right = BranchSeparables(
                in_channels=out_channels_left,
                out_channels=out_channels_left,
                kernel_size=3,
                strides=1,
                padding=1)

            self.comb_iter_1_left = BranchSeparables(
                in_channels=out_channels_left,
                out_channels=out_channels_left,
                kernel_size=5,
                strides=1,
                padding=2)
            self.comb_iter_1_right = BranchSeparables(
                in_channels=out_channels_left,
                out_channels=out_channels_left,
                kernel_size=3,
                strides=1,
                padding=1)

            self.comb_iter_2_left = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_3_left = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)
            self.comb_iter_3_right = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_4_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=3,
                strides=1,
                padding=1)

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = F.concat(*[x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out


class ReductionCell0(HybridBlock):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right,
                 **kwargs):
        super(ReductionCell0, self).__init__(**kwargs)

        with self.name_scope():
            self.conv_prev_1x1 = ConvBlock(
                in_channels=in_channels_left,
                out_channels=out_channels_left,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.conv_1x1 = ConvBlock(
                in_channels=in_channels_right,
                out_channels=out_channels_right,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.comb_iter_0_left = BranchSeparablesReduction(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=2,
                padding=2)
            self.comb_iter_0_right = BranchSeparablesReduction(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=7,
                strides=2,
                padding=3)

            self.comb_iter_1_left = MaxPoolPad()
            self.comb_iter_1_right = BranchSeparablesReduction(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=7,
                strides=2,
                padding=3)

            self.comb_iter_2_left = AvgPoolPad()
            self.comb_iter_2_right = BranchSeparablesReduction(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=2,
                padding=2)

            self.comb_iter_3_right = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_4_left = BranchSeparablesReduction(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=3,
                strides=1,
                padding=1)
            self.comb_iter_4_right = MaxPoolPad()

    def hybrid_forward(self, F, x, x_prev):
        assert ((not TESTING) or x.shape == (1, 264, 28, 28))
        assert ((not TESTING) or x_prev.shape == (1, 264, 28, 28))

        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out


class ReductionCell1(HybridBlock):

    def __init__(self,
                 in_channels_left,
                 out_channels_left,
                 in_channels_right,
                 out_channels_right,
                 **kwargs):
        super(ReductionCell1, self).__init__(**kwargs)

        with self.name_scope():
            self.conv_prev_1x1 = ConvBlock(
                in_channels=in_channels_left,
                out_channels=out_channels_left,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.conv_1x1 = ConvBlock(
                in_channels=in_channels_right,
                out_channels=out_channels_right,
                kernel_size=1,
                strides=1,
                use_bias=False)

            self.comb_iter_0_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=2,
                padding=2)
            self.comb_iter_0_right = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=7,
                strides=2,
                padding=3)

            self.comb_iter_1_left = MaxPoolPad()
            self.comb_iter_1_right = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=7,
                strides=2,
                padding=3)

            self.comb_iter_2_left = AvgPoolPad()
            self.comb_iter_2_right = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=5,
                strides=2,
                padding=2)

            self.comb_iter_3_right = nn.AvgPool2D(
                pool_size=3,
                strides=1,
                padding=1,
                count_include_pad=False)

            self.comb_iter_4_left = BranchSeparables(
                in_channels=out_channels_right,
                out_channels=out_channels_right,
                kernel_size=3,
                strides=1,
                padding=1)
            self.comb_iter_4_right = MaxPoolPad()

    def hybrid_forward(self, F, x, x_prev):
        assert ((not TESTING) or x.shape == (1, 528, 14, 14))
        assert ((not TESTING) or x_prev.shape == (1, 528, 14, 14))

        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(*[x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], dim=1)
        return x_out


class NASNetInitBlock(HybridBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(NASNetInitBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=3,
                strides=2,
                padding=0,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                momentum=0.1,
                epsilon=0.001,
                in_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NASNet(HybridBlock):

    def __init__(self,
                 cell_repeats,
                 penultimate_filters,
                 classes=1000,
                 **kwargs):
        super(NASNet, self).__init__(**kwargs)

        input_channels = 3
        stem_filters = 32
        filters = penultimate_filters // 24
        filters_multiplier = 2

        with self.name_scope():
            self.conv0 = NASNetInitBlock(
                in_channels=input_channels,
                out_channels=stem_filters)

            self.cell_stem_0 = CellStem0(
                stem_filters=stem_filters,
                num_filters=filters // (filters_multiplier ** 2))
            self.cell_stem_1 = CellStem1(
                stem_filters=stem_filters,
                num_filters=filters // filters_multiplier)

            self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters//2,  # 1, 0.5
                                    in_channels_right=2*filters, out_channels_right=filters)  # 2, 1
            self.cell_1 = NormalCell(in_channels_left=2*filters, out_channels_left=filters,  # 2, 1
                                     in_channels_right=6*filters, out_channels_right=filters)  # 6, 1
            self.cell_2 = NormalCell(in_channels_left=6*filters, out_channels_left=filters,  # 6, 1
                                     in_channels_right=6*filters, out_channels_right=filters)  # 6, 1
            self.cell_3 = NormalCell(in_channels_left=6*filters, out_channels_left=filters,  # 6, 1
                                     in_channels_right=6*filters, out_channels_right=filters)  # 6, 1

            self.reduction_cell_0 = ReductionCell0(in_channels_left=6*filters, out_channels_left=2*filters,  # 6, 2
                                                   in_channels_right=6*filters, out_channels_right=2*filters)  # 6, 2

            self.cell_6 = FirstCell(in_channels_left=6*filters, out_channels_left=filters,  # 6, 1
                                    in_channels_right=8*filters, out_channels_right=2*filters)  # 8, 2
            self.cell_7 = NormalCell(in_channels_left=8*filters, out_channels_left=2*filters,  # 8, 2
                                     in_channels_right=12*filters, out_channels_right=2*filters)  # 12, 2
            self.cell_8 = NormalCell(in_channels_left=12*filters, out_channels_left=2*filters,  # 12, 2
                                     in_channels_right=12*filters, out_channels_right=2*filters)  # 12, 2
            self.cell_9 = NormalCell(in_channels_left=12*filters, out_channels_left=2*filters,  # 12, 2
                                     in_channels_right=12*filters, out_channels_right=2*filters)  # 12, 2

            self.reduction_cell_1 = ReductionCell1(in_channels_left=12*filters, out_channels_left=4*filters,  # 12, 4
                                                   in_channels_right=12*filters, out_channels_right=4*filters)  # 12, 4

            self.cell_12 = FirstCell(in_channels_left=12*filters, out_channels_left=2*filters,  # 12, 2
                                     in_channels_right=16*filters, out_channels_right=4*filters)  # 16, 4
            self.cell_13 = NormalCell(in_channels_left=16*filters, out_channels_left=4*filters,  # 16, 4
                                      in_channels_right=24*filters, out_channels_right=4*filters)  # 24, 4
            self.cell_14 = NormalCell(in_channels_left=24*filters, out_channels_left=4*filters,  # 24, 4
                                      in_channels_right=24*filters, out_channels_right=4*filters)  # 24, 4
            self.cell_15 = NormalCell(in_channels_left=24*filters, out_channels_left=4*filters,  # 24, 4
                                      in_channels_right=24*filters, out_channels_right=4*filters)  # 24, 4

            self.activ = nn.Activation(activation='relu')
            self.avg_pool = nn.AvgPool2D(pool_size=7, strides=1, padding=0)
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(rate=0.5)
            self.output = nn.Dense(
                units=classes,
                in_units=penultimate_filters)

    def features(self, x):
        assert ((not TESTING) or x.shape == (1, 3, 224, 224))

        x_conv0 = self.conv0(x)
        assert ((not TESTING) or x_conv0.shape == (1, 32, 111, 111))
        x_stem_0 = self.cell_stem_0(x_conv0)
        assert ((not TESTING) or x_stem_0.shape == (1, 44, 56, 56))
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        assert ((not TESTING) or x_stem_1.shape == (1, 88, 28, 28))

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        assert ((not TESTING) or x_cell_0.shape == (1, 264, 28, 28))
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        assert ((not TESTING) or x_cell_1.shape == (1, 264, 28, 28))
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        assert ((not TESTING) or x_cell_2.shape == (1, 264, 28, 28))
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        assert ((not TESTING) or x_cell_3.shape == (1, 264, 28, 28))

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        assert ((not TESTING) or x_reduction_cell_0.shape == (1, 352, 14, 14))

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        assert ((not TESTING) or x_cell_6.shape == (1, 528, 14, 14))
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        assert ((not TESTING) or x_cell_7.shape == (1, 528, 14, 14))
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        assert ((not TESTING) or x_cell_8.shape == (1, 528, 14, 14))
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        assert ((not TESTING) or x_cell_9.shape == (1, 528, 14, 14))

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        assert ((not TESTING) or x_reduction_cell_1.shape == (1, 704, 7, 7))

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        assert ((not TESTING) or x_cell_12.shape == (1, 1056, 7, 7))
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        assert ((not TESTING) or x_cell_13.shape == (1, 1056, 7, 7))
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        assert ((not TESTING) or x_cell_14.shape == (1, 1056, 7, 7))
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        assert ((not TESTING) or x_cell_15.shape == (1, 1056, 7, 7))

        return x_cell_15

    def classifier(self, x):
        assert ((not TESTING) or x.shape == (1, 1056, 7, 7))

        x = self.activ(x)
        assert ((not TESTING) or x.shape == (1, 1056, 7, 7))
        x = self.avg_pool(x)
        assert ((not TESTING) or x.shape == (1, 1056, 1, 1))
        x = self.flatten(x)
        assert ((not TESTING) or x.shape == (1, 1056))
        x = self.dropout(x)
        assert ((not TESTING) or x.shape == (1, 1056))
        x = self.output(x)
        assert ((not TESTING) or x.shape == (1, 1000))

        return x

    def hybrid_forward(self, F, x):
        assert ((not TESTING) or x.shape == (1, 3, 224, 224))

        x = self.features(x)
        assert ((not TESTING) or x.shape == (1, 1056, 7, 7))

        x = self.classifier(x)
        assert ((not TESTING) or x.shape == (1, 1000))

        return x


def get_nasnet(cell_repeats,
               penultimate_filters,
               pretrained=False,
               ctx=cpu(),
               **kwargs):

    if pretrained:
        raise ValueError("Pretrained model is not supported")

    net = NASNet(
        cell_repeats=cell_repeats,
        penultimate_filters=penultimate_filters,
        **kwargs)
    return net


def nasnet_a_mobile(**kwargs):
    return get_nasnet(4, 1056, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    global TESTING
    TESTING = True

    net = nasnet_a_mobile()

    ctx = mx.cpu()
    net.initialize(ctx=ctx)

    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    assert (weight_count == 5289978)

    x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    y = net(x)
    assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()

