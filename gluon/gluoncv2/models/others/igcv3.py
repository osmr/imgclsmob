# coding=utf-8
## initial script written by <zhangcycat@gmail.com>,
## list of changes made by liangfu <liangfu.chen@harman.com> :
# 1. add multiplier argument
# 2. add an argument to global average pooling
# 3. make number of expansion filters depend on the input tensor shape

import mxnet as mx
def relu6(data, prefix):
    return mx.sym.clip(data,0,6,name='%s-relu6'%prefix)

def shortcut(data_in, data_residual, prefix):
    out=mx.sym.elemwise_add(data_in, data_residual, name='%s-shortcut'%prefix)
    return out

def permutation(data, groups):
	data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
	data = mx.sym.swapaxes(data, 1, 2)
	data = mx.sym.reshape(data, shape=(0, -3, -2))
	return data 

def mobilenet_unit(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, if_act=True, prefix=''):
    conv = mx.sym.Convolution(
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        num_group=num_group,
        stride=stride,
        pad=pad,
        no_bias=True,
        name='%s-conv2d'%prefix)
    bn = mx.sym.BatchNorm(data=conv, name='%s-batchnorm'%prefix, fix_gamma=False, use_global_stats=False, eps=1e-5)
    if if_act:
        act = relu6(bn, prefix)
        return act
    else:
        return bn

def inverted_residual_unit(data, num_in_filter, num_filter, ifshortcut, stride, kernel, pad, expansion_factor, prefix):
    num_expfilter = int(round(num_in_filter*expansion_factor))

    channel_expand = mobilenet_unit(
        data=data,
        num_filter=num_expfilter,
        kernel=(1,1),
        stride=(1,1),
        pad=(0,0),
        num_group=2,
        if_act=False,
        prefix='%s-exp'%prefix,
    )
    channel_expand = permutation(data=channel_expand,groups=2)
    bottleneck_conv = mobilenet_unit(
        data= channel_expand,
        num_filter=num_expfilter,
        stride=stride,
        kernel=kernel,
        pad=pad,
        num_group=num_expfilter,
        if_act=True,
        prefix='%s-depthwise'%prefix,
    )
    linear_out = mobilenet_unit(
        data=bottleneck_conv,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=2,
        if_act=False,
        prefix='%s-linear'%prefix
    )
#     linear_out = permutation(data=linear_out,groups=num_filter/2)
    if ifshortcut:
        out = shortcut(
            data_in=data,
            data_residual=linear_out,
            prefix=prefix,
        ) 
        return out
    else:
        return linear_out

def NLinverted_residual_unit(data, num_in_filter, num_filter, ifshortcut, stride, kernel, pad, expansion_factor, prefix):
    num_expfilter = int(round(num_in_filter*expansion_factor))

    channel_expand = mobilenet_unit(
        data=data,
        num_filter=num_expfilter,
        kernel=(1,1),
        stride=(1,1),
        pad=(0,0),
        num_group=1,
        if_act=True,
        prefix='%s-exp'%prefix,
    )
    bottleneck_conv = mobilenet_unit(
        data= channel_expand,
        num_filter=num_expfilter,
        stride=stride,
        kernel=kernel,
        pad=pad,
        num_group=num_expfilter,
        if_act=True,
        prefix='%s-depthwise'%prefix,
    )
    linear_out = mobilenet_unit(
        data=bottleneck_conv,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=1,
        if_act=False,
        prefix='%s-linear'%prefix
    )
    if ifshortcut:
        out = shortcut(
            data_in=data,
            data_residual=linear_out,
            prefix=prefix,
        ) 
        return out
    else:
        return linear_out

def invresi_blocks(data, in_c, t, c, n, s, prefix):
    first_block = inverted_residual_unit(
        data=data,
        num_in_filter=in_c,
        num_filter=c,
        ifshortcut=False,
        stride=(s,s),
        kernel=(3,3),
        pad=(1,1),
        expansion_factor=t,
        prefix='%s-block0'%prefix
    )
    print(in_c)
    print(c)
    last_residual_block = first_block
    last_c = c

    for i in range(1,n):
        last_residual_block = inverted_residual_unit(
            data=last_residual_block,
            num_in_filter=last_c,
            num_filter=c,
            ifshortcut=True,
            stride=(1,1),
            kernel=(3,3),
            pad=(1,1),
            expansion_factor=t,
            prefix='%s-block%d'%(prefix, i)
        )
    return last_residual_block

MNETV2_CONFIGS_MAP = {
    (224,224):{
        'firstconv_filter_num': 32, # 3*224*224 -> 32*112*112
        # t, c, n, s
        'bottleneck_params_list':[
            (1, 16, 1, 1), # 32x112x112 -> 16x112x112
            (6, 24, 4, 2), # 16x112x112 -> 24x56x56
            (6, 32, 6, 2), # 24x56x56 -> 32x28x28
            (6, 64, 8, 2), # 32x28x28 -> 64x14x14
            (6, 96, 6, 1), # 64x14x14 -> 96x14x14
            (6, 160, 6, 2), # 96x14x14 -> 160x7x7
            (6, 320, 1, 1), # 160x7x7 -> 320x7x7
        ],
        'filter_num_before_gp': 1280, # 320x7x7 -> 1280x7x7
    } 
}
class MNetV2Gen(object):
    def __init__(self, data_wh, multiplier, **kargs):
        super(MNetV2Gen, self).__init__()
        self.data_wh=data_wh
        self.multiplier=multiplier
        if self.data_wh in MNETV2_CONFIGS_MAP:
            self.MNetConfigs=MNETV2_CONFIGS_MAP[self.data_wh]
        else:
            self.MNetConfigs=MNETV2_CONFIGS_MAP[(224, 224)]
    
    def genNet(self, class_num=1000, **configs):
        data = mx.sym.Variable('data')
        self.MNetConfigs.update(configs)
        # first conv2d block
        first_c = int(round(self.MNetConfigs['firstconv_filter_num']*self.multiplier))
        if first_c%2!=0:
            first_c =first_c+1
        first_layer = mobilenet_unit(
            data=data,
            num_filter=first_c,
            kernel=(3,3),
            stride=(2,2),
            pad=(1,1),
            if_act=True,
            prefix='first-3x3-conv'
        )
        last_bottleneck_layer = first_layer
        in_c = first_c
        # bottleneck sequences
        for i, layer_setting in enumerate(self.MNetConfigs['bottleneck_params_list']):
            t, c, n, s = layer_setting
            c=int(round(c*self.multiplier))
            if c%2!=0:
                c=c+1
            print("channels:%d"%c)
            last_bottleneck_layer = invresi_blocks(
                data=last_bottleneck_layer,
                in_c=in_c, t=t, c=c, n=n, s=s, 
                prefix='seq-%d'%i
            )
            in_c = int(round(c*self.multiplier))
            if in_c%2!=0:
                in_c=c+1
        # last conv2d block before global pooling
        num_filter = int(1280 * self.multiplier)
        if num_filter!=0:
            num_filter=num_filter+1
        last_fm = mobilenet_unit(
            data=last_bottleneck_layer,
            num_filter=num_filter if self.multiplier > 1.0 else 1280,
            kernel=(1,1),
            stride=(1,1),
            pad=(0,0),
            if_act=True,
            prefix='last-1x1-conv'
        )
        # global average pooling
        pool_size = int(self.data_wh[0] / 32)
        pool = mx.sym.Pooling(data=last_fm, kernel=(pool_size, pool_size), stride=(1, 1), pool_type="avg", name="global_pool", global_pool=True)
        flatten = mx.sym.Flatten(data=pool, name="flatten")
        fc = mx.symbol.FullyConnected(data=flatten, num_hidden=class_num, name='fc')
        softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
        
        return softmax

    def __call__(self, class_num=1000, layer_out=None, **configs):
        # build the whole architecture of mobilenet v2 here
        sym = self.genNet(class_num=class_num,**configs)
        if layer_out is None:
            return sym

        internals = sym.get_internals()
        if type(layer_out) is list or type(layer_out) is tuple:
            layers_out = [internals[layer_nm.strip() + '_output'] for layer_nm in layer_out]
            return layers_out
        else:
            layer_out = internals[layer_out.strip() + '_output']
            return layer_out

def get_symbol(num_classes=1000, multiplier=1.0):
    mnetgen = MNetV2Gen((224,224), multiplier=multiplier)
    mnetv2_sym = mnetgen(class_num=num_classes, layer_out=None)
    return mnetv2_sym

if __name__ == '__main__':
    mnetgen = MNetV2Gen((224,224))
    class_num=1000

    # classification network 
    layer_out = None
    mnetv2_sym = mnetgen(class_num=class_num, layer_out=layer_out)
    net_plot = mx.viz.plot_network(
        symbol=mnetv2_sym,
        title='mobilenet-v2',
        shape={
            'data':(16,3,224,224),
            'softmax_label':(16,)
        }
    )
    # net_plot.view()
    # specific layer
    layer_out = 'seq-2-block2-shortcut'
    layer_sym = mnetgen(class_num=class_num, layer_out=layer_out)
    layer_plot = mx.viz.plot_network(
        symbol=layer_sym,
        title='layer_out',
        shape={
            'data':(16,3,224,224),
        }
    )
    # check whether bottle neck structure was right
    layer_plot.view()
    # specific layers
    layer_out = [
        'seq-1-block1-shortcut',
        'seq-2-block2-shortcut',
        'seq-3-block3-shortcut',
        'seq-4-block2-shortcut',
        'seq-5-block2-shortcut'
    ]
    layer_sym_ls = mnetgen(class_num=class_num, layer_out=layer_out)
    # no need to plot these layers any more, right? :)
