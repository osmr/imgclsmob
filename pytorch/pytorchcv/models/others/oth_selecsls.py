import torch
import torch.nn as nn

__all__ = ['oth_selecsls42b', 'oth_selecsls60', 'oth_selecsls60b']


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (3, 3),
#         'crop_pct': 0.875, 'interpolation': 'bilinear',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'stem', 'classifier': 'fc',
#         **kwargs
#     }


def conv_bn(in_chs,
            out_chs,
            k=3,
            stride=1,
            padding=None,
            dilation=1):
    if padding is None:
        padding = ((stride - 1) + dilation * (k - 1)) // 2
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, k, stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True)
    )


class SelecSLSBlock(nn.Module):
    def __init__(self,
                 in_chs,
                 skip_chs,
                 mid_chs,
                 out_chs,
                 is_first,
                 stride,
                 dilation=1):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]

        # Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = conv_bn(in_chs, mid_chs, 3, stride, dilation=dilation)
        self.conv2 = conv_bn(mid_chs, mid_chs, 1)
        self.conv3 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv4 = conv_bn(mid_chs // 2, mid_chs, 1)
        self.conv5 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv6 = conv_bn(2 * mid_chs + (0 if is_first else skip_chs), out_chs, 1)

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) in [1, 2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.is_first:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


class SelecSLS(nn.Module):
    """SelecSLS42 / SelecSLS60 / SelecSLS84

    Parameters
    ----------
    cfg : network config dictionary specifying block type, feature, and head args
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    """
    def __init__(self, cfg, num_classes=1000, in_chans=3, in_channels=3, in_size=(224, 224)):
        self.in_size = in_size
        self.num_classes = num_classes
        super(SelecSLS, self).__init__()

        self.stem = conv_bn(
            in_chans,
            32,
            stride=2)
        self.features = nn.Sequential(*[SelecSLSBlock(*block_args) for block_args in cfg['features']])
        self.head = nn.Sequential(*[conv_bn(*conv_args) for conv_args in cfg['head']])
        self.num_features = cfg['num_features']

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            self.num_features,
            num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x


def _create_model(variant, pretrained=False, in_size=None, model_kwargs=None):
    cfg = {}
    if variant.startswith('selecsls42'):
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2),
            (64, 64, 64, 128, False, 1),
            (128, 0, 144, 144, True, 2),
            (144, 144, 144, 288, False, 1),
            (288, 0, 304, 304, True, 2),
            (304, 304, 304, 480, False, 1),
        ]
        # Head can be replaced with alternative configurations depending on the problem
        if variant == 'selecsls42b':
            cfg['head'] = [
                (480, 960, 3, 2),
                (960, 1024, 3, 1),
                (1024, 1280, 3, 2),
                (1280, 1024, 1, 1),
            ]
            cfg['num_features'] = 1024
        else:
            cfg['head'] = [
                (480, 960, 3, 2),
                (960, 1024, 3, 1),
                (1024, 1024, 3, 2),
                (1024, 1280, 1, 1),
            ]
            cfg['num_features'] = 1280
    elif variant.startswith('selecsls60'):
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2),
            (64, 64, 64, 128, False, 1),
            (128, 0, 128, 128, True, 2),
            (128, 128, 128, 128, False, 1),
            (128, 128, 128, 288, False, 1),
            (288, 0, 288, 288, True, 2),
            (288, 288, 288, 288, False, 1),
            (288, 288, 288, 288, False, 1),
            (288, 288, 288, 416, False, 1),
        ]
        # Head can be replaced with alternative configurations depending on the problem
        if variant == 'selecsls60b':
            cfg['head'] = [
                (416, 756, 3, 2),
                (756, 1024, 3, 1),
                (1024, 1280, 3, 2),
                (1280, 1024, 1, 1),
            ]
            cfg['num_features'] = 1024
        else:
            cfg['head'] = [
                (416, 756, 3, 2),
                (756, 1024, 3, 1),
                (1024, 1024, 3, 2),
                (1024, 1280, 1, 1),
            ]
            cfg['num_features'] = 1280
    elif variant == 'selecsls84':
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2),
            (64, 64, 64, 144, False, 1),
            (144, 0, 144, 144, True, 2),
            (144, 144, 144, 144, False, 1),
            (144, 144, 144, 144, False, 1),
            (144, 144, 144, 144, False, 1),
            (144, 144, 144, 304, False, 1),
            (304, 0, 304, 304, True, 2),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 304, False, 1),
            (304, 304, 304, 512, False, 1),
        ]
        # Head can be replaced with alternative configurations depending on the problem
        cfg['head'] = [
            (512, 960, 3, 2),
            (960, 1024, 3, 1),
            (1024, 1024, 3, 2),
            (1024, 1280, 3, 1),
        ]
        cfg['num_features'] = 1280
    else:
        raise ValueError('Invalid net configuration ' + variant + ' !!!')

    model = SelecSLS(cfg, **model_kwargs)
    return model


def oth_selecsls42b(pretrained=False, **kwargs):
    return _create_model('selecsls42b', pretrained=pretrained, model_kwargs=kwargs)


def oth_selecsls60(pretrained=False, **kwargs):
    return _create_model('selecsls60', pretrained=pretrained, model_kwargs=kwargs)


def oth_selecsls60b(pretrained=False, **kwargs):
    return _create_model('selecsls60b', pretrained=pretrained, model_kwargs=kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        oth_selecsls42b,
        oth_selecsls60,
        oth_selecsls60b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_selecsls42b or weight_count == 32458248)
        assert (model != oth_selecsls60 or weight_count == 30670768)
        assert (model != oth_selecsls60b or weight_count == 32774064)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
