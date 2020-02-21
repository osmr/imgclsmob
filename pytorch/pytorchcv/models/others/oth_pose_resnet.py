from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class HeatmapMaxDetBlock(nn.Module):
    """
    Heatmap maximum detector block (for human pose estimation task).
    """
    def __init__(self):
        super(HeatmapMaxDetBlock, self).__init__()

    def forward(self, x):
        heatmap = x
        vector_dim = 2
        batch = heatmap.shape[0]
        channels = heatmap.shape[1]
        in_size = x.shape[2:]
        heatmap_vector = heatmap.view(batch, channels, -1)
        scores, indices = heatmap_vector.max(dim=vector_dim, keepdims=True)
        scores_mask = (scores > 0.0).float()
        pts_x = (indices % in_size[1]) * scores_mask
        pts_y = (indices // in_size[1]) * scores_mask
        pts = torch.cat((pts_x, pts_y, scores), dim=vector_dim)
        for b in range(batch):
            for k in range(channels):
                hm = heatmap[b, k, :, :]
                px = int(pts[b, k, 0])
                py = int(pts[b, k, 1])
                if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                    pts[b, k, 0] += (hm[py, px + 1] - hm[py, px - 1]).sign() * 0.25
                    pts[b, k, 1] += (hm[py + 1, px] - hm[py - 1, px]).sign() * 0.25
        return pts

    @staticmethod
    def calc_flops(x):
        assert (x.shape[0] == 1)
        num_flops = x.numel() + 26 * x.shape[1]
        num_macs = 0
        return num_flops, num_macs


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, **kwargs):
        extra_DECONV_WITH_BIAS = False
        extra_NUM_DECONV_LAYERS = 3
        extra_NUM_DECONV_FILTERS = [256, 256, 256]
        extra_NUM_DECONV_KERNELS = [4, 4, 4]
        extra_FINAL_CONV_KERNEL = 1
        MODEL_NUM_JOINTS = 17


        self.inplanes = 64
        self.deconv_with_bias = extra_DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra_NUM_DECONV_LAYERS,
            extra_NUM_DECONV_FILTERS,
            extra_NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra_NUM_DECONV_FILTERS[-1],
            out_channels=MODEL_NUM_JOINTS,
            kernel_size=extra_FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra_FINAL_CONV_KERNEL == 3 else 0
        )

        self.heatmap_max_det = HeatmapMaxDetBlock()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
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

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        keypoints = self.heatmap_max_det(x)
        return keypoints

        # return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, **kwargs):
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, **kwargs)

    return model


def oth_pose_coco_resnet_50_256x192(pretrained=False, **kwargs):
    model = get_pose_net(num_layers=50, **kwargs)
    return model


def oth_pose_coco_resnet_50_384x288(pretrained=False, **kwargs):
    model = get_pose_net(num_layers=50, **kwargs)
    return model


def oth_pose_coco_resnet_101_256x192(pretrained=False, **kwargs):
    model = get_pose_net(num_layers=101, **kwargs)
    return model


def oth_pose_coco_resnet_101_384x288(pretrained=False, **kwargs):
    model = get_pose_net(num_layers=101, **kwargs)
    return model


def oth_pose_coco_resnet_152_256x192(pretrained=False, **kwargs):
    model = get_pose_net(num_layers=152, **kwargs)
    return model


def oth_pose_coco_resnet_152_384x288(pretrained=False, **kwargs):
    model = get_pose_net(num_layers=152, **kwargs)
    return model


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
        oth_pose_coco_resnet_50_256x192,
        oth_pose_coco_resnet_50_384x288,
        oth_pose_coco_resnet_101_256x192,
        oth_pose_coco_resnet_101_384x288,
        oth_pose_coco_resnet_152_256x192,
        oth_pose_coco_resnet_152_384x288,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_pose_coco_resnet_50_256x192 or weight_count == 33999697)
        assert (model != oth_pose_coco_resnet_50_384x288 or weight_count == 33999697)

        x = torch.randn(14, 3, 1024, 2048)
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (14, 19, 1024, 2048))


if __name__ == "__main__":
    _test()
