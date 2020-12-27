from .models.alexnet import *
from .models.zfnet import *
from .models.vgg import *
from .models.bninception import *
from .models.resnet import *
from .models.preresnet import *
from .models.resnext import *
from .models.seresnet import *
from .models.sepreresnet import *
from .models.seresnext import *
from .models.senet import *
from .models.resnesta import *
from .models.ibnresnet import *
from .models.ibnbresnet import *
from .models.ibnresnext import *
from .models.ibndensenet import *
from .models.airnet import *
from .models.airnext import *
from .models.bamresnet import *
from .models.cbamresnet import *
from .models.scnet import *
from .models.regnet import *
from .models.pyramidnet import *
from .models.diracnetv2 import *
from .models.densenet import *
from .models.peleenet import *
from .models.wrn import *
from .models.drn import *
from .models.dpn import *
from .models.darknet import *
from .models.darknet53 import *
from .models.bagnet import *
from .models.dla import *
from .models.hrnet import *
from .models.vovnet import *
from .models.selecsls import *
from .models.hardnet import *
from .models.squeezenet import *
from .models.squeezenext import *
from .models.shufflenet import *
from .models.shufflenetv2 import *
from .models.shufflenetv2b import *
from .models.menet import *
from .models.mobilenet import *
from .models.mobilenetb import *
from .models.fdmobilenet import *
from .models.mobilenetv2 import *
from .models.mobilenetv3 import *
from .models.igcv3 import *
from .models.ghostnet import *
from .models.mnasnet import *
from .models.proxylessnas import *
from .models.fbnet import *
from .models.xception import *
from .models.inceptionv3 import *
from .models.inceptionv4 import *
from .models.inceptionresnetv2 import *
from .models.polynet import *
from .models.nasnet import *
from .models.pnasnet import *
from .models.spnasnet import *
from .models.efficientnet import *
from .models.efficientnetedge import *
from .models.mixnet import *

from .models.resnet_cifar import *
from .models.preresnet_cifar import *
from .models.resnext_cifar import *
from .models.seresnet_cifar import *
from .models.sepreresnet_cifar import *
from .models.pyramidnet_cifar import *
from .models.densenet_cifar import *
from .models.wrn_cifar import *

from .models.resneta import *
from .models.resnetd import *
from .models.fastseresnet import *

from .models.resnet_cub import *
from .models.seresnet_cub import *
from .models.mobilenet_cub import *
from .models.proxylessnas_cub import *

from .models.fcn8sd import *
from .models.pspnet import *
from .models.deeplabv3 import *
from .models.icnet import *
from .models.fastscnn import *
from .models.sinet import *
from .models.bisenet import *
from .models.danet import *

from .models.alphapose_coco import *
from .models.simplepose_coco import *
from .models.simpleposemobile_coco import *
from .models.lwopenpose_cmupan import *
from .models.ibppose_coco import *
from .models.grmiposelite_coco import *

from .models.centernet import *
from .models.lffd import *

from .models.voca import *
from .models.nvpattexp import *

__all__ = ['get_model']


_models = {
    'alexnet': alexnet,
    'alexnetb': alexnetb,

    'zfnet': zfnet,
    'zfnetb': zfnetb,

    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'bn_vgg11': bn_vgg11,
    'bn_vgg13': bn_vgg13,
    'bn_vgg16': bn_vgg16,
    'bn_vgg19': bn_vgg19,
    'bn_vgg11b': bn_vgg11b,
    'bn_vgg13b': bn_vgg13b,
    'bn_vgg16b': bn_vgg16b,
    'bn_vgg19b': bn_vgg19b,

    'bninception': bninception,

    'resnet10': resnet10,
    'resnet12': resnet12,
    'resnet14': resnet14,
    'resnetbc14b': resnetbc14b,
    'resnet16': resnet16,
    'resnet18_wd4': resnet18_wd4,
    'resnet18_wd2': resnet18_wd2,
    'resnet18_w3d4': resnet18_w3d4,
    'resnet18': resnet18,
    'resnet26': resnet26,
    'resnetbc26b': resnetbc26b,
    'resnet34': resnet34,
    'resnetbc38b': resnetbc38b,
    'resnet50': resnet50,
    'resnet50b': resnet50b,
    'resnet101': resnet101,
    'resnet101b': resnet101b,
    'resnet152': resnet152,
    'resnet152b': resnet152b,
    'resnet200': resnet200,
    'resnet200b': resnet200b,

    'preresnet10': preresnet10,
    'preresnet12': preresnet12,
    'preresnet14': preresnet14,
    'preresnetbc14b': preresnetbc14b,
    'preresnet16': preresnet16,
    'preresnet18_wd4': preresnet18_wd4,
    'preresnet18_wd2': preresnet18_wd2,
    'preresnet18_w3d4': preresnet18_w3d4,
    'preresnet18': preresnet18,
    'preresnet26': preresnet26,
    'preresnetbc26b': preresnetbc26b,
    'preresnet34': preresnet34,
    'preresnetbc38b': preresnetbc38b,
    'preresnet50': preresnet50,
    'preresnet50b': preresnet50b,
    'preresnet101': preresnet101,
    'preresnet101b': preresnet101b,
    'preresnet152': preresnet152,
    'preresnet152b': preresnet152b,
    'preresnet200': preresnet200,
    'preresnet200b': preresnet200b,
    'preresnet269b': preresnet269b,

    'resnext14_16x4d': resnext14_16x4d,
    'resnext14_32x2d': resnext14_32x2d,
    'resnext14_32x4d': resnext14_32x4d,
    'resnext26_16x4d': resnext26_16x4d,
    'resnext26_32x2d': resnext26_32x2d,
    'resnext26_32x4d': resnext26_32x4d,
    'resnext38_32x4d': resnext38_32x4d,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,

    'seresnet10': seresnet10,
    'seresnet12': seresnet12,
    'seresnet14': seresnet14,
    'seresnet16': seresnet16,
    'seresnet18': seresnet18,
    'seresnet26': seresnet26,
    'seresnetbc26b': seresnetbc26b,
    'seresnet34': seresnet34,
    'seresnetbc38b': seresnetbc38b,
    'seresnet50': seresnet50,
    'seresnet50b': seresnet50b,
    'seresnet101': seresnet101,
    'seresnet101b': seresnet101b,
    'seresnet152': seresnet152,
    'seresnet152b': seresnet152b,
    'seresnet200': seresnet200,
    'seresnet200b': seresnet200b,

    'sepreresnet10': sepreresnet10,
    'sepreresnet12': sepreresnet12,
    'sepreresnet14': sepreresnet14,
    'sepreresnet16': sepreresnet16,
    'sepreresnet18': sepreresnet18,
    'sepreresnet26': sepreresnet26,
    'sepreresnetbc26b': sepreresnetbc26b,
    'sepreresnet34': sepreresnet34,
    'sepreresnetbc38b': sepreresnetbc38b,
    'sepreresnet50': sepreresnet50,
    'sepreresnet50b': sepreresnet50b,
    'sepreresnet101': sepreresnet101,
    'sepreresnet101b': sepreresnet101b,
    'sepreresnet152': sepreresnet152,
    'sepreresnet152b': sepreresnet152b,
    'sepreresnet200': sepreresnet200,
    'sepreresnet200b': sepreresnet200b,

    'seresnext50_32x4d': seresnext50_32x4d,
    'seresnext101_32x4d': seresnext101_32x4d,
    'seresnext101_64x4d': seresnext101_64x4d,

    'senet16': senet16,
    'senet28': senet28,
    'senet40': senet40,
    'senet52': senet52,
    'senet103': senet103,
    'senet154': senet154,

    'resnestabc14': resnestabc14,
    'resnesta18': resnesta18,
    'resnestabc26': resnestabc26,
    'resnesta50': resnesta50,
    'resnesta101': resnesta101,
    'resnesta152': resnesta152,
    'resnesta200': resnesta200,
    'resnesta269': resnesta269,

    'ibn_resnet50': ibn_resnet50,
    'ibn_resnet101': ibn_resnet101,
    'ibn_resnet152': ibn_resnet152,

    'ibnb_resnet50': ibnb_resnet50,
    'ibnb_resnet101': ibnb_resnet101,
    'ibnb_resnet152': ibnb_resnet152,

    'ibn_resnext50_32x4d': ibn_resnext50_32x4d,
    'ibn_resnext101_32x4d': ibn_resnext101_32x4d,
    'ibn_resnext101_64x4d': ibn_resnext101_64x4d,

    'ibn_densenet121': ibn_densenet121,
    'ibn_densenet161': ibn_densenet161,
    'ibn_densenet169': ibn_densenet169,
    'ibn_densenet201': ibn_densenet201,

    'airnet50_1x64d_r2': airnet50_1x64d_r2,
    'airnet50_1x64d_r16': airnet50_1x64d_r16,
    'airnet101_1x64d_r2': airnet101_1x64d_r2,

    'airnext50_32x4d_r2': airnext50_32x4d_r2,
    'airnext101_32x4d_r2': airnext101_32x4d_r2,
    'airnext101_32x4d_r16': airnext101_32x4d_r16,

    'bam_resnet18': bam_resnet18,
    'bam_resnet34': bam_resnet34,
    'bam_resnet50': bam_resnet50,
    'bam_resnet101': bam_resnet101,
    'bam_resnet152': bam_resnet152,

    'cbam_resnet18': cbam_resnet18,
    'cbam_resnet34': cbam_resnet34,
    'cbam_resnet50': cbam_resnet50,
    'cbam_resnet101': cbam_resnet101,
    'cbam_resnet152': cbam_resnet152,

    'scnet50': scnet50,
    'scnet101': scnet101,
    'scneta50': scneta50,
    'scneta101': scneta101,

    'regnetx002': regnetx002,
    'regnetx004': regnetx004,
    'regnetx006': regnetx006,
    'regnetx008': regnetx008,
    'regnetx016': regnetx016,
    'regnetx032': regnetx032,
    'regnetx040': regnetx040,
    'regnetx064': regnetx064,
    'regnetx080': regnetx080,
    'regnetx120': regnetx120,
    'regnetx160': regnetx160,
    'regnetx320': regnetx320,

    'regnety002': regnety002,
    'regnety004': regnety004,
    'regnety006': regnety006,
    'regnety008': regnety008,
    'regnety016': regnety016,
    'regnety032': regnety032,
    'regnety040': regnety040,
    'regnety064': regnety064,
    'regnety080': regnety080,
    'regnety120': regnety120,
    'regnety160': regnety160,
    'regnety320': regnety320,

    'pyramidnet101_a360': pyramidnet101_a360,

    'diracnet18v2': diracnet18v2,
    'diracnet34v2': diracnet34v2,

    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,

    'peleenet': peleenet,

    'wrn50_2': wrn50_2,

    'drnc26': drnc26,
    'drnc42': drnc42,
    'drnc58': drnc58,
    'drnd22': drnd22,
    'drnd38': drnd38,
    'drnd54': drnd54,
    'drnd105': drnd105,

    'dpn68': dpn68,
    'dpn68b': dpn68b,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,

    'darknet_ref': darknet_ref,
    'darknet_tiny': darknet_tiny,
    'darknet19': darknet19,
    'darknet53': darknet53,

    'bagnet9': bagnet9,
    'bagnet17': bagnet17,
    'bagnet33': bagnet33,

    'dla34': dla34,
    'dla46c': dla46c,
    'dla46xc': dla46xc,
    'dla60': dla60,
    'dla60x': dla60x,
    'dla60xc': dla60xc,
    'dla102': dla102,
    'dla102x': dla102x,
    'dla102x2': dla102x2,
    'dla169': dla169,

    'hrnet_w18_small_v1': hrnet_w18_small_v1,
    'hrnet_w18_small_v2': hrnet_w18_small_v2,
    'hrnetv2_w18': hrnetv2_w18,
    'hrnetv2_w30': hrnetv2_w30,
    'hrnetv2_w32': hrnetv2_w32,
    'hrnetv2_w40': hrnetv2_w40,
    'hrnetv2_w44': hrnetv2_w44,
    'hrnetv2_w48': hrnetv2_w48,
    'hrnetv2_w64': hrnetv2_w64,

    'vovnet27s': vovnet27s,
    'vovnet39': vovnet39,
    'vovnet57': vovnet57,

    'selecsls42': selecsls42,
    'selecsls42b': selecsls42b,
    'selecsls60': selecsls60,
    'selecsls60b': selecsls60b,
    'selecsls84': selecsls84,

    'hardnet39ds': hardnet39ds,
    'hardnet68ds': hardnet68ds,
    'hardnet68': hardnet68,
    'hardnet85': hardnet85,

    'squeezenet_v1_0': squeezenet_v1_0,
    'squeezenet_v1_1': squeezenet_v1_1,

    'squeezeresnet_v1_0': squeezeresnet_v1_0,
    'squeezeresnet_v1_1': squeezeresnet_v1_1,

    'sqnxt23_w1': sqnxt23_w1,
    'sqnxt23_w3d2': sqnxt23_w3d2,
    'sqnxt23_w2': sqnxt23_w2,
    'sqnxt23v5_w1': sqnxt23v5_w1,
    'sqnxt23v5_w3d2': sqnxt23v5_w3d2,
    'sqnxt23v5_w2': sqnxt23v5_w2,

    'shufflenet_g1_w1': shufflenet_g1_w1,
    'shufflenet_g2_w1': shufflenet_g2_w1,
    'shufflenet_g3_w1': shufflenet_g3_w1,
    'shufflenet_g4_w1': shufflenet_g4_w1,
    'shufflenet_g8_w1': shufflenet_g8_w1,
    'shufflenet_g1_w3d4': shufflenet_g1_w3d4,
    'shufflenet_g3_w3d4': shufflenet_g3_w3d4,
    'shufflenet_g1_wd2': shufflenet_g1_wd2,
    'shufflenet_g3_wd2': shufflenet_g3_wd2,
    'shufflenet_g1_wd4': shufflenet_g1_wd4,
    'shufflenet_g3_wd4': shufflenet_g3_wd4,

    'shufflenetv2_wd2': shufflenetv2_wd2,
    'shufflenetv2_w1': shufflenetv2_w1,
    'shufflenetv2_w3d2': shufflenetv2_w3d2,
    'shufflenetv2_w2': shufflenetv2_w2,

    'shufflenetv2b_wd2': shufflenetv2b_wd2,
    'shufflenetv2b_w1': shufflenetv2b_w1,
    'shufflenetv2b_w3d2': shufflenetv2b_w3d2,
    'shufflenetv2b_w2': shufflenetv2b_w2,

    'menet108_8x1_g3': menet108_8x1_g3,
    'menet128_8x1_g4': menet128_8x1_g4,
    'menet160_8x1_g8': menet160_8x1_g8,
    'menet228_12x1_g3': menet228_12x1_g3,
    'menet256_12x1_g4': menet256_12x1_g4,
    'menet348_12x1_g3': menet348_12x1_g3,
    'menet352_12x1_g8': menet352_12x1_g8,
    'menet456_24x1_g3': menet456_24x1_g3,

    'mobilenet_w1': mobilenet_w1,
    'mobilenet_w3d4': mobilenet_w3d4,
    'mobilenet_wd2': mobilenet_wd2,
    'mobilenet_wd4': mobilenet_wd4,

    'mobilenetb_w1': mobilenetb_w1,
    'mobilenetb_w3d4': mobilenetb_w3d4,
    'mobilenetb_wd2': mobilenetb_wd2,
    'mobilenetb_wd4': mobilenetb_wd4,

    'fdmobilenet_w1': fdmobilenet_w1,
    'fdmobilenet_w3d4': fdmobilenet_w3d4,
    'fdmobilenet_wd2': fdmobilenet_wd2,
    'fdmobilenet_wd4': fdmobilenet_wd4,

    'mobilenetv2_w1': mobilenetv2_w1,
    'mobilenetv2_w3d4': mobilenetv2_w3d4,
    'mobilenetv2_wd2': mobilenetv2_wd2,
    'mobilenetv2_wd4': mobilenetv2_wd4,
    'mobilenetv2b_w1': mobilenetv2b_w1,
    'mobilenetv2b_w3d4': mobilenetv2b_w3d4,
    'mobilenetv2b_wd2': mobilenetv2b_wd2,
    'mobilenetv2b_wd4': mobilenetv2b_wd4,

    'mobilenetv3_small_w7d20': mobilenetv3_small_w7d20,
    'mobilenetv3_small_wd2': mobilenetv3_small_wd2,
    'mobilenetv3_small_w3d4': mobilenetv3_small_w3d4,
    'mobilenetv3_small_w1': mobilenetv3_small_w1,
    'mobilenetv3_small_w5d4': mobilenetv3_small_w5d4,
    'mobilenetv3_large_w7d20': mobilenetv3_large_w7d20,
    'mobilenetv3_large_wd2': mobilenetv3_large_wd2,
    'mobilenetv3_large_w3d4': mobilenetv3_large_w3d4,
    'mobilenetv3_large_w1': mobilenetv3_large_w1,
    'mobilenetv3_large_w5d4': mobilenetv3_large_w5d4,

    'igcv3_w1': igcv3_w1,
    'igcv3_w3d4': igcv3_w3d4,
    'igcv3_wd2': igcv3_wd2,
    'igcv3_wd4': igcv3_wd4,

    'ghostnet': ghostnet,

    'mnasnet_b1': mnasnet_b1,
    'mnasnet_a1': mnasnet_a1,
    'mnasnet_small': mnasnet_small,

    'proxylessnas_cpu': proxylessnas_cpu,
    'proxylessnas_gpu': proxylessnas_gpu,
    'proxylessnas_mobile': proxylessnas_mobile,
    'proxylessnas_mobile14': proxylessnas_mobile14,

    'fbnet_cb': fbnet_cb,

    'xception': xception,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inceptionresnetv2': inceptionresnetv2,
    'polynet': polynet,

    'nasnet_4a1056': nasnet_4a1056,
    'nasnet_6a4032': nasnet_6a4032,

    'pnasnet5large': pnasnet5large,

    'spnasnet': spnasnet,

    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'efficientnet_b8': efficientnet_b8,
    'efficientnet_b0b': efficientnet_b0b,
    'efficientnet_b1b': efficientnet_b1b,
    'efficientnet_b2b': efficientnet_b2b,
    'efficientnet_b3b': efficientnet_b3b,
    'efficientnet_b4b': efficientnet_b4b,
    'efficientnet_b5b': efficientnet_b5b,
    'efficientnet_b6b': efficientnet_b6b,
    'efficientnet_b7b': efficientnet_b7b,
    'efficientnet_b0c': efficientnet_b0c,
    'efficientnet_b1c': efficientnet_b1c,
    'efficientnet_b2c': efficientnet_b2c,
    'efficientnet_b3c': efficientnet_b3c,
    'efficientnet_b4c': efficientnet_b4c,
    'efficientnet_b5c': efficientnet_b5c,
    'efficientnet_b6c': efficientnet_b6c,
    'efficientnet_b7c': efficientnet_b7c,
    'efficientnet_b8c': efficientnet_b8c,

    'efficientnet_edge_small_b': efficientnet_edge_small_b,
    'efficientnet_edge_medium_b': efficientnet_edge_medium_b,
    'efficientnet_edge_large_b': efficientnet_edge_large_b,

    'mixnet_s': mixnet_s,
    'mixnet_m': mixnet_m,
    'mixnet_l': mixnet_l,

    'resnet20_cifar10': resnet20_cifar10,
    'resnet20_cifar100': resnet20_cifar100,
    'resnet20_svhn': resnet20_svhn,
    'resnet56_cifar10': resnet56_cifar10,
    'resnet56_cifar100': resnet56_cifar100,
    'resnet56_svhn': resnet56_svhn,
    'resnet110_cifar10': resnet110_cifar10,
    'resnet110_cifar100': resnet110_cifar100,
    'resnet110_svhn': resnet110_svhn,
    'resnet164bn_cifar10': resnet164bn_cifar10,
    'resnet164bn_cifar100': resnet164bn_cifar100,
    'resnet164bn_svhn': resnet164bn_svhn,
    'resnet272bn_cifar10': resnet272bn_cifar10,
    'resnet272bn_cifar100': resnet272bn_cifar100,
    'resnet272bn_svhn': resnet272bn_svhn,
    'resnet542bn_cifar10': resnet542bn_cifar10,
    'resnet542bn_cifar100': resnet542bn_cifar100,
    'resnet542bn_svhn': resnet542bn_svhn,
    'resnet1001_cifar10': resnet1001_cifar10,
    'resnet1001_cifar100': resnet1001_cifar100,
    'resnet1001_svhn': resnet1001_svhn,
    'resnet1202_cifar10': resnet1202_cifar10,
    'resnet1202_cifar100': resnet1202_cifar100,
    'resnet1202_svhn': resnet1202_svhn,

    'preresnet20_cifar10': preresnet20_cifar10,
    'preresnet20_cifar100': preresnet20_cifar100,
    'preresnet20_svhn': preresnet20_svhn,
    'preresnet56_cifar10': preresnet56_cifar10,
    'preresnet56_cifar100': preresnet56_cifar100,
    'preresnet56_svhn': preresnet56_svhn,
    'preresnet110_cifar10': preresnet110_cifar10,
    'preresnet110_cifar100': preresnet110_cifar100,
    'preresnet110_svhn': preresnet110_svhn,
    'preresnet164bn_cifar10': preresnet164bn_cifar10,
    'preresnet164bn_cifar100': preresnet164bn_cifar100,
    'preresnet164bn_svhn': preresnet164bn_svhn,
    'preresnet272bn_cifar10': preresnet272bn_cifar10,
    'preresnet272bn_cifar100': preresnet272bn_cifar100,
    'preresnet272bn_svhn': preresnet272bn_svhn,
    'preresnet542bn_cifar10': preresnet542bn_cifar10,
    'preresnet542bn_cifar100': preresnet542bn_cifar100,
    'preresnet542bn_svhn': preresnet542bn_svhn,
    'preresnet1001_cifar10': preresnet1001_cifar10,
    'preresnet1001_cifar100': preresnet1001_cifar100,
    'preresnet1001_svhn': preresnet1001_svhn,
    'preresnet1202_cifar10': preresnet1202_cifar10,
    'preresnet1202_cifar100': preresnet1202_cifar100,
    'preresnet1202_svhn': preresnet1202_svhn,

    'resnext20_1x64d_cifar10': resnext20_1x64d_cifar10,
    'resnext20_1x64d_cifar100': resnext20_1x64d_cifar100,
    'resnext20_1x64d_svhn': resnext20_1x64d_svhn,
    'resnext20_2x32d_cifar10': resnext20_2x32d_cifar10,
    'resnext20_2x32d_cifar100': resnext20_2x32d_cifar100,
    'resnext20_2x32d_svhn': resnext20_2x32d_svhn,
    'resnext20_2x64d_cifar10': resnext20_2x64d_cifar10,
    'resnext20_2x64d_cifar100': resnext20_2x64d_cifar100,
    'resnext20_2x64d_svhn': resnext20_2x64d_svhn,
    'resnext20_4x16d_cifar10': resnext20_4x16d_cifar10,
    'resnext20_4x16d_cifar100': resnext20_4x16d_cifar100,
    'resnext20_4x16d_svhn': resnext20_4x16d_svhn,
    'resnext20_4x32d_cifar10': resnext20_4x32d_cifar10,
    'resnext20_4x32d_cifar100': resnext20_4x32d_cifar100,
    'resnext20_4x32d_svhn': resnext20_4x32d_svhn,
    'resnext20_8x8d_cifar10': resnext20_8x8d_cifar10,
    'resnext20_8x8d_cifar100': resnext20_8x8d_cifar100,
    'resnext20_8x8d_svhn': resnext20_8x8d_svhn,
    'resnext20_8x16d_cifar10': resnext20_8x16d_cifar10,
    'resnext20_8x16d_cifar100': resnext20_8x16d_cifar100,
    'resnext20_8x16d_svhn': resnext20_8x16d_svhn,
    'resnext20_16x4d_cifar10': resnext20_16x4d_cifar10,
    'resnext20_16x4d_cifar100': resnext20_16x4d_cifar100,
    'resnext20_16x4d_svhn': resnext20_16x4d_svhn,
    'resnext20_16x8d_cifar10': resnext20_16x8d_cifar10,
    'resnext20_16x8d_cifar100': resnext20_16x8d_cifar100,
    'resnext20_16x8d_svhn': resnext20_16x8d_svhn,
    'resnext20_32x2d_cifar10': resnext20_32x2d_cifar10,
    'resnext20_32x2d_cifar100': resnext20_32x2d_cifar100,
    'resnext20_32x2d_svhn': resnext20_32x2d_svhn,
    'resnext20_32x4d_cifar10': resnext20_32x4d_cifar10,
    'resnext20_32x4d_cifar100': resnext20_32x4d_cifar100,
    'resnext20_32x4d_svhn': resnext20_32x4d_svhn,
    'resnext20_64x1d_cifar10': resnext20_64x1d_cifar10,
    'resnext20_64x1d_cifar100': resnext20_64x1d_cifar100,
    'resnext20_64x1d_svhn': resnext20_64x1d_svhn,
    'resnext20_64x2d_cifar10': resnext20_64x2d_cifar10,
    'resnext20_64x2d_cifar100': resnext20_64x2d_cifar100,
    'resnext20_64x2d_svhn': resnext20_64x2d_svhn,
    'resnext29_32x4d_cifar10': resnext29_32x4d_cifar10,
    'resnext29_32x4d_cifar100': resnext29_32x4d_cifar100,
    'resnext29_32x4d_svhn': resnext29_32x4d_svhn,
    'resnext29_16x64d_cifar10': resnext29_16x64d_cifar10,
    'resnext29_16x64d_cifar100': resnext29_16x64d_cifar100,
    'resnext29_16x64d_svhn': resnext29_16x64d_svhn,
    'resnext56_1x64d_cifar10': resnext56_1x64d_cifar10,
    'resnext56_1x64d_cifar100': resnext56_1x64d_cifar100,
    'resnext56_1x64d_svhn': resnext56_1x64d_svhn,
    'resnext56_2x32d_cifar10': resnext56_2x32d_cifar10,
    'resnext56_2x32d_cifar100': resnext56_2x32d_cifar100,
    'resnext56_2x32d_svhn': resnext56_2x32d_svhn,
    'resnext56_4x16d_cifar10': resnext56_4x16d_cifar10,
    'resnext56_4x16d_cifar100': resnext56_4x16d_cifar100,
    'resnext56_4x16d_svhn': resnext56_4x16d_svhn,
    'resnext56_8x8d_cifar10': resnext56_8x8d_cifar10,
    'resnext56_8x8d_cifar100': resnext56_8x8d_cifar100,
    'resnext56_8x8d_svhn': resnext56_8x8d_svhn,
    'resnext56_16x4d_cifar10': resnext56_16x4d_cifar10,
    'resnext56_16x4d_cifar100': resnext56_16x4d_cifar100,
    'resnext56_16x4d_svhn': resnext56_16x4d_svhn,
    'resnext56_32x2d_cifar10': resnext56_32x2d_cifar10,
    'resnext56_32x2d_cifar100': resnext56_32x2d_cifar100,
    'resnext56_32x2d_svhn': resnext56_32x2d_svhn,
    'resnext56_64x1d_cifar10': resnext56_64x1d_cifar10,
    'resnext56_64x1d_cifar100': resnext56_64x1d_cifar100,
    'resnext56_64x1d_svhn': resnext56_64x1d_svhn,
    'resnext272_1x64d_cifar10': resnext272_1x64d_cifar10,
    'resnext272_1x64d_cifar100': resnext272_1x64d_cifar100,
    'resnext272_1x64d_svhn': resnext272_1x64d_svhn,
    'resnext272_2x32d_cifar10': resnext272_2x32d_cifar10,
    'resnext272_2x32d_cifar100': resnext272_2x32d_cifar100,
    'resnext272_2x32d_svhn': resnext272_2x32d_svhn,

    'seresnet20_cifar10': seresnet20_cifar10,
    'seresnet20_cifar100': seresnet20_cifar100,
    'seresnet20_svhn': seresnet20_svhn,
    'seresnet56_cifar10': seresnet56_cifar10,
    'seresnet56_cifar100': seresnet56_cifar100,
    'seresnet56_svhn': seresnet56_svhn,
    'seresnet110_cifar10': seresnet110_cifar10,
    'seresnet110_cifar100': seresnet110_cifar100,
    'seresnet110_svhn': seresnet110_svhn,
    'seresnet164bn_cifar10': seresnet164bn_cifar10,
    'seresnet164bn_cifar100': seresnet164bn_cifar100,
    'seresnet164bn_svhn': seresnet164bn_svhn,
    'seresnet272bn_cifar10': seresnet272bn_cifar10,
    'seresnet272bn_cifar100': seresnet272bn_cifar100,
    'seresnet272bn_svhn': seresnet272bn_svhn,
    'seresnet542bn_cifar10': seresnet542bn_cifar10,
    'seresnet542bn_cifar100': seresnet542bn_cifar100,
    'seresnet542bn_svhn': seresnet542bn_svhn,
    'seresnet1001_cifar10': seresnet1001_cifar10,
    'seresnet1001_cifar100': seresnet1001_cifar100,
    'seresnet1001_svhn': seresnet1001_svhn,
    'seresnet1202_cifar10': seresnet1202_cifar10,
    'seresnet1202_cifar100': seresnet1202_cifar100,
    'seresnet1202_svhn': seresnet1202_svhn,

    'sepreresnet20_cifar10': sepreresnet20_cifar10,
    'sepreresnet20_cifar100': sepreresnet20_cifar100,
    'sepreresnet20_svhn': sepreresnet20_svhn,
    'sepreresnet56_cifar10': sepreresnet56_cifar10,
    'sepreresnet56_cifar100': sepreresnet56_cifar100,
    'sepreresnet56_svhn': sepreresnet56_svhn,
    'sepreresnet110_cifar10': sepreresnet110_cifar10,
    'sepreresnet110_cifar100': sepreresnet110_cifar100,
    'sepreresnet110_svhn': sepreresnet110_svhn,
    'sepreresnet164bn_cifar10': sepreresnet164bn_cifar10,
    'sepreresnet164bn_cifar100': sepreresnet164bn_cifar100,
    'sepreresnet164bn_svhn': sepreresnet164bn_svhn,
    'sepreresnet272bn_cifar10': sepreresnet272bn_cifar10,
    'sepreresnet272bn_cifar100': sepreresnet272bn_cifar100,
    'sepreresnet272bn_svhn': sepreresnet272bn_svhn,
    'sepreresnet542bn_cifar10': sepreresnet542bn_cifar10,
    'sepreresnet542bn_cifar100': sepreresnet542bn_cifar100,
    'sepreresnet542bn_svhn': sepreresnet542bn_svhn,
    'sepreresnet1001_cifar10': sepreresnet1001_cifar10,
    'sepreresnet1001_cifar100': sepreresnet1001_cifar100,
    'sepreresnet1001_svhn': sepreresnet1001_svhn,
    'sepreresnet1202_cifar10': sepreresnet1202_cifar10,
    'sepreresnet1202_cifar100': sepreresnet1202_cifar100,
    'sepreresnet1202_svhn': sepreresnet1202_svhn,

    'pyramidnet110_a48_cifar10': pyramidnet110_a48_cifar10,
    'pyramidnet110_a48_cifar100': pyramidnet110_a48_cifar100,
    'pyramidnet110_a48_svhn': pyramidnet110_a48_svhn,
    'pyramidnet110_a84_cifar10': pyramidnet110_a84_cifar10,
    'pyramidnet110_a84_cifar100': pyramidnet110_a84_cifar100,
    'pyramidnet110_a84_svhn': pyramidnet110_a84_svhn,
    'pyramidnet110_a270_cifar10': pyramidnet110_a270_cifar10,
    'pyramidnet110_a270_cifar100': pyramidnet110_a270_cifar100,
    'pyramidnet110_a270_svhn': pyramidnet110_a270_svhn,
    'pyramidnet164_a270_bn_cifar10': pyramidnet164_a270_bn_cifar10,
    'pyramidnet164_a270_bn_cifar100': pyramidnet164_a270_bn_cifar100,
    'pyramidnet164_a270_bn_svhn': pyramidnet164_a270_bn_svhn,
    'pyramidnet200_a240_bn_cifar10': pyramidnet200_a240_bn_cifar10,
    'pyramidnet200_a240_bn_cifar100': pyramidnet200_a240_bn_cifar100,
    'pyramidnet200_a240_bn_svhn': pyramidnet200_a240_bn_svhn,
    'pyramidnet236_a220_bn_cifar10': pyramidnet236_a220_bn_cifar10,
    'pyramidnet236_a220_bn_cifar100': pyramidnet236_a220_bn_cifar100,
    'pyramidnet236_a220_bn_svhn': pyramidnet236_a220_bn_svhn,
    'pyramidnet272_a200_bn_cifar10': pyramidnet272_a200_bn_cifar10,
    'pyramidnet272_a200_bn_cifar100': pyramidnet272_a200_bn_cifar100,
    'pyramidnet272_a200_bn_svhn': pyramidnet272_a200_bn_svhn,

    'densenet40_k12_cifar10': densenet40_k12_cifar10,
    'densenet40_k12_cifar100': densenet40_k12_cifar100,
    'densenet40_k12_svhn': densenet40_k12_svhn,
    'densenet40_k12_bc_cifar10': densenet40_k12_bc_cifar10,
    'densenet40_k12_bc_cifar100': densenet40_k12_bc_cifar100,
    'densenet40_k12_bc_svhn': densenet40_k12_bc_svhn,
    'densenet40_k24_bc_cifar10': densenet40_k24_bc_cifar10,
    'densenet40_k24_bc_cifar100': densenet40_k24_bc_cifar100,
    'densenet40_k24_bc_svhn': densenet40_k24_bc_svhn,
    'densenet40_k36_bc_cifar10': densenet40_k36_bc_cifar10,
    'densenet40_k36_bc_cifar100': densenet40_k36_bc_cifar100,
    'densenet40_k36_bc_svhn': densenet40_k36_bc_svhn,
    'densenet100_k12_cifar10': densenet100_k12_cifar10,
    'densenet100_k12_cifar100': densenet100_k12_cifar100,
    'densenet100_k12_svhn': densenet100_k12_svhn,
    'densenet100_k24_cifar10': densenet100_k24_cifar10,
    'densenet100_k24_cifar100': densenet100_k24_cifar100,
    'densenet100_k24_svhn': densenet100_k24_svhn,
    'densenet100_k12_bc_cifar10': densenet100_k12_bc_cifar10,
    'densenet100_k12_bc_cifar100': densenet100_k12_bc_cifar100,
    'densenet100_k12_bc_svhn': densenet100_k12_bc_svhn,
    'densenet190_k40_bc_cifar10': densenet190_k40_bc_cifar10,
    'densenet190_k40_bc_cifar100': densenet190_k40_bc_cifar100,
    'densenet190_k40_bc_svhn': densenet190_k40_bc_svhn,
    'densenet250_k24_bc_cifar10': densenet250_k24_bc_cifar10,
    'densenet250_k24_bc_cifar100': densenet250_k24_bc_cifar100,
    'densenet250_k24_bc_svhn': densenet250_k24_bc_svhn,

    'wrn16_10_cifar10': wrn16_10_cifar10,
    'wrn16_10_cifar100': wrn16_10_cifar100,
    'wrn16_10_svhn': wrn16_10_svhn,
    'wrn28_10_cifar10': wrn28_10_cifar10,
    'wrn28_10_cifar100': wrn28_10_cifar100,
    'wrn28_10_svhn': wrn28_10_svhn,
    'wrn40_8_cifar10': wrn40_8_cifar10,
    'wrn40_8_cifar100': wrn40_8_cifar100,
    'wrn40_8_svhn': wrn40_8_svhn,

    'resneta10': resneta10,
    'resnetabc14b': resnetabc14b,
    'resneta18': resneta18,
    'resneta50b': resneta50b,
    'resneta101b': resneta101b,
    'resneta152b': resneta152b,

    'resnetd50b': resnetd50b,
    'resnetd101b': resnetd101b,
    'resnetd152b': resnetd152b,

    'fastseresnet101b': fastseresnet101b,

    'resnet10_cub': resnet10_cub,
    'resnet12_cub': resnet12_cub,
    'resnet14_cub': resnet14_cub,
    'resnetbc14b_cub': resnetbc14b_cub,
    'resnet16_cub': resnet16_cub,
    'resnet18_cub': resnet18_cub,
    'resnet26_cub': resnet26_cub,
    'resnetbc26b_cub': resnetbc26b_cub,
    'resnet34_cub': resnet34_cub,
    'resnetbc38b_cub': resnetbc38b_cub,
    'resnet50_cub': resnet50_cub,
    'resnet50b_cub': resnet50b_cub,
    'resnet101_cub': resnet101_cub,
    'resnet101b_cub': resnet101b_cub,
    'resnet152_cub': resnet152_cub,
    'resnet152b_cub': resnet152b_cub,
    'resnet200_cub': resnet200_cub,
    'resnet200b_cub': resnet200b_cub,

    'seresnet10_cub': seresnet10_cub,
    'seresnet12_cub': seresnet12_cub,
    'seresnet14_cub': seresnet14_cub,
    'seresnetbc14b_cub': seresnetbc14b_cub,
    'seresnet16_cub': seresnet16_cub,
    'seresnet18_cub': seresnet18_cub,
    'seresnet26_cub': seresnet26_cub,
    'seresnetbc26b_cub': seresnetbc26b_cub,
    'seresnet34_cub': seresnet34_cub,
    'seresnetbc38b_cub': seresnetbc38b_cub,
    'seresnet50_cub': seresnet50_cub,
    'seresnet50b_cub': seresnet50b_cub,
    'seresnet101_cub': seresnet101_cub,
    'seresnet101b_cub': seresnet101b_cub,
    'seresnet152_cub': seresnet152_cub,
    'seresnet152b_cub': seresnet152b_cub,
    'seresnet200_cub': seresnet200_cub,
    'seresnet200b_cub': seresnet200b_cub,

    'mobilenet_w1_cub': mobilenet_w1_cub,
    'mobilenet_w3d4_cub': mobilenet_w3d4_cub,
    'mobilenet_wd2_cub': mobilenet_wd2_cub,
    'mobilenet_wd4_cub': mobilenet_wd4_cub,

    'fdmobilenet_w1_cub': fdmobilenet_w1_cub,
    'fdmobilenet_w3d4_cub': fdmobilenet_w3d4_cub,
    'fdmobilenet_wd2_cub': fdmobilenet_wd2_cub,
    'fdmobilenet_wd4_cub': fdmobilenet_wd4_cub,

    'proxylessnas_cpu_cub': proxylessnas_cpu_cub,
    'proxylessnas_gpu_cub': proxylessnas_gpu_cub,
    'proxylessnas_mobile_cub': proxylessnas_mobile_cub,
    'proxylessnas_mobile14_cub': proxylessnas_mobile14_cub,

    'fcn8sd_resnetd50b_voc': fcn8sd_resnetd50b_voc,
    'fcn8sd_resnetd101b_voc': fcn8sd_resnetd101b_voc,
    'fcn8sd_resnetd50b_coco': fcn8sd_resnetd50b_coco,
    'fcn8sd_resnetd101b_coco': fcn8sd_resnetd101b_coco,
    'fcn8sd_resnetd50b_ade20k': fcn8sd_resnetd50b_ade20k,
    'fcn8sd_resnetd101b_ade20k': fcn8sd_resnetd101b_ade20k,
    'fcn8sd_resnetd50b_cityscapes': fcn8sd_resnetd50b_cityscapes,
    'fcn8sd_resnetd101b_cityscapes': fcn8sd_resnetd101b_cityscapes,

    'pspnet_resnetd50b_voc': pspnet_resnetd50b_voc,
    'pspnet_resnetd101b_voc': pspnet_resnetd101b_voc,
    'pspnet_resnetd50b_coco': pspnet_resnetd50b_coco,
    'pspnet_resnetd101b_coco': pspnet_resnetd101b_coco,
    'pspnet_resnetd50b_ade20k': pspnet_resnetd50b_ade20k,
    'pspnet_resnetd101b_ade20k': pspnet_resnetd101b_ade20k,
    'pspnet_resnetd50b_cityscapes': pspnet_resnetd50b_cityscapes,
    'pspnet_resnetd101b_cityscapes': pspnet_resnetd101b_cityscapes,

    'deeplabv3_resnetd50b_voc': deeplabv3_resnetd50b_voc,
    'deeplabv3_resnetd101b_voc': deeplabv3_resnetd101b_voc,
    'deeplabv3_resnetd152b_voc': deeplabv3_resnetd152b_voc,
    'deeplabv3_resnetd50b_coco': deeplabv3_resnetd50b_coco,
    'deeplabv3_resnetd101b_coco': deeplabv3_resnetd101b_coco,
    'deeplabv3_resnetd152b_coco': deeplabv3_resnetd152b_coco,
    'deeplabv3_resnetd50b_ade20k': deeplabv3_resnetd50b_ade20k,
    'deeplabv3_resnetd101b_ade20k': deeplabv3_resnetd101b_ade20k,
    'deeplabv3_resnetd50b_cityscapes': deeplabv3_resnetd50b_cityscapes,
    'deeplabv3_resnetd101b_cityscapes': deeplabv3_resnetd101b_cityscapes,

    'icnet_resnetd50b_cityscapes': icnet_resnetd50b_cityscapes,

    'fastscnn_cityscapes': fastscnn_cityscapes,

    'sinet_cityscapes': sinet_cityscapes,

    'bisenet_resnet18_celebamaskhq': bisenet_resnet18_celebamaskhq,

    'danet_resnetd50b_cityscapes': danet_resnetd50b_cityscapes,
    'danet_resnetd101b_cityscapes': danet_resnetd101b_cityscapes,

    'alphapose_fastseresnet101b_coco': alphapose_fastseresnet101b_coco,

    'simplepose_resnet18_coco': simplepose_resnet18_coco,
    'simplepose_resnet50b_coco': simplepose_resnet50b_coco,
    'simplepose_resnet101b_coco': simplepose_resnet101b_coco,
    'simplepose_resnet152b_coco': simplepose_resnet152b_coco,
    'simplepose_resneta50b_coco': simplepose_resneta50b_coco,
    'simplepose_resneta101b_coco': simplepose_resneta101b_coco,
    'simplepose_resneta152b_coco': simplepose_resneta152b_coco,

    'simplepose_mobile_resnet18_coco': simplepose_mobile_resnet18_coco,
    'simplepose_mobile_resnet50b_coco': simplepose_mobile_resnet50b_coco,
    'simplepose_mobile_mobilenet_w1_coco': simplepose_mobile_mobilenet_w1_coco,
    'simplepose_mobile_mobilenetv2b_w1_coco': simplepose_mobile_mobilenetv2b_w1_coco,
    'simplepose_mobile_mobilenetv3_small_w1_coco': simplepose_mobile_mobilenetv3_small_w1_coco,
    'simplepose_mobile_mobilenetv3_large_w1_coco': simplepose_mobile_mobilenetv3_large_w1_coco,

    'lwopenpose2d_mobilenet_cmupan_coco': lwopenpose2d_mobilenet_cmupan_coco,
    'lwopenpose3d_mobilenet_cmupan_coco': lwopenpose3d_mobilenet_cmupan_coco,

    'ibppose_coco': ibppose_coco,

    'grmiposelite_mobilenet_w1_coco': grmiposelite_mobilenet_w1_coco,

    'centernet_resnet18_voc': centernet_resnet18_voc,
    'centernet_resnet18_coco': centernet_resnet18_coco,
    'centernet_resnet50b_voc': centernet_resnet50b_voc,
    'centernet_resnet50b_coco': centernet_resnet50b_coco,
    'centernet_resnet101b_voc': centernet_resnet101b_voc,
    'centernet_resnet101b_coco': centernet_resnet101b_coco,

    'lffd20x5s320v2_widerface': lffd20x5s320v2_widerface,
    'lffd25x8s560v1_widerface': lffd25x8s560v1_widerface,

    'voca8flame': voca8flame,
    'nvpattexp116bazel76': nvpattexp116bazel76,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns:
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
