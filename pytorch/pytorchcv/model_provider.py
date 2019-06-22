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
from .models.ibnresnet import *
from .models.ibnbresnet import *
from .models.ibnresnext import *
from .models.ibndensenet import *
from .models.airnet import *
from .models.airnext import *
from .models.bamresnet import *
from .models.cbamresnet import *
from .models.resattnet import *
from .models.sknet import *
from .models.diaresnet import *
from .models.diapreresnet import *
from .models.pyramidnet import *
from .models.diracnetv2 import *
from .models.sharesnet import *
from .models.densenet import *
from .models.condensenet import *
from .models.sparsenet import *
from .models.peleenet import *
from .models.wrn import *
from .models.drn import *
from .models.dpn import *
from .models.darknet import *
from .models.darknet53 import *
from .models.channelnet import *
from .models.isqrtcovresnet import *
from .models.revnet import *
from .models.irevnet import *
from .models.bagnet import *
from .models.dla import *
from .models.msdnet import *
from .models.fishnet import *
from .models.espnetv2 import *
from .models.xdensenet import *
from .models.squeezenet import *
from .models.squeezenext import *
from .models.shufflenet import *
from .models.shufflenetv2 import *
from .models.shufflenetv2b import *
from .models.menet import *
from .models.mobilenet import *
from .models.mobilenetv2 import *
from .models.mobilenetv3 import *
from .models.igcv3 import *
from .models.mnasnet import *
from .models.darts import *
from .models.proxylessnas import *
from .models.xception import *
from .models.inceptionv3 import *
from .models.inceptionv4 import *
from .models.inceptionresnetv2 import *
from .models.polynet import *
from .models.nasnet import *
from .models.pnasnet import *
from .models.efficientnet import *

from .models.nin_cifar import *
from .models.resnet_cifar import *
from .models.preresnet_cifar import *
from .models.resnext_cifar import *
from .models.seresnet_cifar import *
from .models.pyramidnet_cifar import *
from .models.densenet_cifar import *
from .models.xdensenet_cifar import *
from .models.wrn_cifar import *
from .models.wrn1bit_cifar import *
from .models.ror_cifar import *
from .models.rir_cifar import *
from .models.msdnet_cifar10 import *
from .models.resdropresnet_cifar import *
from .models.shakeshakeresnet_cifar import *
from .models.shakedropresnet_cifar import *
from .models.fractalnet_cifar import *
from .models.diaresnet_cifar import *
from .models.diapreresnet_cifar import *

from .models.octresnet import *

from .models.resnetd import *

from .models.resnet_cub import *
from .models.seresnet_cub import *
from .models.mobilenet_cub import *
from .models.proxylessnas_cub import *
from .models.ntsnet_cub import *

from .models.fcn8sd import *
from .models.pspnet import *
from .models.deeplabv3 import *

from .models.superpointnet import *
from .models.others.oth_superpointnet import *

__all__ = ['get_model']


_models = {
    'alexnet': alexnet,

    'zfnet': zfnet,

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

    'resnext14_32x4d': resnext14_32x4d,
    'resnext26_32x4d': resnext26_32x4d,
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

    'sepreresnet18': sepreresnet18,
    'sepreresnet34': sepreresnet34,
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

    'resattnet56': resattnet56,
    'resattnet92': resattnet92,
    'resattnet128': resattnet128,
    'resattnet164': resattnet164,
    'resattnet200': resattnet200,
    'resattnet236': resattnet236,
    'resattnet452': resattnet452,

    'sknet50': sknet50,
    'sknet101': sknet101,
    'sknet152': sknet152,

    'diaresnet10': diaresnet10,
    'diaresnet12': diaresnet12,
    'diaresnet14': diaresnet14,
    'diaresnetbc14b': diaresnetbc14b,
    'diaresnet16': diaresnet16,
    'diaresnet18': diaresnet18,
    'diaresnet26': diaresnet26,
    'diaresnetbc26b': diaresnetbc26b,
    'diaresnet34': diaresnet34,
    'diaresnetbc38b': diaresnetbc38b,
    'diaresnet50': diaresnet50,
    'diaresnet50b': diaresnet50b,
    'diaresnet101': diaresnet101,
    'diaresnet101b': diaresnet101b,
    'diaresnet152': diaresnet152,
    'diaresnet152b': diaresnet152b,
    'diaresnet200': diaresnet200,
    'diaresnet200b': diaresnet200b,

    'diapreresnet10': diapreresnet10,
    'diapreresnet12': diapreresnet12,
    'diapreresnet14': diapreresnet14,
    'diapreresnetbc14b': diapreresnetbc14b,
    'diapreresnet16': diapreresnet16,
    'diapreresnet18': diapreresnet18,
    'diapreresnet26': diapreresnet26,
    'diapreresnetbc26b': diapreresnetbc26b,
    'diapreresnet34': diapreresnet34,
    'diapreresnetbc38b': diapreresnetbc38b,
    'diapreresnet50': diapreresnet50,
    'diapreresnet50b': diapreresnet50b,
    'diapreresnet101': diapreresnet101,
    'diapreresnet101b': diapreresnet101b,
    'diapreresnet152': diapreresnet152,
    'diapreresnet152b': diapreresnet152b,
    'diapreresnet200': diapreresnet200,
    'diapreresnet200b': diapreresnet200b,
    'diapreresnet269b': diapreresnet269b,

    'pyramidnet101_a360': pyramidnet101_a360,

    'diracnet18v2': diracnet18v2,
    'diracnet34v2': diracnet34v2,

    'sharesnet18': sharesnet18,
    'sharesnet34': sharesnet34,
    'sharesnet50': sharesnet50,
    'sharesnet50b': sharesnet50b,
    'sharesnet101': sharesnet101,
    'sharesnet101b': sharesnet101b,
    'sharesnet152': sharesnet152,
    'sharesnet152b': sharesnet152b,

    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,

    'condensenet74_c4_g4': condensenet74_c4_g4,
    'condensenet74_c8_g8': condensenet74_c8_g8,

    'sparsenet121': sparsenet121,
    'sparsenet161': sparsenet161,
    'sparsenet169': sparsenet169,
    'sparsenet201': sparsenet201,
    'sparsenet264': sparsenet264,

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

    'channelnet': channelnet,

    'revnet38': revnet38,
    'revnet110': revnet110,
    'revnet164': revnet164,

    'irevnet301': irevnet301,

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

    'msdnet22': msdnet22,

    'fishnet99': fishnet99,
    'fishnet150': fishnet150,

    'espnetv2_wd2': espnetv2_wd2,
    'espnetv2_w1': espnetv2_w1,
    'espnetv2_w5d4': espnetv2_w5d4,
    'espnetv2_w3d2': espnetv2_w3d2,
    'espnetv2_w2': espnetv2_w2,

    'xdensenet121_2': xdensenet121_2,
    'xdensenet161_2': xdensenet161_2,
    'xdensenet169_2': xdensenet169_2,
    'xdensenet201_2': xdensenet201_2,

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

    'fdmobilenet_w1': fdmobilenet_w1,
    'fdmobilenet_w3d4': fdmobilenet_w3d4,
    'fdmobilenet_wd2': fdmobilenet_wd2,
    'fdmobilenet_wd4': fdmobilenet_wd4,

    'mobilenetv2_w1': mobilenetv2_w1,
    'mobilenetv2_w3d4': mobilenetv2_w3d4,
    'mobilenetv2_wd2': mobilenetv2_wd2,
    'mobilenetv2_wd4': mobilenetv2_wd4,

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

    'mnasnet': mnasnet,

    'darts': darts,

    'proxylessnas_cpu': proxylessnas_cpu,
    'proxylessnas_gpu': proxylessnas_gpu,
    'proxylessnas_mobile': proxylessnas_mobile,
    'proxylessnas_mobile14': proxylessnas_mobile14,

    'xception': xception,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inceptionresnetv2': inceptionresnetv2,
    'polynet': polynet,

    'nasnet_4a1056': nasnet_4a1056,
    'nasnet_6a4032': nasnet_6a4032,

    'pnasnet5large': pnasnet5large,

    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'efficientnet_b0b': efficientnet_b0b,
    'efficientnet_b1b': efficientnet_b1b,
    'efficientnet_b2b': efficientnet_b2b,
    'efficientnet_b3b': efficientnet_b3b,

    'nin_cifar10': nin_cifar10,
    'nin_cifar100': nin_cifar100,
    'nin_svhn': nin_svhn,

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
    'preresnet1001_cifar10': preresnet1001_cifar10,
    'preresnet1001_cifar100': preresnet1001_cifar100,
    'preresnet1001_svhn': preresnet1001_svhn,
    'preresnet1202_cifar10': preresnet1202_cifar10,
    'preresnet1202_cifar100': preresnet1202_cifar100,
    'preresnet1202_svhn': preresnet1202_svhn,

    'resnext29_32x4d_cifar10': resnext29_32x4d_cifar10,
    'resnext29_32x4d_cifar100': resnext29_32x4d_cifar100,
    'resnext29_32x4d_svhn': resnext29_32x4d_svhn,
    'resnext29_16x64d_cifar10': resnext29_16x64d_cifar10,
    'resnext29_16x64d_cifar100': resnext29_16x64d_cifar100,
    'resnext29_16x64d_svhn': resnext29_16x64d_svhn,

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
    'seresnet1001_cifar10': seresnet1001_cifar10,
    'seresnet1001_cifar100': seresnet1001_cifar100,
    'seresnet1001_svhn': seresnet1001_svhn,
    'seresnet1202_cifar10': seresnet1202_cifar10,
    'seresnet1202_cifar100': seresnet1202_cifar100,
    'seresnet1202_svhn': seresnet1202_svhn,

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

    'xdensenet40_2_k24_bc_cifar10': xdensenet40_2_k24_bc_cifar10,
    'xdensenet40_2_k24_bc_cifar100': xdensenet40_2_k24_bc_cifar100,
    'xdensenet40_2_k24_bc_svhn': xdensenet40_2_k24_bc_svhn,
    'xdensenet40_2_k36_bc_cifar10': xdensenet40_2_k36_bc_cifar10,
    'xdensenet40_2_k36_bc_cifar100': xdensenet40_2_k36_bc_cifar100,
    'xdensenet40_2_k36_bc_svhn': xdensenet40_2_k36_bc_svhn,

    'wrn16_10_cifar10': wrn16_10_cifar10,
    'wrn16_10_cifar100': wrn16_10_cifar100,
    'wrn16_10_svhn': wrn16_10_svhn,
    'wrn28_10_cifar10': wrn28_10_cifar10,
    'wrn28_10_cifar100': wrn28_10_cifar100,
    'wrn28_10_svhn': wrn28_10_svhn,
    'wrn40_8_cifar10': wrn40_8_cifar10,
    'wrn40_8_cifar100': wrn40_8_cifar100,
    'wrn40_8_svhn': wrn40_8_svhn,

    'wrn20_10_1bit_cifar10': wrn20_10_1bit_cifar10,
    'wrn20_10_1bit_cifar100': wrn20_10_1bit_cifar100,
    'wrn20_10_1bit_svhn': wrn20_10_1bit_svhn,
    'wrn20_10_32bit_cifar10': wrn20_10_32bit_cifar10,
    'wrn20_10_32bit_cifar100': wrn20_10_32bit_cifar100,
    'wrn20_10_32bit_svhn': wrn20_10_32bit_svhn,

    'ror3_56_cifar10': ror3_56_cifar10,
    'ror3_56_cifar100': ror3_56_cifar100,
    'ror3_56_svhn': ror3_56_svhn,
    'ror3_110_cifar10': ror3_110_cifar10,
    'ror3_110_cifar100': ror3_110_cifar100,
    'ror3_110_svhn': ror3_110_svhn,
    'ror3_164_cifar10': ror3_164_cifar10,
    'ror3_164_cifar100': ror3_164_cifar100,
    'ror3_164_svhn': ror3_164_svhn,

    'rir_cifar10': rir_cifar10,
    'rir_cifar100': rir_cifar100,
    'rir_svhn': rir_svhn,

    'msdnet22_cifar10': msdnet22_cifar10,

    'resdropresnet20_cifar10': resdropresnet20_cifar10,
    'resdropresnet20_cifar100': resdropresnet20_cifar100,
    'resdropresnet20_svhn': resdropresnet20_svhn,

    'shakeshakeresnet20_2x16d_cifar10': shakeshakeresnet20_2x16d_cifar10,
    'shakeshakeresnet20_2x16d_cifar100': shakeshakeresnet20_2x16d_cifar100,
    'shakeshakeresnet20_2x16d_svhn': shakeshakeresnet20_2x16d_svhn,
    'shakeshakeresnet26_2x32d_cifar10': shakeshakeresnet26_2x32d_cifar10,
    'shakeshakeresnet26_2x32d_cifar100': shakeshakeresnet26_2x32d_cifar100,
    'shakeshakeresnet26_2x32d_svhn': shakeshakeresnet26_2x32d_svhn,

    'shakedropresnet20_cifar10': shakedropresnet20_cifar10,
    'shakedropresnet20_cifar100': shakedropresnet20_cifar100,
    'shakedropresnet20_svhn': shakedropresnet20_svhn,

    'fractalnet_cifar10': fractalnet_cifar10,
    'fractalnet_cifar100': fractalnet_cifar100,

    'diaresnet20_cifar10': diaresnet20_cifar10,
    'diaresnet20_cifar100': diaresnet20_cifar100,
    'diaresnet20_svhn': diaresnet20_svhn,
    'diaresnet56_cifar10': diaresnet56_cifar10,
    'diaresnet56_cifar100': diaresnet56_cifar100,
    'diaresnet56_svhn': diaresnet56_svhn,
    'diaresnet110_cifar10': diaresnet110_cifar10,
    'diaresnet110_cifar100': diaresnet110_cifar100,
    'diaresnet110_svhn': diaresnet110_svhn,
    'diaresnet164bn_cifar10': diaresnet164bn_cifar10,
    'diaresnet164bn_cifar100': diaresnet164bn_cifar100,
    'diaresnet164bn_svhn': diaresnet164bn_svhn,
    'diaresnet1001_cifar10': diaresnet1001_cifar10,
    'diaresnet1001_cifar100': diaresnet1001_cifar100,
    'diaresnet1001_svhn': diaresnet1001_svhn,
    'diaresnet1202_cifar10': diaresnet1202_cifar10,
    'diaresnet1202_cifar100': diaresnet1202_cifar100,
    'diaresnet1202_svhn': diaresnet1202_svhn,

    'diapreresnet20_cifar10': diapreresnet20_cifar10,
    'diapreresnet20_cifar100': diapreresnet20_cifar100,
    'diapreresnet20_svhn': diapreresnet20_svhn,
    'diapreresnet56_cifar10': diapreresnet56_cifar10,
    'diapreresnet56_cifar100': diapreresnet56_cifar100,
    'diapreresnet56_svhn': diapreresnet56_svhn,
    'diapreresnet110_cifar10': diapreresnet110_cifar10,
    'diapreresnet110_cifar100': diapreresnet110_cifar100,
    'diapreresnet110_svhn': diapreresnet110_svhn,
    'diapreresnet164bn_cifar10': diapreresnet164bn_cifar10,
    'diapreresnet164bn_cifar100': diapreresnet164bn_cifar100,
    'diapreresnet164bn_svhn': diapreresnet164bn_svhn,
    'diapreresnet1001_cifar10': diapreresnet1001_cifar10,
    'diapreresnet1001_cifar100': diapreresnet1001_cifar100,
    'diapreresnet1001_svhn': diapreresnet1001_svhn,
    'diapreresnet1202_cifar10': diapreresnet1202_cifar10,
    'diapreresnet1202_cifar100': diapreresnet1202_cifar100,
    'diapreresnet1202_svhn': diapreresnet1202_svhn,

    'isqrtcovresnet18': isqrtcovresnet18,
    'isqrtcovresnet34': isqrtcovresnet34,
    'isqrtcovresnet50': isqrtcovresnet50,
    'isqrtcovresnet50b': isqrtcovresnet50b,
    'isqrtcovresnet101': isqrtcovresnet101,
    'isqrtcovresnet101b': isqrtcovresnet101b,

    'resnetd50b': resnetd50b,
    'resnetd101b': resnetd101b,
    'resnetd152b': resnetd152b,

    'octresnet10_ad2': octresnet10_ad2,
    'octresnet50b_ad2': octresnet50b_ad2,

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

    'ntsnet_cub': ntsnet_cub,

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

    'superpointnet': superpointnet,
    'oth_superpointnet': oth_superpointnet,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
