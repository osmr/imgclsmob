"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_model', 'download_model', 'calc_num_params']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag, caption, paper, ds, img_size, scale, batch, rem) for
               name, error, checksum, repo_release_tag, caption, paper, ds, img_size, scale, batch, rem in [
    ('alexnet', '1664', '2768cdb312d584e33e93f31b0c569589bb289749', 'v0.0.481', 'AlexNet', '1404.5997', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('alexnetb', '1747', 'ac887bf7eada4179857d243584ac30b4d74a6493', 'v0.0.485', 'AlexNet-b', '1404.5997', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('zfnet', '1727', 'd010ddca1eb32a50a8cceb475c792f53e769b631', 'v0.0.395', 'ZFNet', '1311.2901', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('zfnetb', '1490', 'f6bec24eba037c8e4956704ed5bafaed29966601', 'v0.0.400', 'ZFNet-b', '1311.2901', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('vgg11', '1036', '71e85f6ef76f56e3e89d597d2fc461496ed281e9', 'v0.0.381', 'VGG-11', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('vgg13', '0975', '2b2c8770a7610d9dcd444ec8ae992681e270eb42', 'v0.0.388', 'VGG-13', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('vgg16', '0865', '5ca155da3dc6687e070ff34815cb5aabd0bed4b9', 'v0.0.401', 'VGG-16', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('vgg19', '0790', '9bd923a82ece9f038e944d7666f1c11b478dc7e6', 'v0.0.420', 'VGG-19', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg11', '0961', '10f01fba064ec168df074b98d59ae7b82b1207d4', 'v0.0.339', 'BN-VGG-11', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg13', '0913', 'b1acd7158e6e9973ce9e274c65ceb64a244f9967', 'v0.0.353', 'BN-VGG-13', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg16', '0779', '0f570b928b180f909fa39df3924f89c746816722', 'v0.0.359', 'BN-VGG-16', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg19', '0712', '3f286cbd2a57abb4c516425c5e095c2cfc8d54e3', 'v0.0.360', 'BN-VGG-19', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg11b', '0996', 'ef747edc87705e1ed500a31c80199273b2fbd5fa', 'v0.0.407', 'BN-VGG-11b', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg13b', '0924', '5f313c535fc47c3ad6bd2f741f453dbcf8191be6', 'v0.0.488', 'BN-VGG-13b', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg16b', '0795', 'bfff365ac38a763aaed4b4d9bdc7b2cdbe6c8e9f', 'v0.0.489', 'BN-VGG-16b', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bn_vgg19b', '0746', 'f523b4e4b070a170f63e9bb6965fca3764751aa9', 'v0.0.490', 'BN-VGG-19b', '1409.1556', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('bninception', '0774', 'd79ba5f573ba2da5fea5e4c9a7f67ddd526e234b', 'v0.0.405', 'BN-Inception', '1502.03167', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet10', '1293', 'cedc302c71cfa87c1fb2c52a9c156522187fd929', 'v0.0.483', 'ResNet-10', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet12', '1223', '84a43cf672c708a016dd1142ca1a23c278931532', 'v0.0.485', 'ResNet-12', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet14', '1109', 'b3132cbfb7d64ae83b1cd2e3954f4c5b1180fd7b', 'v0.0.491', 'ResNet-14', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnetbc14b', '1074', '14b1fd95d8b7964c0e7c6eba22f6f58db03d3df0', 'v0.0.481', 'ResNet-BC-14b', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet16', '1009', '4352d6a91d6e28aa839f741006a5a41cfa62bfd6', 'v0.0.493', 'ResNet-16', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet18_wd4', '1785', 'fe79b31f56e7becab9c014dbc14ccdb564b5148f', 'v0.0.262', 'ResNet-18 x0.25', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet18_wd2', '1327', '6654f50ad357f4596502b92b3dca2147776089ac', 'v0.0.263', 'ResNet-18 x0.5', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet18_w3d4', '1106', '3636648b504e1ba134947743eb34dd0e78feda02', 'v0.0.266', 'ResNet-18 x0.75', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet18', '0896', '77a56f155214819bfc79ff09795370f955b20e6d', 'v0.0.478', 'ResNet-18', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet26', '0849', '4bfbc640f218e0eaf4c380cfdb98d55f259862d6', 'v0.0.489', 'ResNet-26', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnetbc26b', '0797', '7af52a73b234dc56ab4b0757cf3ea772d0699622', 'v0.0.313', 'ResNet-BC-26b', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet34', '0780', '3f775482a327e5fc4850fbb77785bfc55e171e5f', 'v0.0.291', 'ResNet-34', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnetbc38b', '0700', '3fbac61d86810d489988a92f425f1a6bfe46f155', 'v0.0.328', 'ResNet-BC-38b', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet50', '0633', 'b00d1c8e52aa7a2badc705b1545aaf6ccece6ce9', 'v0.0.329', 'ResNet-50', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet50b', '0638', '8a5473ef985d65076a3758117ad5700d726bd952', 'v0.0.308', 'ResNet-50b', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet101', '0622', 'ab0cf005bbe9b17e53f9e3c330c6147a8c80b3a5', 'v0.0.1', 'ResNet-101', '1512.03385', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('resnet101b', '0530', 'f059ba3c7fa4a65f2da6e17f3718662d59836637', 'v0.0.357', 'ResNet-101b', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnet152', '0550', '800b2cb1959a0d3648483e86917502b8f63dc37e', 'v0.0.144', 'ResNet-152', '1512.03385', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('resnet152b', '0499', '667ea926f3753e0c8336fa78969171d64f819cc4', 'v0.0.378', 'ResNet-152b', '1512.03385', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet10', '1421', 'b3973cd4461287d61df081d6f689d293eacf2248', 'v0.0.249', 'PrepResNet-10', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet12', '1348', '563066fa8fcf8b5f19906b933fea784965d68192', 'v0.0.257', 'PreResNet-12', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet14', '1239', '4be725fd3f06c99c46817fce3b69caf2ebc62414', 'v0.0.260', 'PreResNet-14', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnetbc14b', '1181', 'a68d31c372e647474ae954e51e5bc2ba9fb3f166', 'v0.0.315', 'PreResNet=BC-14b', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet16', '1108', '06d8c87e29284dac19a9019485e210541532411a', 'v0.0.261', 'PreResNet-16', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet18_wd4', '1811', '41135c15210390e9a564b14e8ae2ebda1a662ec1', 'v0.0.272', 'PreResNet-18 x0.25', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet18_wd2', '1340', 'c1fe4e314188eeb93302432d03731a91ce8bc9f2', 'v0.0.273', 'PreResNet-18 x0.5', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet18_w3d4', '1105', 'ed2f9ca434b6910b92657eefc73ad186396578d5', 'v0.0.274', 'PreResNet-18 x0.75', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet18', '0972', '5651bc2dbb200382822a6b64375d240f747cc726', 'v0.0.140', 'PreResNet-18', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet26', '0851', '99e7d6cc5944cd7cf6d4746e6fdf18b477d3d9a0', 'v0.0.316', 'PreResNet-26', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnetbc26b', '0803', 'd7283bdd70e1b75520fe2cdcc273d51715e077b4', 'v0.0.325', 'PreResNet-BC-26b', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet34', '0774', 'fd5bd1e883048e29099768465df2dd9e891803f4', 'v0.0.300', 'PreResNet-34', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnetbc38b', '0657', '9e523bb92dc592ee576a6bb73a328dc024bdc967', 'v0.0.348', 'PreResNet-BC-38b', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet50', '0647', '222ca73b021f893b925c15e24ea2a6bc0fdf2546', 'v0.0.330', 'PreResNet-50', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet50b', '0655', '8b60378ee3aed878d27a2b4a9ddc596a812c7649', 'v0.0.307', 'PreResNet-50b', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet101', '0591', '4bacff796e113562e1dfdf71cfa7c6ed33e0ba86', 'v0.0.2', 'PreResNet-101', '1603.05027', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('preresnet101b', '0556', '76bfe6d020b55f163e77de6b1c27be6b0bed8b7b', 'v0.0.351', 'PreResNet-101b', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet152', '0555', 'c842a030abbcc21a0f2a9a8299fc42204897a611', 'v0.0.14', 'PreResNet-152', '1603.05027', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('preresnet152b', '0516', 'f3805f4b8c845798b711171ad6632bcf56259844', 'v0.0.386', 'PreResNet-152b', '1603.05027', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('preresnet200b', '0588', 'f7104ff306ed5de2c27f3c855051c22bda167981', 'v0.0.45', 'PreResNet-200b', '1603.05027', 'in1k', 224, 0.875, 200, '[tornadomeet/ResNet]'),  # noqa
    ('preresnet269b', '0581', '1a7878bb10923b22bda58d7935dfa6e5e8a7b67d', 'v0.0.239', 'PreResNet-269b', '1603.05027', 'in1k', 224, 0.875, 200, '[soeaver/mxnet-model]'),  # noqa
    ('resnext14_16x4d', '1248', '35ffac2a26374e71b6bf4bc9f90b7a1a1dd47e7d', 'v0.0.370', 'ResNeXt-14 (16x4d)', '1611.05431', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnext14_32x2d', '1281', '14521186b8c78c7c07f3904360839f22c180f65e', 'v0.0.371', 'ResNeXt-14 (32x2d)', '1611.05431', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnext14_32x4d', '1146', '89aa679393d8356ce5589749b4371714bf4ceac0', 'v0.0.327', 'ResNeXt-14 (32x4d)', '1611.05431', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnext26_32x2d', '0887', 'c3bd130747909a8c89546f3b3f5ce08bb4f55731', 'v0.0.373', 'ResNeXt-26 (32x2d)', '1611.05431', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnext26_32x4d', '0746', '1011ac35e30d753b79f0600a5376c87a37b67a61', 'v0.0.332', 'ResNeXt-26 (32x4d)', '1611.05431', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnext50_32x4d', '0558', 'b629a5227df20b2d522e4f72c40f1cf87ee9b055', 'v0.0.417', 'ResNeXt-50 (32x4d)', '1611.05431', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('resnext101_32x4d', '0523', '279a3189c6fed46dd12c2ee210bc6a493f629c76', 'v0.0.417', 'ResNeXt-101 (32x4d)', '1611.05431', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('resnext101_64x4d', '0509', '2af0b82274bbcf5ee75575c58c3dd6a4a292b0ae', 'v0.0.417', 'ResNeXt-101 (64x4d)', '1611.05431', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('seresnet10', '1202', '8dace12e6aaac68d3c272f52b2513a5b40a4f959', 'v0.0.486', 'SE-ResNet-10', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnet18', '0961', '022123a5e88c9917e63165f5b5a7808a606d452a', 'v0.0.355', 'SE-ResNet-18', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnet26', '0824', '64fc8759c5bb9b9b40b2e33a46420ee22ae268c9', 'v0.0.363', 'SE-ResNet-26', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnetbc26b', '0703', 'b98d9d6afca4d79d0347001542162b9fe4071d39', 'v0.0.366', 'SE-ResNet-BC-26b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnetbc38b', '0595', '03671c05f5f684b44085383b7b89a8b44a7524fe', 'v0.0.374', 'SE-ResNet-BC-38b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnet50', '0575', '004bfde422c860c4f11b1e1190bb5a8db477d939', 'v0.0.441', 'SE-ResNet-50', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnet50b', '0539', '459e6871e944d1c7102ee9c055ea428b8d9a168c', 'v0.0.387', 'SE-ResNet-50b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnet101', '0589', '5e6e831b7518b9b8a049dd60ed1ff82ae75ff55e', 'v0.0.11', 'SE-ResNet-101', '1709.01507', 'in1k', 224, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('seresnet101b', '0487', 'b83a20fd2ad9a32e0fe5cb3daef45aac03ea3194', 'v0.0.460', 'SE-ResNet-101b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnet152', '0576', '814cf72e0deeab530332b16fb9b609e574afec61', 'v0.0.11', 'SE-ResNet-152', '1709.01507', 'in1k', 224, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('sepreresnet10', '1338', '935ed56009a64c893153cdba8e4a4f87f7184e71', 'v0.0.377', 'SE-PreResNet-10', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sepreresnet18', '0963', 'c065cd9e1c026d0529526cfc945c137bade6f0c7', 'v0.0.380', 'SE-PreResNet-18', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sepreresnetbc26b', '0660', 'f750b2f588a27620b30c86f0060a41422d4a0f75', 'v0.0.399', 'SE-PreResNet-BC-26b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sepreresnetbc38b', '0578', '12827fcd3c8c1a8c8ba1d109e85ffa67e7ab306a', 'v0.0.409', 'SE-PreResNet-BC-38b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sepreresnet50b', '0549', '4628a07d7dd92c775868dffd33fd6e3e7522c261', 'v0.0.461', 'SE-PreResNet-50b', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('seresnext50_32x4d', '0521', 'b0ce2520bd87a50b63b1365c74356dba333de68c', 'v0.0.418', 'SE-ResNeXt-50 (32x4d)', '1709.01507', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('seresnext101_32x4d', '0480', '4f6479f0801a92d35a256a47e5c11a97b3555016', 'v0.0.418', 'SE-ResNeXt-101 (32x4d)', '1709.01507', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('seresnext101_64x4d', '0476', 'da806109a2346be16f2b3b9aa60aa8f52bc6a1fa', 'v0.0.418', 'SE-ResNeXt-101 (64x4d)', '1709.01507', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('senet16', '0820', '373aeafdc994c3e03bf483a9fa3ecb152353722a', 'v0.0.341', 'SENet-16', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('senet28', '0598', '27165b63696061e57c141314d44732aa65f807a8', 'v0.0.356', 'SENet-28', '1709.01507', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('senet154', '0461', '6512228c820897cd09f877527a553ca99d673956', 'v0.0.13', 'SENet-154', '1709.01507', 'in1k', 224, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('resnestabc14', '0647', '0c3d9e34aebf0dee0dbcbb937eb54f2a7fc8f64a', 'v0.0.493', 'ResNeSt(A)-BC-14', '2004.08955', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnesta18', '0707', 'efca5a69587dcdff3aa5d3d7cbd621d082299e27', 'v0.0.489', 'ResNeSt(A)-18', '2004.08955', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnestabc26', '0571', 'd6a8a7ae2f6b1224ff51a6c1ee4b94c4795218db', 'v0.0.465', 'ResNeSt(A)-BC-26', '2004.08955', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnesta50', '0462', 'c98fe61543ea770d120d157eed2030c60a6bc70d', 'v0.0.465', 'ResNeSt(A)-50', '2004.08955', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnesta101', '0403', '61e147732069b54ed4da4b342b1b8526a0e9df54', 'v0.0.465', 'ResNeSt(A)-101', '2004.08955', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('resnesta200', '0339', '6dc300871b186950ee64fd28bb168f7fb4a036e3', 'v0.0.465', 'ResNeSt(A)-200', '2004.08955', 'in1k', 256, 0.875, 150, ''),  # noqa
    ('resnesta269', '0338', '6a555ce85eb177299eb43747cf019a50d3a143c1', 'v0.0.465', 'ResNeSt(A)-269', '2004.08955', 'in1k', 320, 0.875, 100, ''),  # noqa
    ('ibn_resnet50', '0641', 'e48a1fe5f7e448d4b784ef4dc0f33832f3370a9b', 'v0.0.127', 'IBN-ResNet-50', '1807.09441', 'in1k', 224, 0.875, 200, '[XingangPan/IBN-Net]'),  # noqa
    ('ibn_resnet101', '0561', '5279c78a0dbfc722cfcfb788af479b6133920528', 'v0.0.127', 'IBN-ResNet-101', '1807.09441', 'in1k', 224, 0.875, 200, '[XingangPan/IBN-Net]'),  # noqa
    ('ibnb_resnet50', '0686', 'e138995e6acda4b496375beac6d01cd7a9f79876', 'v0.0.127', 'IBN(b)-ResNet-50', '1807.09441', 'in1k', 224, 0.875, 200, '[XingangPan/IBN-Net]'),  # noqa
    ('ibn_resnext101_32x4d', '0542', 'b5233c663a4d207d08c21107d6c951956e910be8', 'v0.0.127', 'IBN-ResNeXt-101 (32x4d)', '1807.09441', 'in1k', 224, 0.875, 200, '[XingangPan/IBN-Net]'),  # noqa
    ('ibn_densenet121', '0673', '0ea2c535382c7a3d92e712617d8405ba631c071f', 'v0.0.493', 'IBN-DenseNet-121', '1807.09441', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('ibn_densenet169', '0651', '96dd755e0df8a54349278e0cd23a043a5554de08', 'v0.0.127', 'IBN-DenseNet-169', '1807.09441', 'in1k', 224, 0.875, 200, '[XingangPan/IBN-Net]'),  # noqa
    ('airnet50_1x64d_r2', '0590', '3ec422128d17314124c02e3bb0f77e26777fb385', 'v0.0.120', 'AirNet50-1x64d (r=2)', '', 'in1k', 224, 0.875, 200, '[soeaver/AirNet-PyTorch]'),  # noqa
    ('airnet50_1x64d_r16', '0619', '090179e777f47057bedded22d669bf9f9ce3169c', 'v0.0.120', 'AirNet50-1x64d (r=16)', '', 'in1k', 224, 0.875, 200, '[soeaver/AirNet-PyTorch]'),  # noqa
    ('airnext50_32x4d_r2', '0551', 'c68156e5e446a1116b1b42bc94b3f881ab73fe92', 'v0.0.120', 'AirNeXt50-32x4d (r=2)', '', 'in1k', 224, 0.875, 200, '[soeaver/AirNet-PyTorch]'),  # noqa
    ('bam_resnet50', '0658', '96a37c82bdba821385b29859ad1db83061a0ca5b', 'v0.0.124', 'BAM-ResNet-50', '1807.06514', 'in1k', 224, 0.875, 200, '[Jongchan/attention-module]'),  # noqa
    ('cbam_resnet50', '0605', 'a1172fe679622224dcc88c00020936ad381806fb', 'v0.0.125', 'CBAM-ResNet-50', '1807.06521', 'in1k', 224, 0.875, 200, '[Jongchan/attention-module]'),  # noqa
    ('scnet50', '0547', '18741240886d8e260c228027f3ac44fc1c741f90', 'v0.0.493', 'SCNet-50', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('scnet101', '0575', '40cd4d4ca4407798b569e883eb248a5abfddeb75', 'v0.0.472', 'SCNet-101', '', 'in1k', 224, 0.875, 200, '[MCG-NKU/SCNet]'),  # noqa
    ('scneta50', '0468', 'eb3c25d6c9c8b6c0815a724d798b9b5a2b27ce34', 'v0.0.472', 'SCNet(A)-50', '', 'in1k', 224, 0.875, 200, '[MCG-NKU/SCNet]'),  # noqa
    ('regnetx002', '1066', 'e389d6ce5846b65a5859152243d821308252e202', 'v0.0.475', 'RegNetX-200MF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnetx004', '0866', '9584cc0b8e461f624b3050a59bb36b15e04df980', 'v0.0.479', 'RegNetX-400MF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnetx006', '0791', '30ca597ae0506cb588a7fd8d2fecc4be8402b0cf', 'v0.0.482', 'RegNetX-600MF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnetx008', '0740', '157abf5e7c9244a482bf7655e75bfaea143b4d61', 'v0.0.482', 'RegNetX-800MF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnetx016', '0637', '6de8a97b67a34be6e9acc234261f051da1b9444a', 'v0.0.486', 'RegNetX-1.6GF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnetx032', '0592', '75dc82ab5cbc1b715444b8336b5178580bd6d7d9', 'v0.0.492', 'RegNetX-3.2GF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnetx040', '0586', '54660c9ceef3013f2bdb4ec0fd4ec404f97d9861', 'v0.0.473', 'RegNetX-4.0GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnetx064', '0557', 'e28df79ca0a7a4a6895e9a497f82aebcaffbac79', 'v0.0.473', 'RegNetX-6.4GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnetx080', '0551', 'e8d5baaac129477540ef4614fcb45cafd9173851', 'v0.0.473', 'RegNetX-8.0GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnetx120', '0538', '5eb7ad44af359b980da57a849e93973dcfe2646f', 'v0.0.473', 'RegNetX-12GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnetx160', '0517', '27653d34da3b42c6a868fb2a3ad404107bdeacda', 'v0.0.473', 'RegNetX-16GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnetx320', '0494', '54a1c651c4a248af6fc64f34fce27ede65b32785', 'v0.0.473', 'RegNetX-32GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety002', '0980', '57f04168f284797b799d624d906f5d38dcf23177', 'v0.0.476', 'RegNetY-200MF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety004', '0769', '8c36573f17d3ef2ab8770be2593e94d714b035d7', 'v0.0.481', 'RegNetY-400MF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety006', '0712', 'd6401a374a2c35ed1b2ac29a885438834c38cd0a', 'v0.0.483', 'RegNetY-600MF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety008', '0660', 'ed298c233ef1ce2e3f82a6d23be1eebd43afdd75', 'v0.0.483', 'RegNetY-800MF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety016', '0581', 'b45eccd6d1a80dc6e5608abd89c79db7547f2735', 'v0.0.486', 'RegNetY-1.6GF', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('regnety032', '0404', 'cb3314864b68dfd2e0037928a3b635c81f86ccb2', 'v0.0.473', 'RegNetY-3.2GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety040', '0541', '238ef52bb276368285b3a7b810af4ca78a97c5c9', 'v0.0.473', 'RegNetY-4.0GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety064', '0523', '494ac81bb4bb1e9528b54f751907e3f2c32ba50e', 'v0.0.473', 'RegNetY-6.4GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety080', '0513', 'c69743cd3e6f38d0b6ef366496b95a04b71b2912', 'v0.0.473', 'RegNetY-8.0GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety120', '0492', 'ba4fb43d03c1ad4bc94d6a3ae8a240a4c081c6f7', 'v0.0.473', 'RegNetY-12GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety160', '0503', '2c0ad1f9aad67a79390af3f5ef3dfc70fc8b517d', 'v0.0.473', 'RegNetY-16GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('regnety320', '0474', '643155ebbf529baf5c36d0894ebba86bdf789cc8', 'v0.0.473', 'RegNetY-32GF', '', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('pyramidnet101_a360', '0620', '3a24427baf21ee6566d7e4c7dee25da0e5744f7f', 'v0.0.104', 'PyramidNet-101 (a=360)', '1610.02915', 'in1k', 224, 0.875, 200, '[dyhan0920/Pyramid...PyTorch]'),  # noqa
    ('diracnet18v2', '1170', 'e06737707a1f5a5c7fe4e57da92ed890b034cb9a', 'v0.0.111', 'DiracNetV2-18', '1706.00388', 'in1k', 224, 0.875, 200, '[szagoruyko/diracnets]'),  # noqa
    ('diracnet34v2', '0993', 'a6a661c0c3e96af320e5b9bf65a6c8e5e498a474', 'v0.0.111', 'DiracNetV2-34', '1706.00388', 'in1k', 224, 0.875, 200, '[szagoruyko/diracnets]'),  # noqa
    ('densenet121', '0704', 'cf90d1394d197fde953f57576403950345bd0a66', 'v0.0.314', 'DenseNet-121', '1608.06993', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('densenet161', '0606', 'da489277afe7f53048ec15bed7919486e22f1afa', 'v0.0.432', 'DenseNet-161', '1608.06993', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('densenet169', '0629', '44974a17309bb378e97c8f70f96f961ffbf9458d', 'v0.0.406', 'DenseNet-169', '1608.06993', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('densenet201', '0612', '6adc8625a4afa53e335272bab01b4908a0ca3f00', 'v0.0.426', 'DenseNet-201', '1608.06993', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('condensenet74_c4_g4', '0828', '5ba550494cae7081d12c14b02b2a02365539d377', 'v0.0.4', 'CondenseNet-74 (C=G=4)', '1711.09224', 'in1k', 224, 0.875, 200, '[ShichenLiu/CondenseNet]'),  # noqa
    ('condensenet74_c8_g8', '1006', '3574d874fefc3307f241690bad51f20e61be1542', 'v0.0.4', 'CondenseNet-74 (C=G=8)', '1711.09224', 'in1k', 224, 0.875, 200, '[ShichenLiu/CondenseNet]'),  # noqa
    ('peleenet', '1151', '9c47b80297ac072a923cda763b78e7218cd52d3a', 'v0.0.141', 'PeleeNet', '1804.06882', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('wrn50_2', '0641', '83897ab9f015f6f988e51108e12518b08e1819dd', 'v0.0.113', 'WRN-50-2', '1605.07146', 'in1k', 224, 0.875, 200, '[szagoruyko/functional-zoo]'),  # noqa
    ('drnc26', '0755', '35405bd52a0c721f3dc64f18d433074f263b7339', 'v0.0.116', 'DRN-C-26', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('drnc42', '0657', '7c99c4608a9a5e5f073f657b92f258ba4ba5ac77', 'v0.0.116', 'DRN-C-42', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('drnc58', '0601', '70ec1f56c23da863628d126a6ed0ad10f037a2ac', 'v0.0.116', 'DRN-C-58', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('drnd22', '0823', '5c2c6a0cf992409ab388e04e9fbd06b7141bdf47', 'v0.0.116', 'DRN-D-22', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('drnd38', '0695', '4630f0fb3f721f4a2296e05aacb1231ba7530ae5', 'v0.0.116', 'DRN-D-38', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('drnd54', '0586', 'bfdc1f8826027b247e2757be45b176b3b91b9ea3', 'v0.0.116', 'DRN-D-54', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('drnd105', '0548', 'a643f4dcf9e4b69eab06b76e54ce22169f837592', 'v0.0.116', 'DRN-D-105', '1705.09914', 'in1k', 224, 0.875, 200, '[fyu/drn]'),  # noqa
    ('dpn68', '0679', 'a33c98c783cbf93cca4cc9ce1584da50a6b12077', 'v0.0.310', 'DPN-68', '1707.01629', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('dpn98', '0553', '52c55969835d56185afa497c43f09df07f58f0d3', 'v0.0.17', 'DPN-98', '1707.01629', 'in1k', 224, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('dpn131', '0548', '0c53e5b380137ccb789e932775e8bd8a811eeb3e', 'v0.0.17', 'DPN-131', '1707.01629', 'in1k', 224, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('darknet_tiny', '1784', '4561e1ada619e33520d1f765b3321f7f8ea6196b', 'v0.0.69', 'DarkNet Tiny', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('darknet_ref', '1718', '034595b49113ee23de72e36f7d8a3dbb594615f6', 'v0.0.64', 'DarkNet Ref', '', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('darknet53', '0564', 'b36bef6b297055dda3d17a3f79596511730e1963', 'v0.0.150', 'DarkNet-53', '1804.02767', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('irevnet301', '0841', '95dc8d94257bf16027edd7077b785a8676369fca', 'v0.0.251', 'i-RevNet-301', '1802.07088', 'in1k', 224, 0.875, 200, '[jhjacobsen/pytorch-i-revnet]'),  # noqa
    ('bagnet9', '2961', 'cab1179284e9749697f38c1c7e5f0e172be12c89', 'v0.0.255', 'BagNet-9', '', 'in1k', 224, 0.875, 200, '[wielandbrendel/bag...models]'),  # noqa
    ('bagnet17', '1884', '6b2a100f8d14d4616709586483f625743ed04769', 'v0.0.255', 'BagNet-17', '', 'in1k', 224, 0.875, 200, '[wielandbrendel/bag...models]'),  # noqa
    ('bagnet33', '1301', '4f17b6e837dacd978b15708ffbb2c1e6be3c371a', 'v0.0.255', 'BagNet-33', '', 'in1k', 224, 0.875, 200, '[wielandbrendel/bag...models]'),  # noqa
    ('dla34', '0724', '649c67e61942283abe7f6a798fb9fcae346e5a5d', 'v0.0.486', 'DLA-34', '1707.06484', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('dla46c', '1323', 'efcd363642a4b479892f47edae7440f0eea05edb', 'v0.0.282', 'DLA-46-C', '1707.06484', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('dla46xc', '1269', '00d3754ad0ff22636bb1f4b4fb8baebf4751a1ee', 'v0.0.293', 'DLA-X-46-C', '1707.06484', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('dla60', '0669', 'b2cd6e51a322512a6cb45414982a2ec71285daad', 'v0.0.202', 'DLA-60', '1707.06484', 'in1k', 224, 0.875, 200, '[ucbdrive/dla]'),  # noqa
    ('dla60x', '0575', 'fae6dc6d434d4cf0b52e5d4b3da13b5230d08c02', 'v0.0.493', 'DLA-X-60', '1707.06484', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('dla60xc', '1091', '0f6381f335e5bbb4c69b360be61a4a08e5c7a9de', 'v0.0.289', 'DLA-X-60-C', '1707.06484', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('dla102', '0605', '11df13220b44f51dc8c925fbd9fc334bc8d115b4', 'v0.0.202', 'DLA-102', '1707.06484', 'in1k', 224, 0.875, 200, '[ucbdrive/dla]'),  # noqa
    ('dla102x', '0577', '58331655844f9d95bcf2bb90de6ac9cf3b66bd5e', 'v0.0.202', 'DLA-X-102', '1707.06484', 'in1k', 224, 0.875, 200, '[ucbdrive/dla]'),  # noqa
    ('dla102x2', '0536', '079361117045dc661b63ce4b14408d403bc91844', 'v0.0.202', 'DLA-X2-102', '1707.06484', 'in1k', 224, 0.875, 200, '[ucbdrive/dla]'),  # noqa
    ('dla169', '0566', 'ae0c6a82acfaf9dc459ac5a032106c2727b71d4f', 'v0.0.202', 'DLA-169', '1707.06484', 'in1k', 224, 0.875, 200, '[ucbdrive/dla]'),  # noqa
    ('fishnet150', '0604', 'f5af4873ff5730f589a6c4a505ede8268e6ce3e3', 'v0.0.168', 'FishNet-150', '', 'in1k', 224, 0.875, 200, '[kevin-ssy/FishNet]'),  # noqa
    ('espnetv2_wd2', '2015', 'd234781f81e5d1b5ae6070fc851e3f7bb860b9fd', 'v0.0.238', 'ESPNetv2 x0.5', '1811.11431', 'in1k', 224, 0.875, 200, '[sacmehta/ESPNetv2]'),  # noqa
    ('espnetv2_w1', '1345', '550d54229d7fd8f7c090601c2123ab3ca106393b', 'v0.0.238', 'ESPNetv2 x1.0', '1811.11431', 'in1k', 224, 0.875, 200, '[sacmehta/ESPNetv2]'),  # noqa
    ('espnetv2_w5d4', '1218', '85d97b2b1c9ebb176f634949ef5ca6d7fe70f09c', 'v0.0.238', 'ESPNetv2 x1.25', '1811.11431', 'in1k', 224, 0.875, 200, '[sacmehta/ESPNetv2]'),  # noqa
    ('espnetv2_w3d2', '1129', '3bbb49adaa4fa984a67f82862db7dcfc4998429e', 'v0.0.238', 'ESPNetv2 x1.5', '1811.11431', 'in1k', 224, 0.875, 200, '[sacmehta/ESPNetv2]'),  # noqa
    ('espnetv2_w2', '0961', '13ba0f7200eb745bacdf692905fde711236448ef', 'v0.0.238', 'ESPNetv2 x2.0', '1811.11431', 'in1k', 224, 0.875, 200, '[sacmehta/ESPNetv2]'),  # noqa
    ('hrnet_w18_small_v1', '0901', '300230646c0796b7ba20954a9245803ecac4cdf0', 'v0.0.492', 'HRNet-W18 Small V1', '1908.07919', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('hrnet_w18_small_v2', '0758', '27f85f3124c081c44da8506169f52349aa2e49d5', 'v0.0.421', 'HRNet-W18 Small V2', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w18', '0656', '78b1f85b07e1c2fdd038b4c71ea415015caf5455', 'v0.0.421', 'HRNetV2-W18', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w30', '0578', '839e57ebc3018be3d793e5c5ce1a6655347427b7', 'v0.0.421', 'HRNetV2-W30', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w32', '0581', 'bef9ada0e564bdc1645f80ff69a713b2bc47cfba', 'v0.0.421', 'HRNetV2-W32', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w40', '0553', 'e4b5a38af98c811c10d1b536f4fe48eb20d37e31', 'v0.0.421', 'HRNetV2-W40', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w44', '0563', '9321bfd82f7f02a789dc054ac079706f0e8784c4', 'v0.0.421', 'HRNetV2-W44', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w48', '0548', '40f986102a5650bae90d62c459781704626bd890', 'v0.0.421', 'HRNetV2-W48', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('hrnetv2_w64', '0535', '5961efd0e93740184a582ed4e00de445c91447b9', 'v0.0.421', 'HRNetV2-W64', '1908.07919', 'in1k', 224, 0.875, 200, '[HRNet/HRNet...ation]'),  # noqa
    ('vovnet39', '0564', '63bfa613870b37bd4fb5b71412e7875392aa4f66', 'v0.0.493', 'VoVNet-39', '1904.09730', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('vovnet57', '0628', '99f8a0c8958de38e89194014a08e312205bb3e1e', 'v0.0.431', 'VoVNet-57', '1904.09730', 'in1k', 224, 0.875, 200, '[stigma0617/VoVNet.pytorch]'),  # noqa
    ('selecsls42b', '0611', 'acff1e8b36428719059eec4b60c7b2c045a54d8e', 'v0.0.493', 'SelecSLS-42b', '1907.00837', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('selecsls60', '0612', '5261403fce27354305ea1c1d0a7526bdb7cfb6c9', 'v0.0.430', 'SelecSLS-60', '1907.00837', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('selecsls60b', '0584', '470ace6b9b8db32b84924825908b426a9e38dd09', 'v0.0.430', 'SelecSLS-60b', '1907.00837', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('hardnet39ds', '0881', 'ea47fc939a130a70c5fa3326c3af6ba049a99f92', 'v0.0.485', 'HarDNet-39DS', '1909.00948', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('hardnet68ds', '0756', 'e0da07508c1eb92fee49df42243836892fe2f4c8', 'v0.0.487', 'HarDNet-68DS', '1909.00948', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('hardnet68', '0699', '2e207f79a1995f5f30d5b9fca3391bb8e7b8594f', 'v0.0.435', 'HarDNet-68', '1909.00948', 'in1k', 224, 0.875, 200, '[PingoLH/Pytorch-HarDNet]'),  # noqa
    ('hardnet85', '0611', 'ae85d8af40610e08765f1bfc25b8414ac70d7451', 'v0.0.435', 'HarDNet-85', '1909.00948', 'in1k', 224, 0.875, 200, '[PingoLH/Pytorch-HarDNet]'),  # noqa
    ('squeezenet_v1_0', '1766', 'afdbcf1aef39237300656d2c5a7dba19230e29fc', 'v0.0.128', 'SqueezeNet v1.0', '1602.07360', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('squeezenet_v1_1', '1772', '25b77bc39e35612abbe7c2344d2c3e1e6756c2f8', 'v0.0.88', 'SqueezeNet v1.1', '1602.07360', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('squeezeresnet_v1_0', '1809', '25bfc02edeffb279010242614e7d73bbeacc0170', 'v0.0.178', 'SqueezeResNet v1.0', '1602.07360', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('squeezeresnet_v1_1', '1821', 'c27ed88f1b19eb233d3925efc71c71d25e4c434e', 'v0.0.70', 'SqueezeResNet v1.1', '1602.07360', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sqnxt23_w1', '1906', '97b74e0c4d6bf9fc939771d94b2f6dd97de34024', 'v0.0.171', '1.0-SqNxt-23', '1803.10615', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sqnxt23v5_w1', '1785', '2fe3ad67d73313193a77690b10c17cbceef92340', 'v0.0.172', '1.0-SqNxt-23v5', '1803.10615', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sqnxt23_w3d2', '1350', 'c2f21bce669dbe50fba544bcc39bc1302f63e1e8', 'v0.0.210', '1.5-SqNxt-23', '1803.10615', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sqnxt23v5_w3d2', '1301', 'c244844ba2f02dadd350dddd74e21360b452f9dd', 'v0.0.212', '1.5-SqNxt-23v5', '1803.10615', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sqnxt23_w2', '1100', 'b9bb7302824f89f16e078f0a506e3a8c0ad9c74e', 'v0.0.240', '2.0-SqNxt-23', '1803.10615', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('sqnxt23v5_w2', '1066', '229b0d3de06197e399eeebf42dc826b78f0aba86', 'v0.0.216', '2.0-SqNxt-23v5', '1803.10615', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g1_wd4', '3729', '47dbd0f279da6d3056079bb79ad39cabbb3b9415', 'v0.0.134', 'ShuffleNet x0.25 (g=1)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g3_wd4', '3653', '6abdd65e087e71f80345415cdf7ada6ed2762d60', 'v0.0.135', 'ShuffleNet x0.25 (g=3)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g1_wd2', '2261', 'dae4bdadd7d48bee791dff2a08cd697cff0e9320', 'v0.0.174', 'ShuffleNet x0.5 (g=1)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g3_wd2', '2080', 'ccaacfc8d9ac112c6143269df6e258fd55b662a7', 'v0.0.167', 'ShuffleNet x0.5 (g=3)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g1_w3d4', '1711', '161cd24aa0b2e2afadafa69b44a28af222f2ec7a', 'v0.0.218', 'ShuffleNet x0.75 (g=1)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g3_w3d4', '1650', '3f3b0aef0ce3174c78ff42cf6910c6e34540fc41', 'v0.0.219', 'ShuffleNet x0.75 (g=3)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g1_w1', '1389', '4cfb65a30761fe548e0b5afbb5d89793ec41e4e9', 'v0.0.223', 'ShuffleNet x1.0 (g=1)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g2_w1', '1363', '07256203e217a7b31f1c69a5bd38a6674bce75bc', 'v0.0.241', 'ShuffleNet x1.0 (g=2)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g3_w1', '1348', 'ce54f64ecff87556a4303380f46abaaf649eb308', 'v0.0.244', 'ShuffleNet x1.0 (g=3)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g4_w1', '1335', 'e2415f8270a4b6cbfe7dc97044d497edbc898577', 'v0.0.245', 'ShuffleNet x1.0 (g=4)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenet_g8_w1', '1342', '9a979b365424addba75c559a61a77ac7154b26eb', 'v0.0.250', 'ShuffleNet x1.0 (g=8)', '1707.01083', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2_wd2', '1865', '9c22238b5fa9c09541564e8ed7f357a5f7e8cd7c', 'v0.0.90', 'ShuffleNetV2 x0.5', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2_w1', '1163', 'c71dfb7a814c8d8ef704bdbd80995e9ea49ff4ff', 'v0.0.133', 'ShuffleNetV2 x1.0', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2_w3d2', '0942', '26a9230405d956643dcd563a5a383844c49b5907', 'v0.0.288', 'ShuffleNetV2 x1.5', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2_w2', '0845', '337255f6ad40a93c2f23fc593bad4b2755a327fa', 'v0.0.301', 'ShuffleNetV2 x2.0', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2b_wd2', '1822', '01d18d6fa1a6136f605a4277f47c9a757f9ede3b', 'v0.0.157', 'ShuffleNetV2b x0.5', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2b_w1', '1125', '6a5d3dc446e6a00cf60fe8aa2f4139d74d766305', 'v0.0.161', 'ShuffleNetV2b x1.0', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2b_w3d2', '0911', 'f2106fee0748d7f0d40db16b228782b6d7636737', 'v0.0.203', 'ShuffleNetV2b x1.5', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('shufflenetv2b_w2', '0834', 'cb36b92ca4ca3bee470b739021d01177e0601c5f', 'v0.0.242', 'ShuffleNetV2b x2.0', '1807.11164', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet108_8x1_g3', '2076', '6acc82e46dfc1ce0dd8c59668aed4a464c8cbdb5', 'v0.0.89', '108-MENet-8x1 (g=3)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet128_8x1_g4', '1959', '48fa80fc363adb88ff580788faa8053c9d7507f3', 'v0.0.103', '128-MENet-8x1 (g=4)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet160_8x1_g8', '2084', '0f4fce43b4234c5bca5dd76450b698c2d4daae65', 'v0.0.154', '160-MENet-8x1 (g=8)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet228_12x1_g3', '1316', '5b670c42031d0078e2ae981829358d7c1b92ee30', 'v0.0.131', '228-MENet-12x1 (g=3)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet256_12x1_g4', '1252', '14c6c86df96435c693eb7d0fcd8d3bf4079dd621', 'v0.0.152', '256-MENet-12x1 (g=4)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet348_12x1_g3', '0958', 'ad50f635a1f7b799a19a0a9c71aa9939db8ffe77', 'v0.0.173', '348-MENet-12x1 (g=3)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet352_12x1_g8', '1200', '4ee200c5c98c64a2503cea82ebf62d1d3c07fb91', 'v0.0.198', '352-MENet-12x1 (g=8)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('menet456_24x1_g3', '0799', '826c002244f1cdc945a95302b1ce5c66d949db74', 'v0.0.237', '456-MENet-24x1 (g=3)', '1803.09127', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenet_wd4', '2249', '1ad5e8fe8674cdf7ffda8450095eb96d227397e0', 'v0.0.62', 'MobileNet x0.25', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenet_wd2', '1355', '41a21242c95050407df876cfa44bb5d3676aa751', 'v0.0.156', 'MobileNet x0.5', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenet_w3d4', '1076', 'd801bcaea83885b16a0306b8b77fe314bbc585c3', 'v0.0.130', 'MobileNet x0.75', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenet_w1', '0895', '7e1d739f0fd4b95c16eef077c5dc0a5bb1da8ad5', 'v0.0.155', 'MobileNet x1.0', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetb_wd4', '2201', '428da928e43ecc387763bea8faa8ccc51244cb0e', 'v0.0.481', 'MobileNet(B) x0.25', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetb_wd2', '1310', 'd1549ead8d09cc81f8a1542952a8a30fa937caee', 'v0.0.480', 'MobileNet(B) x0.5', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetb_w3d4', '1037', '8d732bc9e6f5326ce1f31ce836623ac0970f1e16', 'v0.0.481', 'MobileNet(B) x0.75', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetb_w1', '0816', '107275a1173b201634cca077dd126a550bc99dae', 'v0.0.489', 'MobileNet(B) x1.0', '1704.04861', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('fdmobilenet_wd4', '3098', '2b22b709a05d7ca6e43acc6f3a9f27d0eb2e01cd', 'v0.0.177', 'FD-MobileNet x0.25', '1802.03750', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('fdmobilenet_wd2', '2015', '414dbeedb2f829dcd8f94cd7fef10aae6829f06f', 'v0.0.83', 'FD-MobileNet x0.5', '1802.03750', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('fdmobilenet_w3d4', '1641', '5561d58aa8889d8d93f2062a2af4e4b35ad7e769', 'v0.0.159', 'FD-MobileNet x0.75', '1802.03750', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('fdmobilenet_w1', '1338', '9d026c04112de9f40e15fa40457d77941443c327', 'v0.0.162', 'FD-MobileNet x1.0', '1802.03750', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_wd4', '2451', '05e1e3a286b27c17ea11928783c4cd48b1e7a9b2', 'v0.0.137', 'MobileNetV2 x0.25', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_wd2', '1493', 'b82d79f6730eac625e6b55b0618bff8f7a1ed86d', 'v0.0.170', 'MobileNetV2 x0.5', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_w3d4', '1082', '8656de5a8d90b29779c35c5ce521267c841fd717', 'v0.0.230', 'MobileNetV2 x0.75', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_w1', '0887', '13a021bca5b679b76156829743f7182da42e8bb6', 'v0.0.213', 'MobileNetV2 x1.0', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_wd4', '2368', '399f95e6cb3c15d57516c1d328201a0af3de5882', 'v0.0.483', 'MobileNetV2b x0.25', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_wd2', '1408', 'f820ea858dd7be1bbe0ca4639581911d98183cde', 'v0.0.486', 'MobileNetV2b x0.5', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_w3d4', '1105', '0924efc9ca677d2bccfe3987b1e0e1e47afe69e8', 'v0.0.483', 'MobileNetV2b x0.75', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_w1', '0912', '2bcab1d0cd3be4eb270d65e390ff7c9776e38a04', 'v0.0.483', 'MobileNetV2b x1.0', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv3_large_w1', '0744', 'b59cae6daf1edc5f412fcd794693bb22dc3d4573', 'v0.0.491', 'MobileNetV3 L/224/1.0', '1905.02244', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('igcv3_wd4', '2871', 'c9f28301391601e5e8ae93139431a9e0d467317c', 'v0.0.142', 'IGCV3 x0.25', '1806.00178', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('igcv3_wd2', '1732', '8c504f443283d8a32787275b23771082fcaab61b', 'v0.0.132', 'IGCV3 x0.5', '1806.00178', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('igcv3_w3d4', '1140', '63f43cf8d334111d55d06f2f9bf7e1e4871d162c', 'v0.0.207', 'IGCV3 x0.75', '1806.00178', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('igcv3_w1', '0920', '12385791681f09adb3a08926c95471f332f538b6', 'v0.0.243', 'IGCV3 x1.0', '1806.00178', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mnasnet_b1', '0740', '7025b43c5c0251980ada2c591dd3e7e28d856e79', 'v0.0.493', 'MnasNet-B1', '1807.11626', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mnasnet_a1', '0720', 'e155916ce24d06e273e8f90540707bcb7e1f9eab', 'v0.0.486', 'MnasNet-A1', '1807.11626', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('darts', '0775', 'fc3171c5b89b270fc7673dbbb5047f5879d7e774', 'v0.0.485', 'DARTS', '1806.09055', 'in1k', 224, 0.875, 200, '[quark0/darts]'),  # noqa
    ('proxylessnas_cpu', '0761', 'fe9572b11899395acbeef9374827dcc04e103ce3', 'v0.0.304', 'ProxylessNAS CPU', '1812.00332', 'in1k', 224, 0.875, 200, '[MIT-HAN-LAB/ProxylessNAS]'),  # noqa
    ('proxylessnas_gpu', '0745', 'acca5941c454d896410060434b8f983d2db80727', 'v0.0.333', 'ProxylessNAS GPU', '1812.00332', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('proxylessnas_mobile', '0780', '639a90c27de088402db76b09e410326795b6fbdd', 'v0.0.304', 'ProxylessNAS Mobile', '1812.00332', 'in1k', 224, 0.875, 200, '[MIT-HAN-LAB/ProxylessNAS]'),  # noqa
    ('proxylessnas_mobile14', '0662', '0c0ad983f4fb88470d0f3e557d0b23f15e16624f', 'v0.0.331', 'ProxylessNAS Mob-14', '1812.00332', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('fbnet_cb', '0762', '2edb61f8e4b5c45d958d0e57beff41fbfacd6061', 'v0.0.415', 'FBNet-Cb', '1812.03443', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('xception', '0549', 'e4f0232c99fa776e630189d62fea18e248a858b2', 'v0.0.115', 'Xception', '1610.02357', 'in1k', 299, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('inceptionv3', '0565', 'cf4061800bc1dc3b090920fc9536d8ccc15bb86e', 'v0.0.92', 'InceptionV3', '1512.00567', 'in1k', 299, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('inceptionv4', '0529', '5cb7b4e4b8f62d6b4346855d696b06b426b44f3d', 'v0.0.105', 'InceptionV4', '1602.07261', 'in1k', 299, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('inceptionresnetv2', '0490', '1d1b4d184e6d41091c5ac3321d99fa554b498dbe', 'v0.0.107', 'InceptionResNetV2', '1602.07261', 'in1k', 299, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('polynet', '0452', '6a1b295dad3f261b48e845f1b283e4eef3ab5a0b', 'v0.0.96', 'PolyNet', '1611.05725', 'in1k', 331, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('nasnet_4a1056', '0816', 'd21bbaf5e937c2e06134fa40e7bdb1f501423b86', 'v0.0.97', 'NASNet-A 4@1056', '1707.07012', 'in1k', 224, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('nasnet_6a4032', '0421', 'f354d28f4acdde399e081260c3f46152eca5d27e', 'v0.0.101', 'NASNet-A 6@4032', '1707.07012', 'in1k', 331, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('pnasnet5large', '0428', '65de46ebd049e494c13958d5671aba5abf803ff3', 'v0.0.114', 'PNASNet-5-Large', '1712.00559', 'in1k', 331, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('spnasnet', '0798', 'a25ca15768d91c0c09b473352bf54a2b954257d4', 'v0.0.490', 'SPNASNet', '1904.02877', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('efficientnet_b0', '0752', '0e3861300b8f1d1d0fb1bd15f0e06bba1ad6309b', 'v0.0.364', 'EfficientNet-B0', '1905.11946', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('efficientnet_b1', '0638', 'ac77bcd722dc4f3edfa24b9fb7b8f9cece3d85ab', 'v0.0.376', 'EfficientNet-B1', '1905.11946', 'in1k', 240, 0.882, 200, ''),  # noqa
    ('efficientnet_b0b', '0702', 'ecf61b9b50666a6b444a9d789a5ff1087c65d0d8', 'v0.0.403', 'EfficientNet-B0b', '1905.11946', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b1b', '0594', '614e81663902850a738fa6c862fe406ecf205f73', 'v0.0.403', 'EfficientNet-B1b', '1905.11946', 'in1k', 240, 0.882, 200, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b2b', '0527', '531f10e6898778b7c3a82c2c149f8b3e6393a892', 'v0.0.403', 'EfficientNet-B2b', '1905.11946', 'in1k', 260, 0.890, 100, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b3b', '0445', '3c5fbba8c86121d4bc3bbc169804f24dd4c3d1f6', 'v0.0.403', 'EfficientNet-B3b', '1905.11946', 'in1k', 300, 0.904, 90, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b4b', '0389', '6305bfe688b261f0d4fef6829f520d5c98c46301', 'v0.0.403', 'EfficientNet-B4b', '1905.11946', 'in1k', 380, 0.922, 80, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b5b', '0337', 'e1c2ffcf710cbd3c53b9c08723282a370906731c', 'v0.0.403', 'EfficientNet-B5b', '1905.11946', 'in1k', 456, 0.934, 70, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b6b', '0323', 'e5c1d7c35fcff5fac07921a7696f7c04aba84012', 'v0.0.403', 'EfficientNet-B6b', '1905.11946', 'in1k', 528, 0.942, 60, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b7b', '0322', 'b9c5965a1e2572aaa772e20e8a2e3af7b4bee9a6', 'v0.0.403', 'EfficientNet-B7b', '1905.11946', 'in1k', 600, 0.949, 50, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b0c', '0675', '21778c6e3b5a1b9b08b60c3e69401ce7e12bead4', 'v0.0.433', 'EfficientNet-B0с', '1905.11946', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b1c', '0569', '239ed6a412530f60f810b29807da70c8ca63d8cc', 'v0.0.433', 'EfficientNet-B1с', '1905.11946', 'in1k', 240, 0.882, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b2c', '0503', 'be48d3d79f25a13a807b137d8a7ced41e8aab2bf', 'v0.0.433', 'EfficientNet-B2с', '1905.11946', 'in1k', 260, 0.890, 100, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b3c', '0442', 'ea7080aba3fc20ac25c3c925bfadf1e8c1e7df4d', 'v0.0.433', 'EfficientNet-B3с', '1905.11946', 'in1k', 300, 0.904, 90, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b4c', '0369', '5954cc05cfba3b0c8ee488b4488354fc0cef6623', 'v0.0.433', 'EfficientNet-B4с', '1905.11946', 'in1k', 380, 0.922, 80, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b5c', '0310', '589fefc6de5d93b54698b5b03f1e05637f9d0cb6', 'v0.0.433', 'EfficientNet-B5с', '1905.11946', 'in1k', 456, 0.934, 70, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b6c', '0296', '546e61da82bec69e3db5870b8df977e4615f7b32', 'v0.0.433', 'EfficientNet-B6с', '1905.11946', 'in1k', 528, 0.942, 60, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b7c', '0288', '13d683f2ca56c1007acd9ad0be450f45efeec828', 'v0.0.433', 'EfficientNet-B7с', '1905.11946', 'in1k', 600, 0.949, 50, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b8c', '0276', 'a9973d66d599c4e83029577842c039a20799f2c9', 'v0.0.433', 'EfficientNet-B8с', '1905.11946', 'in1k', 672, 0.954, 50, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_edge_small_b', '0640', 'e27c3444406ebddd86824e41a924c0b8188c4067', 'v0.0.434', 'EfficientNet-Edge-Small-b', '1905.11946', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_edge_medium_b', '0563', '99fa34c7044281e521fb7cf4267763a5b03b7f1c', 'v0.0.434', 'EfficientNet-Edge-Medium-b', '1905.11946', 'in1k', 240, 0.882, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_edge_large_b', '0491', 'd502326f9568f096491354a117f12562cf47e038', 'v0.0.434', 'EfficientNet-Edge-Large-b', '1905.11946', 'in1k', 300, 0.904, 90, '[rwightman/pyt...models]*'),  # noqa
    ('mixnet_s', '0717', 'ab2c4e37062e7ea34a2cdd112f9354d4e67a0fef', 'v0.0.493', 'MixNet-S', '1907.09595', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mixnet_m', '0647', '4d90d345a38ba5041ac5cae2921e07d1eca083b2', 'v0.0.493', 'MixNet-M', '1907.09595', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mixnet_l', '0582', '6cf2c97538d4173d9f6bc80a6ec299463df2d1f3', 'v0.0.414', 'MixNet-L', '1907.09595', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('resneta10', '1190', 'a066e5e07f13f8f2a67971931496d1c1ac09bbe1', 'v0.0.484', 'ResNet(A)-10', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resnetabc14b', '0990', 'bad51cb083aae58479112ad11a3fe9430346e185', 'v0.0.477', 'ResNet(A)-BC-14b', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resneta18', '0831', 'e9f206f480c46b489fbd300fa77db31d740c4f3b', 'v0.0.486', 'ResNet(A)-18', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resneta50b', '0556', '7cedbb3bd808c0644b4afe1d52e7dad6abd33516', 'v0.0.492', 'ResNet(A)-50b', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resneta101b', '0503', '80d275397837e8f40908cdb4b2cc3e427a1196ee', 'v0.0.452', 'ResNet(A)-101b', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resneta152b', '0482', '9b55f86f63c7402c0093903883e114a9f4809061', 'v0.0.452', 'ResNet(A)-152b', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resnetd50b', '0565', 'ec03d815c0f016c6517ed7b4b40126af46ceb8a4', 'v0.0.296', '', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resnetd101b', '0473', 'f851c920ec1fe4f729d339c933535d038bf2903c', 'v0.0.296', '', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('resnetd152b', '0482', '112e216da50eb20d52c509a28c97b05ef819cefe', 'v0.0.296', '', '', 'in1k', 0, 0.0, 0, ''),  # noqa
    ('nin_cifar10', '0743', '795b082470b58c1aa94e2f861514b7914f6e2f58', 'v0.0.175', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('nin_cifar100', '2839', '627a11c064eb44c6451fe53e0becfc21a6d57d7f', 'v0.0.183', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('nin_svhn', '0376', '1205dc06a4847bece8159754033f325f75565c02', 'v0.0.270', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet20_cifar10', '0597', '9b0024ac4c2f374cde2c5052e0d0344a75871cdb', 'v0.0.163', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet20_cifar100', '2964', 'a5322afed92fa96cb7b3453106f73cf38e316151', 'v0.0.180', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet20_svhn', '0343', '8232e6e4c2c9fac1200386b68311c3bd56f483f5', 'v0.0.265', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet56_cifar10', '0452', '628c42a26fe347b84060136212e018df2bb35e0f', 'v0.0.163', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet56_cifar100', '2488', 'd65f53b10ad5d124698e728432844c65261c3107', 'v0.0.181', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet56_svhn', '0275', '6e08ed929b8f0ee649f75464f06b557089023290', 'v0.0.265', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet110_cifar10', '0369', '4d6ca1fc02eaeed724f4f596011e391528536049', 'v0.0.163', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet110_cifar100', '2280', 'd8d397a767db6d22af040223ec8ae342a088c3e5', 'v0.0.190', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet110_svhn', '0245', 'c971f0a38943d8a75386a60c835cc0843c2f6c1c', 'v0.0.265', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet164bn_cifar10', '0368', '74ae9f4bccb7fb6a8f3f603fdabe8d8632c46b2f', 'v0.0.179', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet164bn_cifar100', '2044', '8fa07b7264a075fa5add58f4c676b99a98fb1c89', 'v0.0.182', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet164bn_svhn', '0242', '549413723d787cf7e96903427a7a14fb3ea1a4c1', 'v0.0.267', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet272bn_cifar10', '0333', '84f28e0ca97eaeae0eb07e9f76054c1ba0c77c0e', 'v0.0.368', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet272bn_cifar100', '2007', 'a80d2b3ce14de6c90bf22d210d76ebd4a8c91928', 'v0.0.368', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet272bn_svhn', '0243', 'ab1d7da51f52cc6acb2e759736f2d58a77ce895e', 'v0.0.368', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet542bn_cifar10', '0343', '0fd36dd16587f49d33e0e36f1e8596d021a11439', 'v0.0.369', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet542bn_cifar100', '1932', 'a631d3ce5f12e145637a7b2faee663cddc94c354', 'v0.0.369', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet542bn_svhn', '0234', '04396c973121e356f2efda9a28c4e4086f1511b2', 'v0.0.369', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet1001_cifar10', '0328', '77a179e240808b7aa3534230d39b845a62413ca2', 'v0.0.201', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet1001_cifar100', '1979', '2728b558748f9c3e70db179afb6c62358020858b', 'v0.0.254', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet1001_svhn', '0241', '9e3d4bb55961db4c0f46a961b5323a4e03aea602', 'v0.0.408', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet1202_cifar10', '0353', '1d5a21290117903fb5fd6ba59f3f7e7da7c08836', 'v0.0.214', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet1202_cifar100', '2156', '86ecd091e5ac9677bf4518c644d08eb3e1d1708a', 'v0.0.410', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet20_cifar10', '0651', '76cec68d11de5b25be2ea5935681645b76195f1d', 'v0.0.164', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet20_cifar100', '3022', '3dbfa6a2b850572bccb28cc2477a0e46c24abcb8', 'v0.0.187', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet20_svhn', '0322', 'c3c00fed49c1d6d9deda6436d041c5788d549299', 'v0.0.269', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet56_cifar10', '0449', 'e9124fcf167d8ca50addef00c3afa4da9f828f29', 'v0.0.164', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet56_cifar100', '2505', 'ca90a2be6002cd378769b9d4e7c497dd883d31d9', 'v0.0.188', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet56_svhn', '0280', 'b51b41476710c0e2c941356ffe992ff883a3ee87', 'v0.0.269', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet110_cifar10', '0386', 'cc08946a2126a1224d1d2560a47cf766a763c52c', 'v0.0.164', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet110_cifar100', '2267', '3954e91581b7f3e5f689385d15f618fe16e995af', 'v0.0.191', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet110_svhn', '0279', 'aa49e0a3c4a918e227ca2d5a5608704f026134c3', 'v0.0.269', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet164bn_cifar10', '0364', '429012d412e82df7961fa071f97c938530e1b005', 'v0.0.196', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet164bn_cifar100', '2018', 'a8e67ca6e14f88b009d618b0e9b554312d862174', 'v0.0.192', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet164bn_svhn', '0258', '94d42de440d5f057a38f4c8cdbdb24acfee3981c', 'v0.0.269', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet272bn_cifar10', '0325', '1a6a016eb4e4a5549c1fcb89ed5af4c1e5715b72', 'v0.0.389', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet272bn_cifar100', '1963', '6fe0d2e24a60d12ab6b3d0e46065e2f14a46bc0b', 'v0.0.389', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet272bn_svhn', '0234', 'c04ef5c20a53f76824339fe75185d181be4bce61', 'v0.0.389', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet542bn_cifar10', '0314', '66fd6f2033dff08428e586bcce3e5151ed4274f9', 'v0.0.391', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet542bn_cifar100', '1871', '07f1fb258207d295789981519e8dab892fc08f8d', 'v0.0.391', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet542bn_svhn', '0236', '6bdf92368873ce1288526dc405f15e689a1d3117', 'v0.0.391', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet1001_cifar10', '0265', '9fedfe5fd643e7355f1062a6db68da310c8962be', 'v0.0.209', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet1001_cifar100', '1841', '88f14ed9df1573e98b0ec2a07009a15066855fda', 'v0.0.283', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('preresnet1202_cifar10', '0339', '6fc686b02191226f39e25a76fc5da26857f7acd9', 'v0.0.246', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext29_32x4d_cifar10', '0315', '30413525cd4466dbef759294eda9b702bc39648f', 'v0.0.169', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext29_32x4d_cifar100', '1950', '13ba13d92f6751022549a3b370ae86d3b13ae2d1', 'v0.0.200', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext29_32x4d_svhn', '0280', 'e85c5217944cdfafb0a538dd7cc817cffaada7c4', 'v0.0.275', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext29_16x64d_cifar10', '0241', '4133d3d04f9b10b132dcb959601d36f10123f8c2', 'v0.0.176', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext29_16x64d_cifar100', '1693', '05e9a7f113099a98b219cad622ecfad5517a3b54', 'v0.0.322', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext29_16x64d_svhn', '0268', '74332b714cd278bfca3f09dafe2a9d117510e9a4', 'v0.0.358', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext272_1x64d_cifar10', '0255', '070ccc35c2841b7715b9eb271197c9bb316f3093', 'v0.0.372', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext272_1x64d_cifar100', '1911', '114eb0f8a0d471487e819b8fd156c1286ef91b7a', 'v0.0.372', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext272_1x64d_svhn', '0235', 'ab0448469bbd7d476f8bed1bf86403304b028e7c', 'v0.0.372', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext272_2x32d_cifar10', '0274', 'd2ace03c413be7e42c839c84db8dd0ebb5d69512', 'v0.0.375', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext272_2x32d_cifar100', '1834', '0b30c4701a719995412882409339f3553a54c9d1', 'v0.0.375', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnext272_2x32d_svhn', '0244', '39b8a33612d335a0193b867b38c0b09d168de6c3', 'v0.0.375', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet20_cifar10', '0601', '935d89433e803c8a3027c81f1267401e7caccce6', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet20_cifar100', '2854', '8c7abf66d8c1418cb3ca760f5d1efbb42738036b', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet20_svhn', '0323', 'd77df31c62d1504209a5ba47e59ccb0ae84500b2', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet56_cifar10', '0413', 'b61c143989cb2901bec48dded4c6ddcae91aabc4', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet56_cifar100', '2294', '7fa54f4593f364c2363cb3ee8d5bc1285af1ade5', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet56_svhn', '0264', '93839c762a97bd0b5bd27c71fd64c227afdae3ed', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet110_cifar10', '0363', '1ddec2309ff61c2c0e14c96d51a1b846afdc2acc', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet110_cifar100', '2086', 'a82c30938028a172dd6a124152bc0952b55a2f49', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet110_svhn', '0235', '9572ba7394c774b8d056b24a7631ef47e53024b8', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet164bn_cifar10', '0339', '1085dab6467cb18e554123663816094f080fc626', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet164bn_cifar100', '1995', '97dd4ab630f6277cf7b07cbdcbf4ae8ddce4d401', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet164bn_svhn', '0245', 'af0a90a50fb3c91eef039178a681e69aae703f3a', 'v0.0.362', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet272bn_cifar10', '0339', '812db5187bab9aa5203611c1c174d0e51c81761c', 'v0.0.390', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet272bn_cifar100', '1907', '179e1c38ba714e1babf6c764ca735f256d4cd122', 'v0.0.390', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet272bn_svhn', '0238', '0e16badab35b483b1a1b0e7ea2a615de714f7424', 'v0.0.390', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet542bn_cifar10', '0347', 'd1542214765f1923f2fdce810aef5dc2e523ffd2', 'v0.0.385', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet542bn_cifar100', '1887', '9c4e7623dc06a56edabf04f4427286916843df85', 'v0.0.385', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('seresnet542bn_svhn', '0226', '71a8f2986cbc1146f9a41d1a08ecba52649b8efd', 'v0.0.385', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet20_cifar10', '0618', 'eabb3fce8373cbeb412ced9a79a1e2f9c6c3689c', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet20_cifar100', '2831', 'fe7558e0ae554d39d8761f234e8328262ee31efd', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet20_svhn', '0324', '061daa587dd483744d5b60d2fd3b2750130dd8a1', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet56_cifar10', '0451', 'fc23e153ccfaddd52de61d77570a0befeee1e687', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet56_cifar100', '2305', 'c4bdc5d7bbaa0d9f6e2ffdf2abe4808ad26d0f66', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet56_svhn', '0271', 'c91e922f1b3d0ea634db8e467e9ab4a6b8dc7722', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet110_cifar10', '0454', '418daea9d2253a3e9fbe4eb80eb4dcc6f29d5925', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet110_cifar100', '2261', 'ed7d3c3e51ed2ea9a827ed942e131c78784813b7', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet110_svhn', '0259', '556909fd942d3a42e424215374b340680b705424', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet164bn_cifar10', '0373', 'ff353a2910f85db66d8afca0a4150176bcdc7a69', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet164bn_cifar100', '2005', 'df1163c4d9de72c53efc37758773cc943be7f055', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet164bn_svhn', '0256', 'f8dd4e06596841f0c7f9979fb566b9e57611522f', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet272bn_cifar10', '0339', '606d096422394857cb1f45ecd7eed13508158a60', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet272bn_cifar100', '1913', 'cb71511346e441cbd36bacc93c821e8b6101456a', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet272bn_svhn', '0249', '904d74a2622d870f8a2384f9e50a84276218acc3', 'v0.0.379', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet542bn_cifar10', '0308', '652bc8846cfac7a2ec6625789531897339800202', 'v0.0.382', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet542bn_cifar100', '1945', '9180f8632657bb8f7b6583e47d04ce85defa956c', 'v0.0.382', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('sepreresnet542bn_svhn', '0247', '318a8325afbfbaa8a35d54cbd1fa7da668ef1389', 'v0.0.382', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a48_cifar10', '0372', 'eb185645cda89e0c3c47b11c4b2d14ff18fa0ae1', 'v0.0.184', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a48_cifar100', '2095', '95da1a209916b3cf4af7e8dc44374345a88c60f4', 'v0.0.186', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a48_svhn', '0247', 'd48bafbebaabe9a68e5924571752b3d7cd95d311', 'v0.0.281', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a84_cifar10', '0298', '7b835a3cf19794478d478aced63ca9e855c3ffeb', 'v0.0.185', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a84_cifar100', '1887', 'ff711084381f217f84646c676e4dcc90269dc516', 'v0.0.199', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a84_svhn', '0243', '971576c61cf30e02f13da616afc9848b2a609e0e', 'v0.0.392', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a270_cifar10', '0251', '31bdd9d51ec01388cbb2adfb9f822c942de3c4ff', 'v0.0.194', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a270_cifar100', '1710', '7417dd99069d6c8775454475968ae226b9d5ac83', 'v0.0.319', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet110_a270_svhn', '0238', '3047a9bb7c92a09adf31590e3fe6c9bcd36c7a67', 'v0.0.393', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet164_a270_bn_cifar10', '0242', 'daa2a402c1081323b8f2239f2201246953774e84', 'v0.0.264', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet164_a270_bn_cifar100', '1670', '54d99c834bee0ed7402ba46e749e48182ad1599a', 'v0.0.312', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet164_a270_bn_svhn', '0233', '42d4c03374f32645924fc091d599ef7b913e2d32', 'v0.0.396', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet200_a240_bn_cifar10', '0244', '44433afdd2bc32c55dfb1e8347bc44d1c2bf82c7', 'v0.0.268', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet200_a240_bn_cifar100', '1609', '087c02d6882e274054f44482060f193b9fc208bb', 'v0.0.317', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet200_a240_bn_svhn', '0232', 'f9660c25f1bcff9d361aeca8fb3efaccdc0546e7', 'v0.0.397', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet236_a220_bn_cifar10', '0247', 'daa91d74979c451ecdd8b59e4350382966f25831', 'v0.0.285', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet236_a220_bn_cifar100', '1634', 'a45816ebe1d6a67468b78b7a93334a41aca1c64b', 'v0.0.312', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet236_a220_bn_svhn', '0235', 'f74fe248b6189699174c90bc21e7949d3cca8130', 'v0.0.398', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet272_a200_bn_cifar10', '0239', '586b1ecdc8b34b69dcae4ba57f71c24583cca9b1', 'v0.0.284', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet272_a200_bn_cifar100', '1619', '98bc2f48da0f2c68bc5376c17b0aefc734a64881', 'v0.0.312', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('pyramidnet272_a200_bn_svhn', '0240', '96f6e740dcdc917d776f6df855e3437c93d0da4f', 'v0.0.404', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k12_cifar10', '0561', '8b8e819467a2e4c450e4ff72ced80582d0628b68', 'v0.0.193', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k12_cifar100', '2490', 'd182c224d6df2e289eef944d54fea9fd04890961', 'v0.0.195', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k12_svhn', '0305', 'ac0de84a1a905b768c66f0360f1fb9bd918833bf', 'v0.0.278', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k12_bc_cifar10', '0643', '6dc86a2ea1d088f088462f5cbac06cc0f37348c0', 'v0.0.231', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k12_bc_cifar100', '2841', '1e9db7651a21e807c363c9f366bd9e91ce2f296f', 'v0.0.232', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k12_bc_svhn', '0320', '320760528b009864c68ff6c5b874e9f351ea7a07', 'v0.0.279', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k24_bc_cifar10', '0452', '669c525548a4a2392c5e3c380936ad019f2be7f9', 'v0.0.220', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k24_bc_cifar100', '2267', '411719c0177abf58eddaddd05511c86db0c9d548', 'v0.0.221', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k24_bc_svhn', '0290', 'f4440d3b8c974c9e1014969f4d5832c6c90195d5', 'v0.0.280', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k36_bc_cifar10', '0404', 'b1a4cc7e67db1ed8c5583a59dc178cc7dc2c572e', 'v0.0.224', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k36_bc_cifar100', '2050', 'cde836fafec1e5d6c8ed69fd3cfe322e8e71ef1d', 'v0.0.225', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet40_k36_bc_svhn', '0260', '8c7db0a291a0797a8bc3c709bff7917bc41471cc', 'v0.0.311', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k12_cifar10', '0366', '26089c6e70236e8f25359de6fda67b84425888ab', 'v0.0.205', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k12_cifar100', '1964', '5e10cd830c06f6ab178e9dd876c83c754ca63f00', 'v0.0.206', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k12_svhn', '0260', '57fde50e9f44edc0486b62a1144565bc77d5bdfe', 'v0.0.311', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k24_cifar10', '0313', '397f0e39b517c05330221d4f3a9755eb5e561be1', 'v0.0.252', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k24_cifar100', '1808', '1c0a8067283952709d8e09c774c3a404f51e0079', 'v0.0.318', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k12_bc_cifar10', '0416', 'b9232829b13c3f3f2ea15f4be97f500b7912c3c2', 'v0.0.189', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet100_k12_bc_cifar100', '2119', '05a6f02772afda51a612f5b92aadf19ffb60eb72', 'v0.0.208', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet190_k40_bc_cifar10', '0252', '2896fa088aeaef36fcf395d404d97ff172d78943', 'v0.0.286', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet250_k24_bc_cifar10', '0267', 'f8f9d3052bae1fea7e33bb1ce143c38b4aa5622b', 'v0.0.290', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('densenet250_k24_bc_cifar100', '1739', '09ac3e7d9fbe6b97b170bd838dac20ec144b4e49', 'v0.0.303', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('xdensenet40_2_k24_bc_cifar10', '0531', 'b91a9dc35877c4285fe86f49953d1118f6b69e57', 'v0.0.226', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('xdensenet40_2_k24_bc_cifar100', '2396', '0ce8f78ab9c6a4786829f816ae0615c7905f292c', 'v0.0.227', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('xdensenet40_2_k24_bc_svhn', '0287', 'fd9b6def10f154378a76383cf023d7f2f5ae02ab', 'v0.0.306', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('xdensenet40_2_k36_bc_cifar10', '0437', 'ed264a2060836c7440f0ccde57315e1ec6263ff0', 'v0.0.233', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('xdensenet40_2_k36_bc_cifar100', '2165', '6f68f83dc31dea5237e6362e6c6cfeed48a8d9e3', 'v0.0.234', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('xdensenet40_2_k36_bc_svhn', '0274', '540a69f13a6ce70bfef13657e70dfa414d966581', 'v0.0.306', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn16_10_cifar10', '0293', 'ce810d8a17a2deb73eddb5bec8709f93278bc53e', 'v0.0.166', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn16_10_cifar100', '1895', 'bef9809c845deb1b2bb0c9aaaa7c58bd97740504', 'v0.0.204', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn16_10_svhn', '0278', '5ab2a4edd5398a03d2e28db1b075bf0313ae5828', 'v0.0.271', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn28_10_cifar10', '0239', 'fe97dcd6d0dd8dda8e9e38e6cfa320cffb9955ce', 'v0.0.166', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn28_10_cifar100', '1788', '8c3fe8185d3af9cc3813fe376cab895f6780ac18', 'v0.0.320', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn28_10_svhn', '0271', 'd62b6bbaef7228706a67c2c8416681f97c6d4688', 'v0.0.276', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn40_8_cifar10', '0237', '8dc84ec730f35c4b8968a022bc045c0665410840', 'v0.0.166', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn40_8_cifar100', '1803', '0d18bfbff85951d88a881dc6a15ad46f56ea8c28', 'v0.0.321', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn40_8_svhn', '0254', 'dee59602c10e5d56bd9c168e8e8400792b9a8b08', 'v0.0.277', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn20_10_1bit_cifar10', '0326', 'e6140f8a5eacd5227e8748457b5ee9f5f519d2d5', 'v0.0.302', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn20_10_1bit_cifar100', '1904', '149860c829a812224dbf2086c8ce95c2eba322fe', 'v0.0.302', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn20_10_1bit_svhn', '0273', 'ffe96cb78cd304d5207fff0cf08835ba2a71f666', 'v0.0.302', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn20_10_32bit_cifar10', '0314', 'a18146e8b0f99a900c588eb8995547393c2d9d9e', 'v0.0.302', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn20_10_32bit_cifar100', '1812', '70d8972c7455297bc21fdbe4fc040c2f6b3593a3', 'v0.0.302', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('wrn20_10_32bit_svhn', '0259', 'ce402a58887cbae3a38da1e845a1c1479a6d7213', 'v0.0.302', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_56_cifar10', '0543', '44f0f47d2e1b609880ee1b623014c52a9276e2ea', 'v0.0.228', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_56_cifar100', '2549', '34be6719cd128cfe60ba93ac6d250ac4c1acf0a5', 'v0.0.229', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_56_svhn', '0269', '5a9ad66c8747151be1d2fb9bc854ae382039bdb9', 'v0.0.287', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_110_cifar10', '0435', 'fb2a2b0499e4a4d92bdc1d6792bd5572256d5165', 'v0.0.235', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_110_cifar100', '2364', 'd599e3a93cd960c8bfc5d05c721cd48fece5fa6f', 'v0.0.236', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_110_svhn', '0257', '155380add8d351d2c12026d886a918f1fc3f9fd0', 'v0.0.287', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_164_cifar10', '0393', 'de7b6dc60ad6a297bd55ab65b6d7b1225b0ef6d1', 'v0.0.294', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_164_cifar100', '2234', 'd37483fccc7fc1a25ff90ef05ecf1b8eab3cc1c4', 'v0.0.294', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('ror3_164_svhn', '0273', 'ff0d9af0d40ef204393ecc904b01a11aa63acc01', 'v0.0.294', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('rir_cifar10', '0328', '414c3e6088ae1e83aa1a77c43e38f940c18a0ce2', 'v0.0.292', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('rir_cifar100', '1923', 'de8ec24a232b94be88f4208153441f66098a681c', 'v0.0.292', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('rir_svhn', '0268', '12fcbd3bfc6b4165e9b23f3339a1b751b4b8f681', 'v0.0.292', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('shakeshakeresnet20_2x16d_cifar10', '0515', 'ef71ec0d5ef928ef8654294114a013895abe3f9a', 'v0.0.215', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('shakeshakeresnet20_2x16d_cifar100', '2922', '4d07f14234b1c796b3c1dfb24d4a3220a1b6b293', 'v0.0.247', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('shakeshakeresnet20_2x16d_svhn', '0317', 'a693ec24fb8fe2c9f15bcc6b1050943c0c5d595a', 'v0.0.295', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('shakeshakeresnet26_2x32d_cifar10', '0317', 'ecd1f8337cc90b5378b4217fb2591f2ed0f02bdf', 'v0.0.217', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('shakeshakeresnet26_2x32d_cifar100', '1880', 'b47e371f60c9fed9eaac960568783fb6f83a362f', 'v0.0.222', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('shakeshakeresnet26_2x32d_svhn', '0262', 'c1b8099ece97e17ce85213e4ecc6e50a064050cf', 'v0.0.295', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet20_cifar10', '0622', '5e1a02bf2347d48651a5feabe97f7caf215bacc9', 'v0.0.340', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet20_cifar100', '2771', '28aa1a18d91334e274d3157114fc5c72e47c6c65', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet20_svhn', '0323', 'b8ee92c9d86de6a6adc80988518fe0544759ca4f', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet56_cifar10', '0505', '8ac8680448b2999bd1e03eed60373ea78eba9a44', 'v0.0.340', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet56_cifar100', '2435', '19085975afc7ee902a6d663eb371554c9519b467', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet56_svhn', '0268', 'bd2ec7558697aff1e0fd229d3e933a08c4c302e9', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet110_cifar10', '0410', '0c00a7daec69b57ab41d4a55e1026da33ecf4539', 'v0.0.340', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet110_cifar100', '2211', '7096ddb3a393ad28b27ece19263c203068a11b6d', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet110_svhn', '0247', '635e42cfac6ed67e15b8a5526c8232f768d11201', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet164bn_cifar10', '0350', 'd31f2ebce3acb419b07dc4d298018ffea2599fea', 'v0.0.340', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet164bn_cifar100', '1953', 'b1c474d27de3a291a45856a3e3d256b7fda90dd0', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diaresnet164bn_svhn', '0244', '0b8f67132b3911e6328733b666bf6a0fed133eeb', 'v0.0.342', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet20_cifar10', '0642', '14a1eb85c6346c81336b490cc49f2e6b809c193e', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet20_cifar100', '2837', 'f7675c09ca5f742376a102e3c8c5156aea4e24b9', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet20_svhn', '0303', 'dc3e3a453ffc8aff7d014bc15867db4ce2d8e1e9', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet56_cifar10', '0483', '41cae958be1bec3f839126cd167051de6a981d0a', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet56_cifar100', '2505', '5d357985236c021ab965101b94980cdc4722a70d', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet56_svhn', '0280', '537ebc66fe32f9bb6fb6bb8f9ac6402f8ec93e09', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet110_cifar10', '0425', '5638501600355b8b195179fb2be5d5989e93b0e0', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet110_cifar100', '2269', 'c993cc296c39bc9c8c0fc6115bfe6c7d720a0903', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet110_svhn', '0242', 'a156cfb58ffda89c0e87cd8aef82f56f79b40ea5', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet164bn_cifar10', '0356', '6ec898c89c66eb32b0e42b78a027af4920b24366', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet164bn_cifar100', '1999', '00872f989c33321f7938a40c0fd9f44669c4c483', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('diapreresnet164bn_svhn', '0256', '134048810bd2e12dc68035d4ecad6af525639db0', 'v0.0.343', '', '', 'cf', 0, 0.0, 0, ''),  # noqa
    ('resnet10_cub', '2777', '4525b5932665698b3f4551dde99d22ce03878172', 'v0.0.335', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('resnet12_cub', '2727', 'c15248832d2fe88c58fb603df3925e09b3d797e7', 'v0.0.336', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('resnet14_cub', '2477', '5051bbc659c0303c1860114f1a32a18942de9099', 'v0.0.337', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('resnet16_cub', '2365', 'b831356c696db80fec8deb2381875f37bf60dd93', 'v0.0.338', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('resnet18_cub', '2333', '200d8b9c48baf073a4c2ea0cbba4d7f81288e684', 'v0.0.344', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('resnet26_cub', '2316', '599ab467f396e979028f2ae5d65330949c9ddc86', 'v0.0.345', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('seresnet10_cub', '2772', 'f52526ec21bbb534a6d51be42bdb5322fbda919b', 'v0.0.361', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('seresnet12_cub', '2651', '5c0e7f835c65d1f2f85048d0169788377490b819', 'v0.0.361', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('seresnet14_cub', '2416', 'a4cda9012ec2380fa74f3d74879f0d206fcaf5b5', 'v0.0.361', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('seresnet16_cub', '2332', '43a819b7e226d65aa77a4c90fdb7c70eb5093505', 'v0.0.361', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('seresnet18_cub', '2352', '414fa2775de28ce3a1a0bc142ab674fa3a6638e3', 'v0.0.361', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('seresnet26_cub', '2299', '5aa0a7d1ef9c33f8dbf3ff1cb1a1a855627163f4', 'v0.0.361', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('mobilenet_w1_cub', '2377', '8428471f4ae08709b71ff2f69cf1a6fd286004c9', 'v0.0.346', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('proxylessnas_mobile_cub', '2266', 'e4b5098a17425c97740fc564460aa95d9eb2a41e', 'v0.0.347', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('ntsnet_cub', '1277', 'f6f330abfabcc2ea17a8d4b8977a6ea322ddf532', 'v0.0.334', '', '', 'cub', 0, 0.0, 0, ''),  # noqa
    ('pspnet_resnetd101b_voc', '8144', 'c22f021948461a7b7ab1ef1265a7948762770c83', 'v0.0.297', '', '', 'voc', 0, 0.0, 0, ''),  # noqa
    ('pspnet_resnetd50b_ade20k', '3687', '13f22137d7dd06c6de2ffc47e6ed33403d3dd2cf', 'v0.0.297', '', '', 'ade20k', 0, 0.0, 0, ''),  # noqa
    ('pspnet_resnetd101b_ade20k', '3797', '115d62bf66477221b83337208aefe0f2f0266da2', 'v0.0.297', '', '', 'ade20k', 0, 0.0, 0, ''),  # noqa
    ('pspnet_resnetd101b_cityscapes', '7172', '0a6efb497bd4fc763d27e2121211e06f72ada7ed', 'v0.0.297', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('pspnet_resnetd101b_coco', '6741', 'c8b13be65cb43402fce8bae945f6e0d0a3246b92', 'v0.0.297', '', '', 'cocoseg', 0, 0.0, 0, ''),  # noqa
    ('deeplabv3_resnetd101b_voc', '8024', 'fd8bf74ffc96c97b30bcd3b6ce194a2daed68098', 'v0.0.298', '', '', 'voc', 0, 0.0, 0, ''),  # noqa
    ('deeplabv3_resnetd152b_voc', '8120', 'f2dae198b3cdc41920ea04f674b665987c68d7dc', 'v0.0.298', '', '', 'voc', 0, 0.0, 0, ''),  # noqa
    ('deeplabv3_resnetd50b_ade20k', '3713', 'bddbb458e362e18f5812c2307b322840394314bc', 'v0.0.298', '', '', 'ade20k', 0, 0.0, 0, ''),  # noqa
    ('deeplabv3_resnetd101b_ade20k', '3784', '977446a5fb32b33f168f2240fb6b7ef9f561fc1e', 'v0.0.298', '', '', 'ade20k', 0, 0.0, 0, ''),  # noqa
    ('deeplabv3_resnetd101b_coco', '6773', 'e59c1d8f7ed5bcb83f927d2820580a2f4970e46f', 'v0.0.298', '', '', 'cocoseg', 0, 0.0, 0, ''),  # noqa
    ('deeplabv3_resnetd152b_coco', '6899', '7e946d7a63ed255dd38afacebb0a0525e735da64', 'v0.0.298', '', '', 'cocoseg', 0, 0.0, 0, ''),  # noqa
    ('fcn8sd_resnetd101b_voc', '8040', '66edc0b073f0dec66c18bb163c7d6de1ddbc32a3', 'v0.0.299', '', '', 'voc', 0, 0.0, 0, ''),  # noqa
    ('fcn8sd_resnetd50b_ade20k', '3339', 'e1dad8a15c2a1be1138bd3ec51ba1b100bb8d9c9', 'v0.0.299', '', '', 'ade20k', 0, 0.0, 0, ''),  # noqa
    ('fcn8sd_resnetd101b_ade20k', '3588', '30d05ca42392a164ea7c93a9cbd7f33911d3c1af', 'v0.0.299', '', '', 'ade20k', 0, 0.0, 0, ''),  # noqa
    ('fcn8sd_resnetd101b_coco', '6011', 'ebe2ad0bc1de5b4cecade61d17d269aa8bf6df7f', 'v0.0.299', '', '', 'coco', 0, 0.0, 0, ''),  # noqa
    ('icnet_resnetd50b_cityscapes', '6402', 'b380f8cc91ffeac29df6c245f34fbc89aa095c53', 'v0.0.457', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('fastscnn_cityscapes', '6576', 'b9859a25c6940383248bf2f53e2a5f02c1727cc8', 'v0.0.474', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('sinet_cityscapes', '6172', '8ecd14141b85a682c2cc1c74e13077fee4746d87', 'v0.0.437', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('bisenet_resnet18_celebamaskhq', '0000', '98affefd74cc7f87314a96f148dbdbf4055bbfcb', 'v0.0.462', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('danet_resnetd50b_cityscapes', '6799', 'c5740c9fd471c141a584455efd2167858dd8cb94', 'v0.0.468', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('danet_resnetd101b_cityscapes', '6810', 'f1eeb724757bbcdc067de9cdfad6d463fb9fdb90', 'v0.0.468', '', '', 'cs', 0, 0.0, 0, ''),  # noqa
    ('alphapose_fastseresnet101b_coco', '7415', 'b9e3f64a9fe44198b23e7278cc3a94fd94247e20', 'v0.0.454', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resnet18_coco', '6631', '7c3656b35607805bdb877e7134938fd4510b2c8c', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resnet50b_coco', '7102', '621d2545c8b39793a0fe3a48054684f8b982a978', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resnet101b_coco', '7244', '540c29ec1794535fe9ee319cdb5527ed3a6d3eb5', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resnet152b_coco', '7253', '3a358d7de566d51e90b9d3a1f44a1c9c948769ed', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resneta50b_coco', '7170', '2d973dc512d02f24d0de5a98008898c0a03a2c99', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resneta101b_coco', '7297', '08175610ce24a4e476b49030c1c1378d74158f70', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_resneta152b_coco', '7344', 'dacb65cfe1261e5f2013cde18f2d5753c6453568', 'v0.0.455', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_mobile_resnet18_coco', '6625', '1e27b206737a33678b67b638bba8a4d024ec2dc3', 'v0.0.456', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_mobile_resnet50b_coco', '7110', '023f910cab8c0750bb24e6a14aecdeb42fcc5561', 'v0.0.456', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_mobile_mobilenet_w1_coco', '6410', '0ca46de0f31cb3d700ce1310f2eba19a3308a3f0', 'v0.0.456', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_mobile_mobilenetv2b_w1_coco', '6374', '94f86097959d1acca6605d0d6487fd2d0899dfeb', 'v0.0.456', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_mobile_mobilenetv3_small_w1_coco', '5434', '5cedb749e09a30c779073fba0e71546ad8b022d5', 'v0.0.456', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('simplepose_mobile_mobilenetv3_large_w1_coco', '6367', '9515de071e264aa95514b9b85ab60a5da23f5f69', 'v0.0.456', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('lwopenpose2d_mobilenet_cmupan_coco', '3999', 'a6b9c66bb43e7819464f1ce23c6e3433b726b95d', 'v0.0.458', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('lwopenpose3d_mobilenet_cmupan_coco', '3999', '4c727e1dece57dede247da2d7b97d647c0d51b0a', 'v0.0.458', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
    ('ibppose_coco', '6487', '1958fe10a02a1c441e40d109d3281845488e1e2f', 'v0.0.459', '', '', 'cocohpe', 0, 0.0, 0, ''),  # noqa
]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag, _, _, _, _, _, _, _ = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters:
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.

    Returns:
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.pth".format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info("Model file not found. Downloading to {}.".format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(
        url="{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip".format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """
    Download an given URL

    Parameters:
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns:
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. " \
            "Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


def _check_sha1(file_name, sha1_hash):
    """
    Check whether the sha1 hash of the file content matches the expected hash.

    Parameters:
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns:
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(file_name, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def load_model(net,
               file_path,
               ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters:
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


def download_model(net,
                   model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models"),
                   ignore_extra=True):
    """
    Load model state dictionary from a file with downloading it if necessary.

    Parameters:
    ----------
    net : Module
        Network in which weights are loaded.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    load_model(
        net=net,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path),
        ignore_extra=ignore_extra)


def calc_num_params(net):
    """
    Calculate the count of trainable parameters for a model.

    Parameters:
    ----------
    net : Module
        Analyzed model.
    """
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count
