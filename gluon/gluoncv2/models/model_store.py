"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
from mxnet.gluon.utils import download, check_sha1

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '2126', '9cb87ebd09523bec00e10d8ba9abb81a2c632e8b', 'v0.0.108'),
    ('vgg11', '1176', '95dd287d0eafa05f8c25a780e41c8760acdb7806', 'v0.0.109'),
    ('vgg13', '1112', 'a0db3c6c854c675e8c83040c35a80da6e5cdf15f', 'v0.0.109'),
    ('vgg16', '0869', '57a2556f64a7f0851f9764e9305126074334ef2d', 'v0.0.109'),
    ('vgg19', '0823', '0e2a1e0a9fdeb74dfef9aedd37712ad306627e35', 'v0.0.109'),
    ('bn_vgg11b', '1057', 'b2d8f382879075193ee128bc7997611462cfda33', 'v0.0.110'),
    ('bn_vgg13b', '1016', 'f384ff5263d4c79c22b8fc1a2bdc19c31e1b12b9', 'v0.0.110'),
    ('bn_vgg16b', '0865', 'b5e33db8aaa77e0a1336e5eb218345a2586f5469', 'v0.0.110'),
    ('bn_vgg19b', '0815', '3a0e43e66836ea5ab4f6d4c0425e2ab2abcb5766', 'v0.0.110'),
    ('bninception', '0776', '8314001b410c26120a9cf9e1d84a3770ba31b128', 'v0.0.139'),
    ('resnet10', '1385', 'a996427482bd5bd427db1d811d2a69bfd35987fc', 'v0.0.248'),
    ('resnet12', '1303', '8f492c28cd7c4c1bfc89c88649629e64487d5d00', 'v0.0.253'),
    ('resnet14', '1241', 'a8955ff3d4facd8e38e776fc72f1945806def16a', 'v0.0.40'),
    ('resnet16', '1110', '1be996d1d18da94d9e8dc0cd35d5135ef4da6bc9', 'v0.0.41'),
    # ('resnet18_wd4', '2445', '28d15cf486f1e159d9f257ae8f6951f135fa6ccd', 'v0.0.47'),
    # ('resnet18_wd2', '1496', 'd839c509bcb8c9c156478f2b91f23e96fc17df15', 'v0.0.46'),
    # ('resnet18_w3d4', '1254', 'd654861258936f678b928a92d88fd205a20777af', 'v0.0.18'),
    ('resnet18', '0951', '98a2545bf310e974848d96dee8da9095db623fbc', 'v0.0.153'),
    ('resnet34', '0792', '5b875f4934da8d83d44afc30d8e91362d3103115', 'v0.0.1'),
    ('resnet50', '0641', '1eaa883b422b797657160ac4efa3390271657243', 'v0.0.147'),
    ('resnet50b', '0618', '8e2541fbb61a27a6464ac884a24ce43d69c2b329', 'v0.0.146'),
    ('resnet101', '0599', 'a6d3a5f4933794d56b61867c050ee730f6310f1b', 'v0.0.1'),
    ('resnet101b', '0539', '7406d85891b64717ae0a823d147e9e9e8921e868', 'v0.0.145'),
    ('resnet152', '0535', 'bbdd7ed1f33a9b33c75635d78143e8bd00e204e0', 'v0.0.144'),
    ('resnet152b', '0525', '6f30d0d99e1765e78370c92cd400f50eeb59b6f9', 'v0.0.143'),
    ('preresnet10', '1401', '2b96c0818dbabc422e98d8fbfc9b684c023922ed', 'v0.0.249'),
    ('preresnet18', '0951', '71279a0b7339f1efd12bed737219a9ed76175a9d', 'v0.0.140'),
    ('preresnet34', '0811', 'f8fe98a25337d747b8687ffdbd1c83ce0d9b9a34', 'v0.0.2'),
    ('preresnet50', '0668', '4940c94b02cf25d015a76a0b09498433729c37b8', 'v0.0.2'),
    ('preresnet50b', '0664', '2fcfddb13fbb8d7f58fb949f137610bf3d99a892', 'v0.0.2'),
    ('preresnet101', '0575', 'e2887e539f2519c36aea0fc991d6503ed384c4fc', 'v0.0.2'),
    ('preresnet101b', '0588', '1015145a6228aa16583a975b9c33f879ee2a6fc0', 'v0.0.2'),
    ('preresnet152', '0532', '31505f719ad76f5aee59d37a695ac7a9b06230fc', 'v0.0.14'),
    ('preresnet152b', '0575', 'dc303191ea47ca258f5abadd203b5de24d059d1a', 'v0.0.2'),
    ('preresnet200b', '0564', '38f849a61f59924d85a9353923424889a77c93dc', 'v0.0.45'),
    ('preresnet269b', '0556', 'f386e3e70ab4df48fd8b444858bd6acd8afcbe6b', 'v0.0.239'),
    ('resnext101_32x4d', '0579', '9afbfdbc5a420a9f56058be0bf80d12b21a627af', 'v0.0.10'),
    ('resnext101_64x4d', '0541', '0d4fd87b8de78c5c0295e1dcb9923a578dce7adb', 'v0.0.10'),
    ('seresnet50', '0644', '10954a846a56a387a6a222e260d95fb8a9bd68c3', 'v0.0.11'),
    ('seresnet101', '0589', '4c10238dd485a540a464bf1c39a8752d2da040b9', 'v0.0.11'),
    ('seresnet152', '0577', 'de6f099dd39f374390639ca8854b2954af3c59b9', 'v0.0.11'),
    ('seresnext50_32x4d', '0558', 'a49f8fb039973979afe2fc70974a8b07c7159bca', 'v0.0.12'),
    ('seresnext101_32x4d', '0500', 'cf1612601f319a0e75190ae756ae380b947dcb1a', 'v0.0.12'),
    ('senet154', '0465', 'dd2445078c0770c4a52cd22aa1d4077eb26f6132', 'v0.0.13'),
    ('ibn_resnet50', '0668', 'db527596f81f5b4aa1f0c490bf0ef5cfeef5fb76', 'v0.0.127'),
    ('ibn_resnet101', '0587', '946e7f1072a70b19f2bbc9776f73b818473482c3', 'v0.0.127'),
    ('ibnb_resnet50', '0697', '0aea51d29d4123676e447b92db800f5a574a35be', 'v0.0.127'),
    ('ibn_resnext101_32x4d', '0562', '05ddba79597927b5c0fa516d435c3788803438f6', 'v0.0.127'),
    ('ibn_densenet121', '0747', '1434d379777ff6b61469f7adc6ed73919da94f02', 'v0.0.127'),
    ('ibn_densenet169', '0682', '6d7c48c5519c6b8595223514564b1061268742a2', 'v0.0.127'),
    ('airnet50_1x64d_r2', '0621', '347358cc4a3ac727784665e8113cd11bfa79c606', 'v0.0.120'),
    ('airnet50_1x64d_r16', '0646', '0b847b998253ba22409eed4b939ec2158928a33f', 'v0.0.120'),
    ('airnext50_32x4d_r2', '0575', 'ab104fb5225b17836d523a525903db254f5fdd99', 'v0.0.120'),
    ('bam_resnet50', '0696', '7e573b617562d7dab94cda3b1a47ec0085aaeba2', 'v0.0.124'),
    ('cbam_resnet50', '0638', '78be56658e9f9452d7c2472c994b332d97807a17', 'v0.0.125'),
    ('pyramidnet101_a360', '0652', '08d5a5d1af3d514d1114ce76277223e8c1f5f426', 'v0.0.104'),
    ('diracnet18v2', '1117', '27601f6fa54e3b10d77981f30650d7a9d4bce91e', 'v0.0.111'),
    ('diracnet34v2', '0946', '1faa6f1245e152d1a3e12de4b5dc1ba554bc3bb8', 'v0.0.111'),
    ('crunet56', '0825', 'ad16523bfa306aefae5f931ed3bd6d01cd6d1804', 'v0.0.197'),
    ('densenet121', '0780', '49b72d04bace00bb1964b38cec13d19059a14e86', 'v0.0.3'),
    ('densenet161', '0618', '52e30516e566bdef53dcb417f86849530c83d0d1', 'v0.0.3'),
    ('densenet169', '0689', '281ec06b02f407b4523245622371da669a287044', 'v0.0.3'),
    ('densenet201', '0636', '65b5d389b1f2a18c62dc39f74960266c601fec76', 'v0.0.3'),
    ('condensenet74_c4_g4', '0864', 'cde68fa2fcc9197e336717a17753a15a6efd7596', 'v0.0.4'),
    ('condensenet74_c8_g8', '1049', '4cf4a08e7fb46f5821049dcae97ae442b0ceb546', 'v0.0.4'),
    ('peleenet', '1125', '38d4fb245659a54204ca8f3562069b786eace1b1', 'v0.0.141'),
    ('wrn50_2', '0612', 'f8013e680bf802301e6830e5ca12de73382edfb1', 'v0.0.113'),
    ('drnc26', '0789', 'ee56ffabbcceba2e4063c80a3f84a4f4f8461bff', 'v0.0.116'),
    ('drnc42', '0692', 'f89c26d6a3792bef0850b7fe09ee10f715dcd3ce', 'v0.0.116'),
    ('drnc58', '0627', '44cbf15ccaea33ee1e91b780e70170e8e66b12d7', 'v0.0.116'),
    ('drnd22', '0852', '085747529f2d4a0490769e753649843c40dea410', 'v0.0.116'),
    ('drnd38', '0736', 'c7d53bc0f70196dda589fcf0bfac904b5d76d872', 'v0.0.116'),
    ('drnd54', '0627', '87d44c87953d98241f85007802a61e3cefd77792', 'v0.0.116'),
    ('drnd105', '0581', 'ab12d66220c1bbf4af5c33db78aaafc9f0d9bd5a', 'v0.0.116'),
    ('dpn68', '0700', '3114719dccf3d9fa30bb7ab5a8c845815328e495', 'v0.0.17'),
    ('dpn98', '0528', 'fa5d6fca985afde21f6374e4a4d4df788d1b4c3a', 'v0.0.17'),
    ('dpn131', '0522', '35ac2f82e69264e0712dcb979da4d99675e2f2aa', 'v0.0.17'),
    ('darknet_tiny', '1746', '16501793621fbcb137f2dfb901760c1f621fa5ec', 'v0.0.69'),
    ('darknet_ref', '1668', '3011b4e14b629f80da54ab57bef305d588f748ab', 'v0.0.64'),
    ('darknet53', '0556', 'e9486353868e0cf78bdc8fa377437de4d02733bb', 'v0.0.150'),
    ('irevnet301', '0897', 'cef9b5bfe9dd51c7d7946de432fd358f54239d35', 'v0.0.251'),
    ('dla34', '0821', '1127fa0a270ed5fe1112d9039cf44ee06ac18817', 'v0.0.202'),
    ('dla46c', '1470', 'bae8b513dd17f6645379e3815ec4ed06467b1b1e', 'v0.0.202'),
    ('dla46xc', '1398', '28deb1fcb3db9cbd68d7001851ebfb8f9cbeb673', 'v0.0.202'),
    ('dla60', '0708', '954571d6917a836cb3433b664fd1b8330955c0fa', 'v0.0.202'),
    ('dla60x', '0621', '35774214758743caeae249482bcf8a0ab8caf456', 'v0.0.202'),
    ('dla60xc', '1241', '338c62411ce457126614f497e610d59b22705cd6', 'v0.0.202'),
    ('dla102', '0644', 'cadbb1cc5feb58497198aac2c8028c843a6d5a9f', 'v0.0.202'),
    ('dla102x', '0602', '193568a7ab3c0a3b6702fcc3e326b7fef011f752', 'v0.0.202'),
    ('dla102x2', '0553', '30c8f409240872053c5bb8523baeb274ba9da4ee', 'v0.0.202'),
    ('dla169', '0587', '4f3e6a6e604cc06ebcf63fc86cc5287399f32683', 'v0.0.202'),
    ('fishnet150', '0638', '5cbd08ec1534f2d5e7861395cc6f224ecbe8cb76', 'v0.0.168'),
    ('espnetv2_wd2', '2107', 'f2e17f0a8845b72f4645a0c06f2cfc8d9ef0253e', 'v0.0.238'),
    ('espnetv2_w1', '1427', '538f31fb92699fddabc27784fe746871fd8a635b', 'v0.0.238'),
    ('espnetv2_w5d4', '1273', 'b119ad9e52bf8c2d88db8eeb62a1fd6e23a1b6a6', 'v0.0.238'),
    ('espnetv2_w3d2', '1194', '3804a85006d874273b80026339b36943e9d813e5', 'v0.0.238'),
    ('espnetv2_w2', '0994', 'c212d81a9d12361b7d49ec841599f6a0f84f7b44', 'v0.0.238'),
    ('squeezenet_v1_0', '1734', 'e6f8b0e8253cef1c5c071dfaf2df5fdfc6a64f8c', 'v0.0.128'),
    ('squeezenet_v1_1', '1739', 'd7a1483aaa1053c7cd0cf08529b2b87ed2781b35', 'v0.0.88'),
    ('squeezeresnet_v1_0', '1767', '66474b9b6a771055b28c37b70621c026a1ef6ef4', 'v0.0.178'),
    ('squeezeresnet_v1_1', '1784', '26064b82773e7a7175d6038976a73abfcd5ed2be', 'v0.0.70'),
    ('sqnxt23_w1', '1866', '73b700c40de5f7be9d2cf4ed30cc8935c670a3c3', 'v0.0.171'),
    ('sqnxt23v5_w1', '1743', '7a83722e7d362cef950d8534020f837caf9e6314', 'v0.0.172'),
    ('sqnxt23_w3d2', '1321', '4d733bcd19f1e502ebc46b52f0b69d959636902e', 'v0.0.210'),
    ('sqnxt23v5_w3d2', '1268', '4f98bbd3841d8d09a067100841f64ce3eccf184a', 'v0.0.212'),
    ('sqnxt23_w2', '1063', '95d9b55a5e857298bdb7974db6e3dbd9ecc94401', 'v0.0.240'),
    ('sqnxt23v5_w2', '1024', '707246f323bc95d0ea2d5608e9e85ae9fe59773a', 'v0.0.216'),
    ('shufflenet_g1_wd4', '3677', 'ee58f36811d023e1b2e651469c470e588c93f9d3', 'v0.0.134'),
    ('shufflenet_g3_wd4', '3617', 'bd08e3ed6aff4993cf5363fe8acaf0b22394bea0', 'v0.0.135'),
    ('shufflenet_g1_wd2', '2238', 'f77dcd18d3b759a3046bd4a2443c40e4ff455313', 'v0.0.174'),
    ('shufflenet_g3_wd2', '2060', 'ea6737a54bce651a0e8c0b533b982799842cb1c8', 'v0.0.167'),
    ('shufflenet_g1_w3d4', '1675', '2f1530aa72ee04e3599c5296b590a835d9d50e7f', 'v0.0.218'),
    ('shufflenet_g3_w3d4', '1609', 'e008e926f370af28e587f349384238d240a0fc02', 'v0.0.219'),
    ('shufflenet_g1_w1', '1350', '01934ee8f4bf7eaf4e36dd6442debb84ca2a2849', 'v0.0.223'),
    ('shufflenet_g2_w1', '1332', 'f5a1479fd8523032ee17a4de00fefd33ff4d31e6', 'v0.0.241'),
    ('shufflenet_g3_w1', '1329', 'ac58d62c5f277c0e9e5a119cc1f48cb1fcfc8306', 'v0.0.244'),
    ('shufflenet_g4_w1', '1310', '73c039ebf56f9561dd6eecc4cbad1ab1db168ed1', 'v0.0.245'),
    ('shufflenet_g8_w1', '1319', '9a50ddd9ce67ec697e3ed085d6c39e3d265f5719', 'v0.0.250'),
    ('shufflenetv2_wd2', '1830', '156953de22d0e749c987da4a58e0e53a5fb18291', 'v0.0.90'),
    ('shufflenetv2_w1', '1123', '27435039ab7794c86ceab11bd93a19a5ecab78d2', 'v0.0.133'),
    ('shufflenetv2_w3d2', '1237', '08c013888fc1f782683eccdb201739b05dfd43aa', 'v0.0.65'),
    ('shufflenetv2_w2', '1210', '544b55d98384504305e9c9fe9458eb5e9dd53dbd', 'v0.0.84'),
    ('shufflenetv2b_wd2', '1782', '845a9c43cf4a9873f89c6116634e74329b977e64', 'v0.0.157'),
    ('shufflenetv2b_w1', '1101', 'f679702f7c626161413320160c6c9c199de9b667', 'v0.0.161'),
    ('shufflenetv2b_w3d2', '0879', '4022da3a5922127b1acf5327bd9f1d4d55726e05', 'v0.0.203'),
    ('shufflenetv2b_w2', '0810', '7429df751916bf24bd7fb86bc137ae36275b9d19', 'v0.0.242'),
    ('menet108_8x1_g3', '2030', 'aa07f925180834389cfd3bf50cb22d2501225118', 'v0.0.89'),
    ('menet128_8x1_g4', '1913', '0c890a76fb23c0af50fdec076cb16d0f0ee70355', 'v0.0.103'),
    ('menet160_8x1_g8', '2028', '4f28279a94e631f6a51735de5ea29703cca69845', 'v0.0.154'),
    ('menet228_12x1_g3', '1289', '2dc2eec7c9ebb41c459450e1843503b5ac7ecb3a', 'v0.0.131'),
    ('menet256_12x1_g4', '1216', '7caf63d15190648e266a4e7520c3ad677716f388', 'v0.0.152'),
    ('menet348_12x1_g3', '0936', '62c72b0b56460f062d4da7155bd64a524f42fb88', 'v0.0.173'),
    ('menet352_12x1_g8', '1167', '5892fea4e44eb27814a9b092a1a06eb81cea7844', 'v0.0.198'),
    ('menet456_24x1_g3', '0780', '7a89b32c89f878ac63fc96ddc71cb1a5e91c84d6', 'v0.0.237'),
    ('mobilenet_wd4', '2218', '3185cdd29b3b964ad51fdd7820bd65f091cf281f', 'v0.0.62'),
    ('mobilenet_wd2', '1330', '94f13ae1375b48892d8ecbb4a253bb583fe27277', 'v0.0.156'),
    ('mobilenet_w3d4', '1051', '6361d4b4192b5fc68f3409100d825e8edb28876b', 'v0.0.130'),
    ('mobilenet_w1', '0865', 'eafd91e9369abb09726f2168aba24453b17fc22e', 'v0.0.155'),
    ('fdmobilenet_wd4', '3053', 'd4f18e5b4ed63e5426eafbf5db7f8e2a97c28581', 'v0.0.177'),
    ('fdmobilenet_wd2', '1969', '242b9fa82d54f54f08b4bdbb194b7c89030e7bc4', 'v0.0.83'),
    ('fdmobilenet_w3d4', '1601', 'cb10c3e129706d3023d752e7402965af08f91ca7', 'v0.0.159'),
    ('fdmobilenet_w1', '1312', '95fa0092aac013c88243771faf66ef1134b7574d', 'v0.0.162'),
    ('mobilenetv2_wd4', '2412', 'd92b5b2dbb52e27354ddd673e6fd240a0cf27175', 'v0.0.137'),
    ('mobilenetv2_wd2', '1442', 'd7c586c716e3ea85e793f7c5aaf9cae2a907117b', 'v0.0.170'),
    ('mobilenetv2_w3d4', '1044', '768454f4bdaae337c180bb81248b8c5b8d31040b', 'v0.0.230'),
    ('mobilenetv2_w1', '0864', '6e58b1cb96852e4c6de6fc9cd11241384af21df9', 'v0.0.213'),
    ('igcv3_wd4', '2830', '71abf6e0b6bff1d3a3938bfea7c752b59ac05e9d', 'v0.0.142'),
    ('igcv3_wd2', '1703', '145b7089e1d0e0ce88f17393a357d5bb4ae37734', 'v0.0.132'),
    ('igcv3_w3d4', '1096', '3c7c86fc43df2e5cf95a451ebe07fccf2d9dc076', 'v0.0.207'),
    ('igcv3_w1', '0900', 'e2c3da1cffd8e42da7a052b80db2f86758c8d35b', 'v0.0.243'),
    ('mnasnet', '1144', 'c972fec0521e0222259934bf77c57ebeebff5bdf', 'v0.0.117'),
    ('darts', '0897', 'aafd645210df6b55587ef02f4edf08c76a15e5a3', 'v0.0.118'),
    ('xception', '0556', 'bd2c1684a5dc41dd00b4676c194a967558ed577e', 'v0.0.115'),
    ('inceptionv3', '0559', '6c087967685135a321ed66b9ad2277512e9b2868', 'v0.0.92'),
    ('inceptionv4', '0525', 'f7aa9536392ea9ec7df5cc8771ff53c19c45fff2', 'v0.0.105'),
    ('inceptionresnetv2', '0494', '3328f7fa4c50c785b525e7b603926ec1fccbce14', 'v0.0.107'),
    ('polynet', '0453', '742803144e5a2a6148212570726350da09adf3f6', 'v0.0.96'),
    ('nasnet_4a1056', '0795', '5c78908e38c531283d86f9cbe7e14c2afd85a7ce', 'v0.0.97'),
    ('nasnet_6a4032', '0424', '73cca5fee009db77412c5fca7c826b3563752757', 'v0.0.101'),
    ('pnasnet5large', '0428', '998a548f44ac1b1ac6c4959a721f2675ab5c48b9', 'v0.0.114'),
    ('nin_cifar10', '0743', '9696dc1a8f67e7aa233836bcbdb99625769b1e86', 'v0.0.175'),
    ('nin_cifar100', '2839', 'eed0e9af2cd8e5aa77bb063204525812dbd9190f', 'v0.0.183'),
    ('resnet20_cifar10', '0597', '13c5ab19145591d75873da3497be1dd1bd2afd46', 'v0.0.163'),
    ('resnet20_cifar100', '2964', '4e1443526ee96648bfe4d4954871a97a9c9622f4', 'v0.0.180'),
    ('resnet56_cifar10', '0452', 'a73e63e9d0f3f7adde59b4142323c0dd05930de7', 'v0.0.163'),
    ('resnet56_cifar100', '2488', '590977100774a289b91088245dd2bd0cbe6567e6', 'v0.0.181'),
    ('resnet110_cifar10', '0369', 'f89f1c4d9fdd9e5cd00949a872211376979ff703', 'v0.0.163'),
    ('resnet110_cifar100', '2280', '6c5fa14bb4ced2dffe6ee1536306687aae57f9cb', 'v0.0.190'),
    ('resnet164bn_cifar10', '0368', 'e7941eeeddef9336664522eaa3af92d77128cac0', 'v0.0.179'),
    ('resnet164bn_cifar100', '2044', 'c7db7b5e6fbe6dc0f9501d25784f1a107c6e0315', 'v0.0.182'),
    ('resnet1001_cifar10', '0328', 'bb979d53089138b5060b418cad6c8ad9a940bf81', 'v0.0.201'),
    ('resnet1001_cifar100', '1979', '692d9516620bc8b7a4da30a98ebcb7432243f5e9', 'v0.0.254'),
    ('resnet1202_cifar10', '0353', '377510a63595e544333f6f57523222cd845744a8', 'v0.0.214'),
    ('preresnet20_cifar10', '0651', 'daa895737a34edda75c40f2d8566660590c84a3f', 'v0.0.164'),
    ('preresnet20_cifar100', '3022', '37f15365d48768f792f4551bd6ccf5259bc70530', 'v0.0.187'),
    ('preresnet56_cifar10', '0449', 'cb37cb9d4524d4e0f5724aeed9face455f527efc', 'v0.0.164'),
    ('preresnet56_cifar100', '2505', '4c39e83f567f15d6ee0d69bf2dcaccd62067dfe5', 'v0.0.188'),
    ('preresnet110_cifar10', '0386', 'd6d4b7bd9f154eca242482a7559413d5c7b6d465', 'v0.0.164'),
    ('preresnet110_cifar100', '2267', '18cf4161c67c03e50cff7eb30988a559f3f97260', 'v0.0.191'),
    ('preresnet164bn_cifar10', '0364', '7ecf30cb818f80908ef4a77af4660c1080d0df81', 'v0.0.196'),
    ('preresnet164bn_cifar100', '2018', 'a20557c8968c04d8d07e40fdc5b0d1ec1fb3339d', 'v0.0.192'),
    ('preresnet1001_cifar10', '0265', '50507ff74b6047abe6d04af6471d9bacafa05e24', 'v0.0.209'),
    ('preresnet1202_cifar10', '0339', '942cf6f22d80b5428256825234a252b8d6ebbe9d', 'v0.0.246'),
    ('resnext29_32x4d_cifar10', '0315', 'c8a1beda8ba616dc9af682d3ac172bfdd7a2472d', 'v0.0.169'),
    ('resnext29_32x4d_cifar100', '1950', '5f2eedcdd5cea6fdec1508f261f556a953ae28c2', 'v0.0.200'),
    ('resnext29_16x64d_cifar10', '0241', '76b97a4dd6185602a8ca8bdd77a70f8ddfcd4e83', 'v0.0.176'),
    ('pyramidnet110_a48_cifar10', '0372', '35b94d0575c2081a142e71955c8ceea8c51ec5e5', 'v0.0.184'),
    ('pyramidnet110_a48_cifar100', '2095', '00fd42a00492b2bbb28cacfb7b1a6c63072c37a3', 'v0.0.186'),
    ('pyramidnet110_a84_cifar10', '0298', '81710d7ab90838a8a299bf5f50aed2a3fa41f0e3', 'v0.0.185'),
    ('pyramidnet110_a84_cifar100', '1887', '6712d5dc69452f2fde1fcc3ee32c3164dcaffc4e', 'v0.0.199'),
    ('pyramidnet110_a270_cifar10', '0251', '1e769ce50ef915a807ee99907912c87766fff60f', 'v0.0.194'),
    ('densenet40_k12_cifar10', '0561', '28dc0035549e51dcb53d1360707bd6f1558a5dcd', 'v0.0.193'),
    ('densenet40_k12_cifar100', '2490', '908f02ba7dbd7b8138f264193189e762a5590b1c', 'v0.0.195'),
    ('densenet40_k12_bc_cifar10', '0643', '7fdeda31c5accbddf47ab0f0b9a32cff723bf70d', 'v0.0.231'),
    ('densenet40_k12_bc_cifar100', '2841', '35cd8e6a2ae0896a8af2b689e076057fa19efa9b', 'v0.0.232'),
    ('densenet40_k24_bc_cifar10', '0452', '13fa807e095b44ecaf3882e488b33a890d9d1e29', 'v0.0.220'),
    ('densenet40_k24_bc_cifar100', '2267', '2c4ef7c4bbe7f64784ad18b3845f4bf533f2ce57', 'v0.0.221'),
    ('densenet40_k36_bc_cifar10', '0404', '4c154567e25619994a2f86371afbf1ad1e7475e9', 'v0.0.224'),
    ('densenet40_k36_bc_cifar100', '2050', 'd7275d39bcf439151c3bbeb707efa54943714b03', 'v0.0.225'),
    ('densenet100_k12_cifar10', '0366', '4e371ccb315d0fcd727a76255ca62ae9e92059cc', 'v0.0.205'),
    ('densenet100_k12_cifar100', '1964', '2ed5ec27a4d4a63876a4cacf52be53c91fbecb5f', 'v0.0.206'),
    ('densenet100_k24_cifar10', '0313', '9f795bac946d1390cf59f686b730fe512c406bd2', 'v0.0.252'),
    ('densenet100_k12_bc_cifar10', '0416', '6685d1f4844b092471f7d03dfc3fa64a302008e6', 'v0.0.189'),
    ('densenet100_k12_bc_cifar100', '2119', 'fbd8a54c1c9e4614f950b8473f8524d25caba4a7', 'v0.0.208'),
    ('xdensenet40_2_k24_bc_cifar10', '0531', '66c9d384d3ef4ec4095c9759bb8b7986f2f58e26', 'v0.0.226'),
    ('xdensenet40_2_k24_bc_cifar100', '2396', '73d5ba88a39b971457b9cea2cd72d1e05ab4d165', 'v0.0.227'),
    ('xdensenet40_2_k36_bc_cifar10', '0437', 'e9bf419295f833b56fa3da27218107ed42310307', 'v0.0.233'),
    ('xdensenet40_2_k36_bc_cifar100', '2165', '78b6e754d90774d7b6ec3d811e6e57192148cfbf', 'v0.0.234'),
    ('wrn16_10_cifar10', '0293', 'ecf1c17c0814763095df562cb27d15a5aeb51836', 'v0.0.166'),
    ('wrn16_10_cifar100', '1895', 'bcb5c89ca71ffc99bc09b861b339724047724659', 'v0.0.204'),
    ('wrn28_10_cifar10', '0239', '16f3c8a249993f23b0f81d9ce3650faef5e455d8', 'v0.0.166'),
    ('wrn40_8_cifar10', '0237', '3b81d261706b751f5b731149b05fa92f500218e8', 'v0.0.166'),
    ('ror3_56_cifar10', '0543', 'ee31a69a0503b41878c49d8925ac8e7ee813293b', 'v0.0.228'),
    ('ror3_56_cifar100', '2549', '4334559313cd9291af3d6ec0df144b21e695228b', 'v0.0.229'),
    ('ror3_110_cifar10', '0435', '0359916596cba01dfa481f105094c1047f592980', 'v0.0.235'),
    ('ror3_110_cifar100', '2364', 'b8c4d317241f54990180443d7fd9702d79c57ccc', 'v0.0.236'),
    ('shakeshakeresnet20_2x16d_cifar10', '0515', 'a7b8a2f77457e151da5d5ad3b9a2473594fecfc0', 'v0.0.215'),
    ('shakeshakeresnet20_2x16d_cifar100', '2922', 'e46e31a7d8308b57d9c0687000c40f15623998c2', 'v0.0.247'),
    ('shakeshakeresnet26_2x32d_cifar10', '0317', '21e60e626765001aaaf4eb26f7cb8f4a69ea3dc1', 'v0.0.217'),
    ('shakeshakeresnet26_2x32d_cifar100', '1880', 'bd46a7418374e3b3c844b33e12b09b6a98eb4e6e', 'v0.0.222')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.mxnet', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $MXNET_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.params'.format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to {}.'.format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + '.zip'
    download(
        url='{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip'.format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')
