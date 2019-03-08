"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '2132', 'cea565f1d8254d6dc3fdbc87568e90c34455a477', 'v0.0.108'),
    ('vgg11', '1179', '3cc057e61154ddbd152e138c31327ebd986d2b2f', 'v0.0.109'),
    ('vgg13', '1116', 'e835ca5af6ad9b9d65ffa4f19ccc544907ee4e13', 'v0.0.109'),
    ('vgg16', '0870', '8741ff5c98cd3e17bdc00a557b010c849e923b3c', 'v0.0.109'),
    ('vgg19', '0823', '18980884d7b7e46d0f564548e09af8ea8313789d', 'v0.0.109'),
    ('bn_vgg11b', '1060', '8964402b8870b2b2463b01e9ba9425737678c258', 'v0.0.110'),
    ('bn_vgg13b', '1019', '0121b0a47782b5b58c02baa148c88cdc848fc642', 'v0.0.110'),
    ('bn_vgg16b', '0863', 'cbaa2105e000ae844b4775390e9be3b30a23e02e', 'v0.0.110'),
    ('bn_vgg19b', '0816', 'dc5e37a5f6a1d5068b18011ad779062d7b4842cd', 'v0.0.110'),
    ('bninception', '0778', '99f685c2a38743e719eb0ed2ac99ee93f8898926', 'v0.0.139'),
    ('resnet10', '1389', '66bddf8086630a36f64445dd8fefd895bd9e7189', 'v0.0.248'),
    ('resnet12', '1302', '0cc61e0d7f45d0a58a649850f44fcd0b913ada2a', 'v0.0.253'),
    ('resnet14', '1242', '4e65746b8a327f2fde5740669f5cd44dc7327e24', 'v0.0.40'),
    ('resnet16', '1107', 'b1d7fb7df91145155f6b1c45133c47ecb26996e9', 'v0.0.41'),
    ('resnet18_wd4', '2448', '58c4a0075a3a240d060a625cefe6e53bf8d28865', 'v0.0.47'),
    ('resnet18_wd2', '1499', '542ed773551add89346117be2430c9f818faeeb1', 'v0.0.46'),
    ('resnet18_w3d4', '1256', 'ce2011dfcddf9cac229d7e3a63b3764e15bcbc47', 'v0.0.22'),
    ('resnet18', '0959', 'd80fbe604f4d4580f72a523e0577e75b2d4d0661', 'v0.0.153'),
    ('resnet34', '0795', '0b392267b08907dc14023b24fd84df0268087002', 'v0.0.22'),
    ('resnet50', '0641', 'ca0cd7a1b40ea232bd20fd5312ca48b81c80b27e', 'v0.0.147'),
    ('resnet50b', '0618', '42fffef9ab51267b3c469ca284f0c75bde2ac000', 'v0.0.146'),
    ('resnet101', '0601', 'd8cddbea530e052e726d5a1007985beb10ec36eb', 'v0.0.22'),
    ('resnet101b', '0540', 'af300066450ec147a104c5bbab92339e1eb3b3c8', 'v0.0.145'),
    ('resnet152', '0535', '64c1daa7752bf9ba8dba6e4e0e4a7947b8c235d9', 'v0.0.144'),
    ('resnet152b', '0527', '6efec2512832ccc6ebb63c8a41ab576b91eca737', 'v0.0.143'),
    ('preresnet10', '1402', '94e8fc28c7129095273a9e17f6f8d7cc7f88aefc', 'v0.0.249'),
    ('preresnet18', '0954', '21e4811aa9c868bc4afb21ca773493322ba09e82', 'v0.0.140'),
    ('preresnet34', '0812', '829f5a239d51b9138d0b3d1aae5ae4a6082d9bc3', 'v0.0.23'),
    ('preresnet50', '0669', '40bd5e93861bf9ee8892cd766afbcc23a6d3b68c', 'v0.0.23'),
    ('preresnet50b', '0667', 'b7d221efa64231c2f3b83b197ddf570fb86a409b', 'v0.0.23'),
    ('preresnet101', '0575', 'f6f6789a895f681be08db6cb9ef184d9009a2f4b', 'v0.0.23'),
    ('preresnet101b', '0587', '4211c5abf0be8d849796a4af36729f74d90620d6', 'v0.0.23'),
    ('preresnet152', '0530', '021d99dc3004530a3a1f591e88807ce84e025033', 'v0.0.23'),
    ('preresnet152b', '0566', 'fdd337e701c06a928e0706ad98fa722508a4dabe', 'v0.0.23'),
    ('preresnet200b', '0560', 'f79bd952c08555e0d7bfbcfb2c8214da9c69a0c2', 'v0.0.45'),
    ('preresnet269b', '0558', 'e2e491e1b920d8a063399642a12f7d3e3a695dfb', 'v0.0.239'),
    ('resnext101_32x4d', '0569', 'c6d1c30dcca4e83c48a2b77cfd36739a0192e244', 'v0.0.26'),
    ('resnext101_64x4d', '0543', 'dd8b7d963c2415ee1207f3705fbc33cb4ba46427', 'v0.0.26'),
    ('seresnet50', '0641', 'f3d68cfc8423b786c53390313cabfe0c4410f2d7', 'v0.0.24'),
    ('seresnet101', '0588', 'e45a9f8f09f1a7439e66032a0d79d7d5a20783b6', 'v0.0.24'),
    ('seresnet152', '0577', 'a089ba52930e9949313b9fba00a1b2e6e68f6ea4', 'v0.0.24'),
    ('seresnext50_32x4d', '0558', '5c435c1b730a0cea61b9657c8796f3c6b95ce9e8', 'v0.0.27'),
    ('seresnext101_32x4d', '0501', '98ea6fc4d36e742a01a0256707a5fa118be166dd', 'v0.0.27'),
    ('senet154', '0463', '381d2494a2ad725f62325188f94cd91c795c9902', 'v0.0.28'),
    ('airnet50_1x64d_r2', '0620', 'b6a9359d735916ff8f6192c631b7c646f489fc41', 'v0.0.120'),
    ('airnet50_1x64d_r16', '0650', '95da530f61ae4b0dda4b52c88f37bbc7cc674a03', 'v0.0.120'),
    ('airnext50_32x4d_r2', '0573', '160860f7a1750d759c36e6000080c839cda7ac56', 'v0.0.120'),
    ('bam_resnet50', '0697', 'a8c65533b4fd5e2ebf20c61d5d56936a9e1032b5', 'v0.0.124'),
    ('cbam_resnet50', '0640', 'b2314d9778b321fad2ecf3b350969038236deb96', 'v0.0.125'),
    ('pyramidnet101_a360', '0649', 'b68c786b43512e4297ce00756bd32f8beaa418ba', 'v0.0.104'),
    ('diracnet18v2', '1113', 'b85b43d13697dfbddbea6e46dea4766359fff7e5', 'v0.0.111'),
    ('diracnet34v2', '0948', '0245163a5c947bd6e07a743f17e6ca92c79c84da', 'v0.0.111'),
    ('densenet121', '0779', '06d5ebbf5b3f923ce8863268995ab5ed0f5b5019', 'v0.0.29'),
    ('densenet161', '0620', '6d05f3b9991bc570cb35fff22410d2065b667835', 'v0.0.29'),
    ('densenet169', '0686', '1978656b46c2b7de94c1e12350c74f492d683f7e', 'v0.0.29'),
    ('densenet201', '0629', '7770293931c03c2852115267dde3100d7140bbba', 'v0.0.29'),
    ('condensenet74_c4_g4', '0861', 'ef6077ec5348504346b3bcbaacbc308f825a9f87', 'v0.0.36'),
    ('condensenet74_c8_g8', '1043', '277fbfb898e0c8c7de8475184bcf5e651da10acc', 'v0.0.36'),
    ('peleenet', '1127', 'ef057fc99fda7df002d9654f0a74452e4b4b75d0', 'v0.0.141'),
    ('wrn50_2', '0613', 'd0cd9171917f04095ba8f4f48413a2ddd1ee5bc2', 'v0.0.113'),
    ('drnc26', '0788', '762c34c1f20d8ad76cec251cc0125936b608a3bc', 'v0.0.116'),
    ('drnc42', '0693', 'ec938cc429d3d0e54c34243c10be83ffae38023e', 'v0.0.116'),
    ('drnc58', '0629', '063ef19974f0158bcc6b9e4020729291462a08a3', 'v0.0.116'),
    ('drnd22', '0850', 'b25d475756dcfceb1321190b9cca6cc1f7e8e55a', 'v0.0.116'),
    ('drnd38', '0736', '153481d6f8d0b113981fc323f5b2c2ad6b2ad7f5', 'v0.0.116'),
    ('drnd54', '0623', '31e8eeb88bdbb07d8613a16471c8c5bd67ae823a', 'v0.0.116'),
    ('drnd105', '0584', 'c0d7657b2d3c4cf7d97ff407cd50dda5d1bd1880', 'v0.0.116'),
    ('dpn68', '0701', 'ad8cd4ec04a611726ee1ffcff69118a5587da691', 'v0.0.34'),
    ('dpn98', '0553', '9cd5733573f7a99062d16cd8850bb82d684704bb', 'v0.0.34'),
    ('dpn131', '0523', 'e37215991fa7e9f49245843d53de63ef1717f293', 'v0.0.34'),
    ('darknet_tiny', '1746', 'b04fa46318a78e977aa5a117786968d98d325871', 'v0.0.69'),
    ('darknet_ref', '1671', 'b2d5721f3a5f6f05cc785d57ff7a63fe82f6325e', 'v0.0.64'),
    ('darknet53', '0556', '42c57951fc2668c1a81ede52e6f4de4aac7e0278', 'v0.0.150'),
    ('irevnet301', '0887', 'ed6e6df033e659893b9021a6381f101feff002b8', 'v0.0.251'),
    ('dla34', '0823', '45504b0927fab7165e863c6801f9c0f10a906731', 'v0.0.202'),
    ('dla46c', '1471', '487ae254f1898e9e577aebb81c4466b3b7f0b238', 'v0.0.202'),
    ('dla46xc', '1396', 'a40cc67575756590ebace9c7ed9132a64afc646e', 'v0.0.202'),
    ('dla60', '0711', '92693875e59ad39963ecd641cef34c0d4b24d02e', 'v0.0.202'),
    ('dla60x', '0620', '444f31ea8f3f17128eae5359c38e874d171c3e60', 'v0.0.202'),
    ('dla60xc', '1238', '5c662c843fa2eaadfa6f61edd41a8628a4fe9763', 'v0.0.202'),
    ('dla102', '0642', 'c4ee6dcb1261ad2e4b69a906877d3cb024197307', 'v0.0.202'),
    ('dla102x', '0599', '7f83bc042bb9ae6f8443d73cacc685f5bc8714b5', 'v0.0.202'),
    ('dla102x2', '0554', '6a27a09408abaffb55ed8a041f0390c47631d522', 'v0.0.202'),
    ('dla169', '0590', '96b692a8f94c2135d5d7fc5eba6b3605c5e0595e', 'v0.0.202'),
    ('fishnet150', '0639', '114d15a6db53a9712a17afdb2a3fba4cdc3250f5', 'v0.0.168'),
    ('espnetv2_wd2', '2108', '72efda3a821eb165b2cccf34532d3c26d6525bb7', 'v0.0.238'),
    ('espnetv2_w1', '1431', 'eab8d605b475bd3659d6834ba5140d327f57c7de', 'v0.0.238'),
    ('espnetv2_w5d4', '1268', 'dc69f420f422154ab7242bcd95488541491d4982', 'v0.0.238'),
    ('espnetv2_w3d2', '1192', '2b7fc5cfacc15a63ec60109bb1b8c48d09df2a7e', 'v0.0.238'),
    ('espnetv2_w2', '0990', 'bfb3ab7c84239ff53003865e456c2a0178c48f12', 'v0.0.238'),
    ('squeezenet_v1_0', '1738', '4c55a6a5c7ae14b88a7989eea5a7dc60960120ef', 'v0.0.128'),
    ('squeezenet_v1_1', '1740', 'b236c2047fe1d9b283ccfaabb763143a214ecc33', 'v0.0.88'),
    ('squeezeresnet_v1_0', '1766', '6dc69dc26e83beaa98fa77ee64d208294f7850f9', 'v0.0.178'),
    ('squeezeresnet_v1_1', '1787', 'f40e60512a8b66f314f4d7ffab9b18dd31715b3a', 'v0.0.70'),
    ('sqnxt23_w1', '1903', 'ef3d725b418277e98ed5e590e615cc13df2f001e', 'v0.0.171'),
    ('sqnxt23v5_w1', '1786', '8b24c6e36f00be6d1b970f3c10e2b956fe281357', 'v0.0.172'),
    ('sqnxt23_w3d2', '1344', 'a5c3b21eb05532cba4b35f530fea2bdaac3d6bf5', 'v0.0.210'),
    ('sqnxt23v5_w3d2', '1292', 'c997e27957a32f89538f23d86207a044d2dc0c93', 'v0.0.212'),
    ('sqnxt23_w2', '1082', 'cf7aebefd6abb1fb3fea72dc10e0ad3dd145be8b', 'v0.0.240'),
    ('sqnxt23v5_w2', '1043', 'e9e849cdfeba0f8b3cdfd34bc214cc6526016dc4', 'v0.0.216'),
    ('shufflenet_g1_wd4', '3681', '15d3e7871b85cee9283663bbbc78dfe5e1a1a1db', 'v0.0.134'),
    ('shufflenet_g3_wd4', '3616', '064f7f7f1dd327f43e16adf5e4864a31e16d9ad9', 'v0.0.135'),
    ('shufflenet_g1_wd2', '2235', '5d83cc2822fbd0669af75d93c7940aa09e78d317', 'v0.0.174'),
    ('shufflenet_g3_wd2', '2061', '557e4397da6cebf2dd7b70e8039100f07414437a', 'v0.0.167'),
    ('shufflenet_g1_w3d4', '1677', 'b5515ea9c945c92fc4272ba7daf0002314cc61de', 'v0.0.218'),
    ('shufflenet_g3_w3d4', '1613', '55129cb578d0d53bb962e703da0746930d092c2a', 'v0.0.219'),
    ('shufflenet_g1_w1', '1348', '37cc6c5f70ad982ff3fc9c92a0ae6405bb46e2c7', 'v0.0.223'),
    ('shufflenet_g2_w1', '1333', 'e473c62fe289cc2563cb17cfa4c8562f25fd6e49', 'v0.0.241'),
    ('shufflenet_g3_w1', '1326', '95df048749f08aa69e9aed33a8bd7182b4caf2df', 'v0.0.244'),
    ('shufflenet_g4_w1', '1308', '8ed92f35a9d69874e3c9d040785f6c71c54d976c', 'v0.0.245'),
    ('shufflenet_g8_w1', '1321', '2fea8945a2115c718cdb09a22a95f4e2808e098b', 'v0.0.250'),
    ('shufflenetv2_wd2', '2073', 'c5e5a23c300c800d55e2f45e1dcb2e12907c0eae', 'v0.0.90'),
    ('shufflenetv2_w1', '1298', '3830a2da0701f2b31385aceeb828101008446812', 'v0.0.133'),
    ('shufflenetv2_w3d2', '1337', '66c1d6ed56e77d7bbf172e698e4a0d9f8a3bb442', 'v0.0.65'),
    ('shufflenetv2_w2', '1303', '349e42b513c3cf3fd7b0f9f647c645fce168f725', 'v0.0.84'),
    ('shufflenetv2b_wd2', '1787', '08a12021fa41000f5f6206446d34daa2eebb8d00', 'v0.0.157'),
    ('shufflenetv2b_w1', '1100', '21562fb22a353559c6c732e54e807766bb576dee', 'v0.0.161'),
    ('shufflenetv2b_w3d2', '0878', '7a5c7ed4aa440788875680b2a12531716ee02f98', 'v0.0.203'),
    ('shufflenetv2b_w2', '0810', '636e281ce91bf852fd20adb07f0037be8dd3d6b6', 'v0.0.242'),
    ('menet108_8x1_g3', '2042', '9e3ff283ac81b4f4e6d4a5b11d8d54b63f4aa2f0', 'v0.0.89'),
    ('menet128_8x1_g4', '1919', 'f6fd56fae09d0c528c902d1381f7cf401590d130', 'v0.0.103'),
    ('menet160_8x1_g8', '2042', '250fd7654d54c79477ef7cbf402e15d69ea3ea6a', 'v0.0.154'),
    ('menet228_12x1_g3', '1301', '39c25ca345751cac91395a602565796393fea60d', 'v0.0.131'),
    ('menet256_12x1_g4', '1218', '57160b09127535a3733f22af10d50fb16d5d2643', 'v0.0.152'),
    ('menet348_12x1_g3', '0936', 'ee7e056d0f38a68a6d6c85fe8162bee944a73121', 'v0.0.173'),
    ('menet352_12x1_g8', '1172', 'c256ae25591e33ce6b9e12177305eacb3dd9620c', 'v0.0.198'),
    ('menet456_24x1_g3', '0779', '5af355f6457347168d5b95323b6d7480360398d8', 'v0.0.237'),
    ('mobilenet_wd4', '2216', '09c50ab8d72049a4aa9cae4bd1502859522b9a70', 'v0.0.62'),
    ('mobilenet_wd2', '1337', '48d12ee398fa6dc23596f669fb202f08108a6ccc', 'v0.0.156'),
    ('mobilenet_w3d4', '1053', 'd7ec3192f88b7017d477fdb704ad6ad77a4c5cc1', 'v0.0.130'),
    ('mobilenet_w1', '0866', 'b888f817a2978cdeb00a09fd5e71c3f2a52ddd8c', 'v0.0.155'),
    ('fdmobilenet_wd4', '3063', '55407f3a3e3370fa2951f651f14faac3bf9a9f28', 'v0.0.177'),
    ('fdmobilenet_wd2', '1976', '6299d44272390440be808e58059219b0d57907e4', 'v0.0.83'),
    ('fdmobilenet_w3d4', '1599', 'cdfc2e043017be0166cf06cb9f49e0f516aa5d15', 'v0.0.159'),
    ('fdmobilenet_w1', '1316', '0ed6f00cbb5095eff002882e31c006edb1c5235e', 'v0.0.162'),
    ('mobilenetv2_wd4', '2411', '9fc398d348226c410659464d12b0fe6b7d4506e7', 'v0.0.137'),
    ('mobilenetv2_wd2', '1444', 'ca0906e176f15855aa8c8d771c841c3f9cd3d454', 'v0.0.170'),
    ('mobilenetv2_w3d4', '1047', 'a25fd26c426b5af8c5761b9d634b508622f019cf', 'v0.0.230'),
    ('mobilenetv2_w1', '0866', 'efc3331e08dfc578526bbf5e161c15e50b146c63', 'v0.0.213'),
    ('igcv3_wd4', '2828', '25942192926a7dcdd0c57238336a8a0ef840e079', 'v0.0.142'),
    ('igcv3_wd2', '1704', '86246558ade35232344a4c448288ae3927143f9c', 'v0.0.132'),
    ('igcv3_w3d4', '1099', 'b0dbc54a5c40c7bd55ebd3cab05e39263064f4ec', 'v0.0.207'),
    ('igcv3_w1', '0898', '5fd85acd8a4ed75845e2ef770c25460c5f7eff95', 'v0.0.243'),
    ('mnasnet', '1144', '688e523d02834b34f4a693a1b18e7a523483eb58', 'v0.0.117'),
    ('darts', '0897', '8986fe64b3f853704a88010f0a735a9e6e33bd97', 'v0.0.118'),
    ('xception', '0547', '7a5be9582fd7a4771ede5290645be394d66d29ca', 'v0.0.115'),
    ('inceptionv3', '0561', '4ddea4df44f132ffc9e2b22b1e7d686f8b59703b', 'v0.0.92'),
    ('inceptionv4', '0526', '02e53701d1bda64b057b41fa90d8e04a17d07f66', 'v0.0.105'),
    ('inceptionresnetv2', '0492', '3d3de82bb9db27b260603fe2f956ad929c3eb277', 'v0.0.107'),
    ('polynet', '0450', '6dc7028b0edc48c452f83dd38448b1242c554a5e', 'v0.0.96'),
    ('nasnet_4a1056', '0796', 'f09950c0f4a333007dc33049531534b8cd9f8521', 'v0.0.97'),
    ('nasnet_6a4032', '0422', 'd49d46631abda0ec7ac4a0076e6f8d05bf99b7d1', 'v0.0.101'),
    ('pnasnet5large', '0426', '3c2755dce80a29dea19b398dce514a640da2aaa3', 'v0.0.114'),
    ('nin_cifar10', '0743', '045abfde63c6b73fbb1b6c6b062c9da5e2485750', 'v0.0.175'),
    ('nin_cifar100', '2839', '891047637c63f274d4138a430fcaf5f92f054ad4', 'v0.0.183'),
    ('resnet20_cifar10', '0597', '15145d2e00c85b5c295b6999068ce4b494febfb0', 'v0.0.163'),
    ('resnet20_cifar100', '2964', '6a85f07e9bda4721ee68f9b7350250b866247324', 'v0.0.180'),
    ('resnet56_cifar10', '0452', 'eb7923aa7d53e4e9951483b05c9629010fbd75a4', 'v0.0.163'),
    ('resnet56_cifar100', '2488', '2d641cdef73a9cdc440d7ebfb665167907a6b3bd', 'v0.0.181'),
    ('resnet110_cifar10', '0369', '27d76fce060ce5737314f491211734bd10c60308', 'v0.0.163'),
    ('resnet110_cifar100', '2280', 'd2ec4ff1c85095343031a0b11a671c4799ae1187', 'v0.0.190'),
    ('resnet164bn_cifar10', '0368', 'd86593667f30bfef0c0ad237f2da32601b048312', 'v0.0.179'),
    ('resnet164bn_cifar100', '2044', '190ab6b485404e43c41a85542e57adb051744aa0', 'v0.0.182'),
    ('resnet1001_cifar10', '0328', '0e27556cdc97b7d0612d4518546a9b0479e030c3', 'v0.0.201'),
    ('resnet1202_cifar10', '0353', 'd82bb4359d16e68989547f8b1153c8f23264e46c', 'v0.0.214'),
    ('preresnet20_cifar10', '0651', '5cf94722c7969e136e2174959fee4d7b95528f54', 'v0.0.164'),
    ('preresnet20_cifar100', '3022', 'e3fd9391a621da1afd77f1c09ae0c9bdda4e17aa', 'v0.0.187'),
    ('preresnet56_cifar10', '0449', '73ea193a6f184d034a4b5b911fe6d23473eb0220', 'v0.0.164'),
    ('preresnet56_cifar100', '2505', 'f879fb4e9c9bc328b97ca8999575ea29343bbd79', 'v0.0.188'),
    ('preresnet110_cifar10', '0386', '544ed0f0e0b3c0da72395924e2ea381dbf381e52', 'v0.0.164'),
    ('preresnet110_cifar100', '2267', '4e010af04fefb74f6535a1de150f695460ec0550', 'v0.0.191'),
    ('preresnet164bn_cifar10', '0364', 'c0ff243801f078c6e6be72e1d3b67d88d61c4454', 'v0.0.196'),
    ('preresnet164bn_cifar100', '2018', '5228dfbdebf0f4699dae38a4a9b8310b08189d48', 'v0.0.192'),
    ('preresnet1001_cifar10', '0265', '1f3028bdf7143b8f99340b1b1a0a8e029d7020a0', 'v0.0.209'),
    ('preresnet1202_cifar10', '0339', 'cc2bd85a97842f7a444deb78262886a264a42c25', 'v0.0.246'),
    ('resnext29_32x4d_cifar10', '0315', '442eca6c30448563f931174d37796c2f08c778b7', 'v0.0.169'),
    ('resnext29_32x4d_cifar100', '1950', 'de139852f2876a04c74c271d50f0a50ba75ece3e', 'v0.0.200'),
    ('resnext29_16x64d_cifar10', '0241', 'e80d3cb5f8d32be2025fe8fb7a7369b2d004217e', 'v0.0.176'),
    ('pyramidnet110_a48_cifar10', '0372', '965fce37e26ef4e3724df869fe90283669fe9daf', 'v0.0.184'),
    ('pyramidnet110_a48_cifar100', '2095', 'b74f12c8d11de3ddd9fa51fe93c1903675a43a3c', 'v0.0.186'),
    ('pyramidnet110_a84_cifar10', '0298', '7b38a0f65de0bec2f4ceb83398fef61009a2c129', 'v0.0.185'),
    ('pyramidnet110_a84_cifar100', '1887', '842b3809619ec81c6e27defcad9df5c3dbc0ae55', 'v0.0.199'),
    ('pyramidnet110_a270_cifar10', '0251', 'b3456ddd5919ef861ec607f8287bd071de0ba077', 'v0.0.194'),
    ('densenet40_k12_cifar10', '0561', 'a37df881a11487fdde772254a82c20c3e45b461b', 'v0.0.193'),
    ('densenet40_k12_cifar100', '2490', 'd06839db7eec0331354ca31b421c6fbcd4665fd3', 'v0.0.195'),
    ('densenet40_k12_bc_cifar10', '0643', '234918e7144b95454e1417035c73391663a68401', 'v0.0.231'),
    ('densenet40_k12_bc_cifar100', '2841', '968e5667c29dd682a90c3f8a488e00a9efe0d29f', 'v0.0.232'),
    ('densenet40_k24_bc_cifar10', '0452', '3ec459af58cf2106bfcbdad090369a1f3d41ef3c', 'v0.0.220'),
    ('densenet40_k24_bc_cifar100', '2267', 'f744296d04d703c202b0b78cdb32e7fc40116584', 'v0.0.221'),
    ('densenet40_k36_bc_cifar10', '0404', '6be4225a6d0e5fb68bdc9cda471207c0b5420395', 'v0.0.224'),
    ('densenet40_k36_bc_cifar100', '2050', '49b6695fe06d98cfac5d4fdbdb716edb268712c2', 'v0.0.225'),
    ('densenet100_k12_cifar10', '0366', '85031735e1c80d3a6254fe8649c5e9bae2d54315', 'v0.0.205'),
    ('densenet100_k12_cifar100', '1964', 'f04f59203ad863f466c25fa9bbfc18686d72a46a', 'v0.0.206'),
    ('densenet100_k24_cifar10', '0313', '939ef3090b6219e5afabc97f03cc34365c729ada', 'v0.0.252'),
    ('densenet100_k12_bc_cifar10', '0416', '160a064165eddf492970a99b5a8ca9689bf94fea', 'v0.0.189'),
    ('densenet100_k12_bc_cifar100', '2119', 'a37ebc2a083fbe8e7642988945d1092fb421f182', 'v0.0.208'),
    ('xdensenet40_2_k24_bc_cifar10', '0531', 'd3c448ab2c110f873579093ff9a69e735d80b4e7', 'v0.0.226'),
    ('xdensenet40_2_k24_bc_cifar100', '2396', '84357bb40bcd1da5cf6237ea5755a309bcf36d49', 'v0.0.227'),
    ('xdensenet40_2_k36_bc_cifar10', '0437', 'fb6d7431c005eb9965da0e1b2872c048d6b31b30', 'v0.0.233'),
    ('xdensenet40_2_k36_bc_cifar100', '2165', '9ac51e902167ba05f1c21ed1a9690c1fd4cad3eb', 'v0.0.234'),
    ('wrn16_10_cifar10', '0293', '4ac60015e3b287580d11e605793b3426e8184137', 'v0.0.166'),
    ('wrn16_10_cifar100', '1895', 'd6e852788e29532c8a12bb39617a2e81aba2483f', 'v0.0.204'),
    ('wrn28_10_cifar10', '0239', 'f8a24941ca542f78eda2d192f461b1bac0600d27', 'v0.0.166'),
    ('wrn40_8_cifar10', '0237', '3f56f24a07be7155fb143cc4360755d564e3761a', 'v0.0.166'),
    ('ror3_56_cifar10', '0543', '7ca1b24c4a573d53484ca92b19bad5c08e38fa8b', 'v0.0.228'),
    ('ror3_56_cifar100', '2549', 'a7903e5f5f80bf53c07e12ce34659e0d9af4b106', 'v0.0.229'),
    ('ror3_110_cifar10', '0435', 'bf021f253fc1cf29b30a1eb579c7c4693f963933', 'v0.0.235'),
    ('ror3_110_cifar100', '2364', '13de922a8f8758a15eaf1d283dc42e7dcf0f3fda', 'v0.0.236'),
    ('shakeshakeresnet20_2x16d_cifar10', '0515', 'e2f524b5196951f48495973a087135ca974ec327', 'v0.0.215'),
    ('shakeshakeresnet20_2x16d_cifar100', '2922', '84772a31f6f6bb3228276515a8d4371c25925c85', 'v0.0.247'),
    ('shakeshakeresnet26_2x32d_cifar10', '0317', '5422fce187dff99fa8f4678274a8dd1519e23e27', 'v0.0.217'),
    ('shakeshakeresnet26_2x32d_cifar100', '1880', '750a574e738cf53079b6965410e07fb3abef82fd', 'v0.0.222')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.chainer', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $CHAINER_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.npz'.format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to {}.'.format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + '.zip'
    _download(
        url='{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip'.format(
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
        raise ValueError('Downloaded file has different hash. Please try again.')


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """Download an given URL

    Parameters
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

    Returns
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
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading {} from {}...'.format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'
                                      ' The repo may be outdated or download may be incomplete. '
                                      'If the "repo_url" is overridden, consider switching to '
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, 's' if retries > 1 else ''))

    return fname


def _check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash
