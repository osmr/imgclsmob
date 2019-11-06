"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_model', 'download_model', 'calc_num_params']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '1824', '8ada73bf8de14507a949c6a4a7e55d001a633bc5', 'v0.0.394'),
    ('alexnetb', '1900', '55176c6ad29c18243f4fdd0764840018a4ed1ca4', 'v0.0.384'),
    ('zfnet', '1727', 'd010ddca1eb32a50a8cceb475c792f53e769b631', 'v0.0.395'),
    ('zfnetb', '1490', 'f6bec24eba037c8e4956704ed5bafaed29966601', 'v0.0.400'),
    ('vgg11', '1036', '71e85f6ef76f56e3e89d597d2fc461496ed281e9', 'v0.0.381'),
    ('vgg13', '0975', '2b2c8770a7610d9dcd444ec8ae992681e270eb42', 'v0.0.388'),
    ('vgg16', '0865', '5ca155da3dc6687e070ff34815cb5aabd0bed4b9', 'v0.0.401'),
    ('vgg19', '0839', 'd4e69a0d393f4d46f1d9c4d4ba96f5a83de3399c', 'v0.0.109'),
    ('bn_vgg11', '0961', '10f01fba064ec168df074b98d59ae7b82b1207d4', 'v0.0.339'),
    ('bn_vgg13', '0913', 'b1acd7158e6e9973ce9e274c65ceb64a244f9967', 'v0.0.353'),
    ('bn_vgg16', '0779', '0f570b928b180f909fa39df3924f89c746816722', 'v0.0.359'),
    ('bn_vgg19', '0712', '3f286cbd2a57abb4c516425c5e095c2cfc8d54e3', 'v0.0.360'),
    ('bn_vgg11b', '0996', 'ef747edc87705e1ed500a31c80199273b2fbd5fa', 'v0.0.407'),
    ('bn_vgg13b', '0963', 'cf9352f47805c18798c0f80ab0e158ec5401331e', 'v0.0.110'),
    ('bn_vgg16b', '0874', 'af4f2d0bbfda667e6b7b3ad4cda5ca331021cd18', 'v0.0.110'),
    ('bn_vgg19b', '0840', 'b6919f7f74b3174a86818062b2d1d4cf5a110b8b', 'v0.0.110'),
    ('bninception', '0774', 'd79ba5f573ba2da5fea5e4c9a7f67ddd526e234b', 'v0.0.405'),
    ('resnet10', '1436', '67d9a618e8670497386af806564f7ac1a4dbcd76', 'v0.0.248'),
    ('resnet12', '1328', 'd7d2f4d6c7fcf3aff0458533ae5204b7f0eee2d7', 'v0.0.253'),
    ('resnet14', '1246', 'd5b55c113168c02f1b39b65f8908b0db467a2d74', 'v0.0.256'),
    ('resnetbc14b', '1151', 'ca61209c4052228edad1fe7bb48ad6a19db509d1', 'v0.0.309'),
    ('resnet16', '1118', 'd54bc41afa244476ca28380111f66d188905ecbc', 'v0.0.259'),
    ('resnet18_wd4', '1785', 'fe79b31f56e7becab9c014dbc14ccdb564b5148f', 'v0.0.262'),
    ('resnet18_wd2', '1327', '6654f50ad357f4596502b92b3dca2147776089ac', 'v0.0.263'),
    ('resnet18_w3d4', '1106', '3636648b504e1ba134947743eb34dd0e78feda02', 'v0.0.266'),
    ('resnet18', '0982', '0126861b4cd7f7b14196b1e01827da688f8bab6d', 'v0.0.153'),
    ('resnet26', '0854', '258347330aefca1c2387583680f812c9d6a8a66c', 'v0.0.305'),
    ('resnetbc26b', '0797', '7af52a73b234dc56ab4b0757cf3ea772d0699622', 'v0.0.313'),
    ('resnet34', '0780', '3f775482a327e5fc4850fbb77785bfc55e171e5f', 'v0.0.291'),
    ('resnetbc38b', '0700', '3fbac61d86810d489988a92f425f1a6bfe46f155', 'v0.0.328'),
    ('resnet50', '0633', 'b00d1c8e52aa7a2badc705b1545aaf6ccece6ce9', 'v0.0.329'),
    ('resnet50b', '0638', '8a5473ef985d65076a3758117ad5700d726bd952', 'v0.0.308'),
    ('resnet101', '0622', 'ab0cf005bbe9b17e53f9e3c330c6147a8c80b3a5', 'v0.0.1'),
    ('resnet101b', '0530', 'f059ba3c7fa4a65f2da6e17f3718662d59836637', 'v0.0.357'),
    ('resnet152', '0550', '800b2cb1959a0d3648483e86917502b8f63dc37e', 'v0.0.144'),
    ('resnet152b', '0499', '667ea926f3753e0c8336fa78969171d64f819cc4', 'v0.0.378'),
    ('preresnet10', '1421', 'b3973cd4461287d61df081d6f689d293eacf2248', 'v0.0.249'),
    ('preresnet12', '1348', '563066fa8fcf8b5f19906b933fea784965d68192', 'v0.0.257'),
    ('preresnet14', '1239', '4be725fd3f06c99c46817fce3b69caf2ebc62414', 'v0.0.260'),
    ('preresnetbc14b', '1181', 'a68d31c372e647474ae954e51e5bc2ba9fb3f166', 'v0.0.315'),
    ('preresnet16', '1108', '06d8c87e29284dac19a9019485e210541532411a', 'v0.0.261'),
    ('preresnet18_wd4', '1811', '41135c15210390e9a564b14e8ae2ebda1a662ec1', 'v0.0.272'),
    ('preresnet18_wd2', '1340', 'c1fe4e314188eeb93302432d03731a91ce8bc9f2', 'v0.0.273'),
    ('preresnet18_w3d4', '1105', 'ed2f9ca434b6910b92657eefc73ad186396578d5', 'v0.0.274'),
    ('preresnet18', '0972', '5651bc2dbb200382822a6b64375d240f747cc726', 'v0.0.140'),
    ('preresnet26', '0851', '99e7d6cc5944cd7cf6d4746e6fdf18b477d3d9a0', 'v0.0.316'),
    ('preresnetbc26b', '0803', 'd7283bdd70e1b75520fe2cdcc273d51715e077b4', 'v0.0.325'),
    ('preresnet34', '0774', 'fd5bd1e883048e29099768465df2dd9e891803f4', 'v0.0.300'),
    ('preresnetbc38b', '0657', '9e523bb92dc592ee576a6bb73a328dc024bdc967', 'v0.0.348'),
    ('preresnet50', '0647', '222ca73b021f893b925c15e24ea2a6bc0fdf2546', 'v0.0.330'),
    ('preresnet50b', '0655', '8b60378ee3aed878d27a2b4a9ddc596a812c7649', 'v0.0.307'),
    ('preresnet101', '0591', '4bacff796e113562e1dfdf71cfa7c6ed33e0ba86', 'v0.0.2'),
    ('preresnet101b', '0556', '76bfe6d020b55f163e77de6b1c27be6b0bed8b7b', 'v0.0.351'),
    ('preresnet152', '0555', 'c842a030abbcc21a0f2a9a8299fc42204897a611', 'v0.0.14'),
    ('preresnet152b', '0516', 'f3805f4b8c845798b711171ad6632bcf56259844', 'v0.0.386'),
    ('preresnet200b', '0588', 'f7104ff306ed5de2c27f3c855051c22bda167981', 'v0.0.45'),
    ('preresnet269b', '0581', '1a7878bb10923b22bda58d7935dfa6e5e8a7b67d', 'v0.0.239'),
    ('resnext14_16x4d', '1248', '35ffac2a26374e71b6bf4bc9f90b7a1a1dd47e7d', 'v0.0.370'),
    ('resnext14_32x2d', '1281', '14521186b8c78c7c07f3904360839f22c180f65e', 'v0.0.371'),
    ('resnext14_32x4d', '1146', '89aa679393d8356ce5589749b4371714bf4ceac0', 'v0.0.327'),
    ('resnext26_32x2d', '0887', 'c3bd130747909a8c89546f3b3f5ce08bb4f55731', 'v0.0.373'),
    ('resnext26_32x4d', '0746', '1011ac35e30d753b79f0600a5376c87a37b67a61', 'v0.0.332'),
    ('resnext101_32x4d', '0611', 'cf962440f11fe683fd02ec04f2102d9f47ce38a7', 'v0.0.10'),
    ('resnext101_64x4d', '0575', '651abd029bcc4ce88c62e1d900a710f284a8281e', 'v0.0.10'),
    ('seresnet10', '1366', '6ec312230962e5f809e2dd77444a2a1bdfbb06f4', 'v0.0.354'),
    ('seresnet18', '0961', '022123a5e88c9917e63165f5b5a7808a606d452a', 'v0.0.355'),
    ('seresnet26', '0824', '64fc8759c5bb9b9b40b2e33a46420ee22ae268c9', 'v0.0.363'),
    ('seresnetbc26b', '0703', 'b98d9d6afca4d79d0347001542162b9fe4071d39', 'v0.0.366'),
    ('seresnetbc38b', '0595', '03671c05f5f684b44085383b7b89a8b44a7524fe', 'v0.0.374'),
    ('seresnet50', '0640', '8820f2af62421ce2e1df989d6e0ce7916c78ff86', 'v0.0.11'),
    ('seresnet50b', '0539', '459e6871e944d1c7102ee9c055ea428b8d9a168c', 'v0.0.387'),
    ('seresnet101', '0589', '5e6e831b7518b9b8a049dd60ed1ff82ae75ff55e', 'v0.0.11'),
    ('seresnet152', '0576', '814cf72e0deeab530332b16fb9b609e574afec61', 'v0.0.11'),
    ('sepreresnet10', '1338', '935ed56009a64c893153cdba8e4a4f87f7184e71', 'v0.0.377'),
    ('sepreresnet18', '0963', 'c065cd9e1c026d0529526cfc945c137bade6f0c7', 'v0.0.380'),
    ('sepreresnetbc26b', '0660', 'f750b2f588a27620b30c86f0060a41422d4a0f75', 'v0.0.399'),
    ('sepreresnetbc38b', '0578', '12827fcd3c8c1a8c8ba1d109e85ffa67e7ab306a', 'v0.0.409'),
    ('seresnext50_32x4d', '0554', '99e0e9aa4578af9f15045c1ceeb684a2e988628a', 'v0.0.12'),
    ('seresnext101_32x4d', '0505', '0924f0a2c1de90dc964c482b7aff6232dbef3600', 'v0.0.12'),
    ('senet16', '0820', '373aeafdc994c3e03bf483a9fa3ecb152353722a', 'v0.0.341'),
    ('senet28', '0598', '27165b63696061e57c141314d44732aa65f807a8', 'v0.0.356'),
    ('senet154', '0461', '6512228c820897cd09f877527a553ca99d673956', 'v0.0.13'),
    ('ibn_resnet50', '0641', 'e48a1fe5f7e448d4b784ef4dc0f33832f3370a9b', 'v0.0.127'),
    ('ibn_resnet101', '0561', '5279c78a0dbfc722cfcfb788af479b6133920528', 'v0.0.127'),
    ('ibnb_resnet50', '0686', 'e138995e6acda4b496375beac6d01cd7a9f79876', 'v0.0.127'),
    ('ibn_resnext101_32x4d', '0542', 'b5233c663a4d207d08c21107d6c951956e910be8', 'v0.0.127'),
    ('ibn_densenet121', '0725', 'b90b0615e6ec5c9652e3e553e27851c8eaf01adf', 'v0.0.127'),
    ('ibn_densenet169', '0651', '96dd755e0df8a54349278e0cd23a043a5554de08', 'v0.0.127'),
    ('airnet50_1x64d_r2', '0590', '3ec422128d17314124c02e3bb0f77e26777fb385', 'v0.0.120'),
    ('airnet50_1x64d_r16', '0619', '090179e777f47057bedded22d669bf9f9ce3169c', 'v0.0.120'),
    ('airnext50_32x4d_r2', '0551', 'c68156e5e446a1116b1b42bc94b3f881ab73fe92', 'v0.0.120'),
    ('bam_resnet50', '0658', '96a37c82bdba821385b29859ad1db83061a0ca5b', 'v0.0.124'),
    ('cbam_resnet50', '0605', 'a1172fe679622224dcc88c00020936ad381806fb', 'v0.0.125'),
    ('pyramidnet101_a360', '0620', '3a24427baf21ee6566d7e4c7dee25da0e5744f7f', 'v0.0.104'),
    ('diracnet18v2', '1170', 'e06737707a1f5a5c7fe4e57da92ed890b034cb9a', 'v0.0.111'),
    ('diracnet34v2', '0993', 'a6a661c0c3e96af320e5b9bf65a6c8e5e498a474', 'v0.0.111'),
    ('densenet121', '0704', 'cf90d1394d197fde953f57576403950345bd0a66', 'v0.0.314'),
    ('densenet161', '0644', 'c0fb22c83e8077a952ce1a0c9703d1a08b2b9e3a', 'v0.0.3'),
    ('densenet169', '0629', '44974a17309bb378e97c8f70f96f961ffbf9458d', 'v0.0.406'),
    ('densenet201', '0663', '71ece4ad7be5d1e2aa4bbf6f1a6b32ac2562d847', 'v0.0.3'),
    ('condensenet74_c4_g4', '0828', '5ba550494cae7081d12c14b02b2a02365539d377', 'v0.0.4'),
    ('condensenet74_c8_g8', '1006', '3574d874fefc3307f241690bad51f20e61be1542', 'v0.0.4'),
    ('peleenet', '1151', '9c47b80297ac072a923cda763b78e7218cd52d3a', 'v0.0.141'),
    ('wrn50_2', '0641', '83897ab9f015f6f988e51108e12518b08e1819dd', 'v0.0.113'),
    ('drnc26', '0755', '35405bd52a0c721f3dc64f18d433074f263b7339', 'v0.0.116'),
    ('drnc42', '0657', '7c99c4608a9a5e5f073f657b92f258ba4ba5ac77', 'v0.0.116'),
    ('drnc58', '0601', '70ec1f56c23da863628d126a6ed0ad10f037a2ac', 'v0.0.116'),
    ('drnd22', '0823', '5c2c6a0cf992409ab388e04e9fbd06b7141bdf47', 'v0.0.116'),
    ('drnd38', '0695', '4630f0fb3f721f4a2296e05aacb1231ba7530ae5', 'v0.0.116'),
    ('drnd54', '0586', 'bfdc1f8826027b247e2757be45b176b3b91b9ea3', 'v0.0.116'),
    ('drnd105', '0548', 'a643f4dcf9e4b69eab06b76e54ce22169f837592', 'v0.0.116'),
    ('dpn68', '0679', 'a33c98c783cbf93cca4cc9ce1584da50a6b12077', 'v0.0.310'),
    ('dpn98', '0553', '52c55969835d56185afa497c43f09df07f58f0d3', 'v0.0.17'),
    ('dpn131', '0548', '0c53e5b380137ccb789e932775e8bd8a811eeb3e', 'v0.0.17'),
    ('darknet_tiny', '1784', '4561e1ada619e33520d1f765b3321f7f8ea6196b', 'v0.0.69'),
    ('darknet_ref', '1718', '034595b49113ee23de72e36f7d8a3dbb594615f6', 'v0.0.64'),
    ('darknet53', '0564', 'b36bef6b297055dda3d17a3f79596511730e1963', 'v0.0.150'),
    ('irevnet301', '0841', '95dc8d94257bf16027edd7077b785a8676369fca', 'v0.0.251'),
    ('bagnet9', '2961', 'cab1179284e9749697f38c1c7e5f0e172be12c89', 'v0.0.255'),
    ('bagnet17', '1884', '6b2a100f8d14d4616709586483f625743ed04769', 'v0.0.255'),
    ('bagnet33', '1301', '4f17b6e837dacd978b15708ffbb2c1e6be3c371a', 'v0.0.255'),
    ('dla34', '0794', '04698d78b16f2d08e4396b5b0c9f46cb42542242', 'v0.0.202'),
    ('dla46c', '1323', 'efcd363642a4b479892f47edae7440f0eea05edb', 'v0.0.282'),
    ('dla46xc', '1269', '00d3754ad0ff22636bb1f4b4fb8baebf4751a1ee', 'v0.0.293'),
    ('dla60', '0669', 'b2cd6e51a322512a6cb45414982a2ec71285daad', 'v0.0.202'),
    ('dla60x', '0598', '88547d3f81c4df711b15457cfcf37e2b703ed895', 'v0.0.202'),
    ('dla60xc', '1091', '0f6381f335e5bbb4c69b360be61a4a08e5c7a9de', 'v0.0.289'),
    ('dla102', '0605', '11df13220b44f51dc8c925fbd9fc334bc8d115b4', 'v0.0.202'),
    ('dla102x', '0577', '58331655844f9d95bcf2bb90de6ac9cf3b66bd5e', 'v0.0.202'),
    ('dla102x2', '0536', '079361117045dc661b63ce4b14408d403bc91844', 'v0.0.202'),
    ('dla169', '0566', 'ae0c6a82acfaf9dc459ac5a032106c2727b71d4f', 'v0.0.202'),
    ('fishnet150', '0604', 'f5af4873ff5730f589a6c4a505ede8268e6ce3e3', 'v0.0.168'),
    ('espnetv2_wd2', '2015', 'd234781f81e5d1b5ae6070fc851e3f7bb860b9fd', 'v0.0.238'),
    ('espnetv2_w1', '1345', '550d54229d7fd8f7c090601c2123ab3ca106393b', 'v0.0.238'),
    ('espnetv2_w5d4', '1218', '85d97b2b1c9ebb176f634949ef5ca6d7fe70f09c', 'v0.0.238'),
    ('espnetv2_w3d2', '1129', '3bbb49adaa4fa984a67f82862db7dcfc4998429e', 'v0.0.238'),
    ('espnetv2_w2', '0961', '13ba0f7200eb745bacdf692905fde711236448ef', 'v0.0.238'),
    ('squeezenet_v1_0', '1766', 'afdbcf1aef39237300656d2c5a7dba19230e29fc', 'v0.0.128'),
    ('squeezenet_v1_1', '1772', '25b77bc39e35612abbe7c2344d2c3e1e6756c2f8', 'v0.0.88'),
    ('squeezeresnet_v1_0', '1809', '25bfc02edeffb279010242614e7d73bbeacc0170', 'v0.0.178'),
    ('squeezeresnet_v1_1', '1821', 'c27ed88f1b19eb233d3925efc71c71d25e4c434e', 'v0.0.70'),
    ('sqnxt23_w1', '1906', '97b74e0c4d6bf9fc939771d94b2f6dd97de34024', 'v0.0.171'),
    ('sqnxt23v5_w1', '1785', '2fe3ad67d73313193a77690b10c17cbceef92340', 'v0.0.172'),
    ('sqnxt23_w3d2', '1350', 'c2f21bce669dbe50fba544bcc39bc1302f63e1e8', 'v0.0.210'),
    ('sqnxt23v5_w3d2', '1301', 'c244844ba2f02dadd350dddd74e21360b452f9dd', 'v0.0.212'),
    ('sqnxt23_w2', '1100', 'b9bb7302824f89f16e078f0a506e3a8c0ad9c74e', 'v0.0.240'),
    ('sqnxt23v5_w2', '1066', '229b0d3de06197e399eeebf42dc826b78f0aba86', 'v0.0.216'),
    ('shufflenet_g1_wd4', '3729', '47dbd0f279da6d3056079bb79ad39cabbb3b9415', 'v0.0.134'),
    ('shufflenet_g3_wd4', '3653', '6abdd65e087e71f80345415cdf7ada6ed2762d60', 'v0.0.135'),
    ('shufflenet_g1_wd2', '2261', 'dae4bdadd7d48bee791dff2a08cd697cff0e9320', 'v0.0.174'),
    ('shufflenet_g3_wd2', '2080', 'ccaacfc8d9ac112c6143269df6e258fd55b662a7', 'v0.0.167'),
    ('shufflenet_g1_w3d4', '1711', '161cd24aa0b2e2afadafa69b44a28af222f2ec7a', 'v0.0.218'),
    ('shufflenet_g3_w3d4', '1650', '3f3b0aef0ce3174c78ff42cf6910c6e34540fc41', 'v0.0.219'),
    ('shufflenet_g1_w1', '1389', '4cfb65a30761fe548e0b5afbb5d89793ec41e4e9', 'v0.0.223'),
    ('shufflenet_g2_w1', '1363', '07256203e217a7b31f1c69a5bd38a6674bce75bc', 'v0.0.241'),
    ('shufflenet_g3_w1', '1348', 'ce54f64ecff87556a4303380f46abaaf649eb308', 'v0.0.244'),
    ('shufflenet_g4_w1', '1335', 'e2415f8270a4b6cbfe7dc97044d497edbc898577', 'v0.0.245'),
    ('shufflenet_g8_w1', '1342', '9a979b365424addba75c559a61a77ac7154b26eb', 'v0.0.250'),
    ('shufflenetv2_wd2', '1865', '9c22238b5fa9c09541564e8ed7f357a5f7e8cd7c', 'v0.0.90'),
    ('shufflenetv2_w1', '1163', 'c71dfb7a814c8d8ef704bdbd80995e9ea49ff4ff', 'v0.0.133'),
    ('shufflenetv2_w3d2', '0942', '26a9230405d956643dcd563a5a383844c49b5907', 'v0.0.288'),
    ('shufflenetv2_w2', '0845', '337255f6ad40a93c2f23fc593bad4b2755a327fa', 'v0.0.301'),
    ('shufflenetv2b_wd2', '1822', '01d18d6fa1a6136f605a4277f47c9a757f9ede3b', 'v0.0.157'),
    ('shufflenetv2b_w1', '1125', '6a5d3dc446e6a00cf60fe8aa2f4139d74d766305', 'v0.0.161'),
    ('shufflenetv2b_w3d2', '0911', 'f2106fee0748d7f0d40db16b228782b6d7636737', 'v0.0.203'),
    ('shufflenetv2b_w2', '0834', 'cb36b92ca4ca3bee470b739021d01177e0601c5f', 'v0.0.242'),
    ('menet108_8x1_g3', '2076', '6acc82e46dfc1ce0dd8c59668aed4a464c8cbdb5', 'v0.0.89'),
    ('menet128_8x1_g4', '1959', '48fa80fc363adb88ff580788faa8053c9d7507f3', 'v0.0.103'),
    ('menet160_8x1_g8', '2084', '0f4fce43b4234c5bca5dd76450b698c2d4daae65', 'v0.0.154'),
    ('menet228_12x1_g3', '1316', '5b670c42031d0078e2ae981829358d7c1b92ee30', 'v0.0.131'),
    ('menet256_12x1_g4', '1252', '14c6c86df96435c693eb7d0fcd8d3bf4079dd621', 'v0.0.152'),
    ('menet348_12x1_g3', '0958', 'ad50f635a1f7b799a19a0a9c71aa9939db8ffe77', 'v0.0.173'),
    ('menet352_12x1_g8', '1200', '4ee200c5c98c64a2503cea82ebf62d1d3c07fb91', 'v0.0.198'),
    ('menet456_24x1_g3', '0799', '826c002244f1cdc945a95302b1ce5c66d949db74', 'v0.0.237'),
    ('mobilenet_wd4', '2249', '1ad5e8fe8674cdf7ffda8450095eb96d227397e0', 'v0.0.62'),
    ('mobilenet_wd2', '1355', '41a21242c95050407df876cfa44bb5d3676aa751', 'v0.0.156'),
    ('mobilenet_w3d4', '1076', 'd801bcaea83885b16a0306b8b77fe314bbc585c3', 'v0.0.130'),
    ('mobilenet_w1', '0895', '7e1d739f0fd4b95c16eef077c5dc0a5bb1da8ad5', 'v0.0.155'),
    ('fdmobilenet_wd4', '3098', '2b22b709a05d7ca6e43acc6f3a9f27d0eb2e01cd', 'v0.0.177'),
    ('fdmobilenet_wd2', '2015', '414dbeedb2f829dcd8f94cd7fef10aae6829f06f', 'v0.0.83'),
    ('fdmobilenet_w3d4', '1641', '5561d58aa8889d8d93f2062a2af4e4b35ad7e769', 'v0.0.159'),
    ('fdmobilenet_w1', '1338', '9d026c04112de9f40e15fa40457d77941443c327', 'v0.0.162'),
    ('mobilenetv2_wd4', '2451', '05e1e3a286b27c17ea11928783c4cd48b1e7a9b2', 'v0.0.137'),
    ('mobilenetv2_wd2', '1493', 'b82d79f6730eac625e6b55b0618bff8f7a1ed86d', 'v0.0.170'),
    ('mobilenetv2_w3d4', '1082', '8656de5a8d90b29779c35c5ce521267c841fd717', 'v0.0.230'),
    ('mobilenetv2_w1', '0887', '13a021bca5b679b76156829743f7182da42e8bb6', 'v0.0.213'),
    ('mobilenetv3_large_w1', '0779', '38e392f58bdf99b2832b26341bc9704ac63a3672', 'v0.0.411'),
    ('igcv3_wd4', '2871', 'c9f28301391601e5e8ae93139431a9e0d467317c', 'v0.0.142'),
    ('igcv3_wd2', '1732', '8c504f443283d8a32787275b23771082fcaab61b', 'v0.0.132'),
    ('igcv3_w3d4', '1140', '63f43cf8d334111d55d06f2f9bf7e1e4871d162c', 'v0.0.207'),
    ('igcv3_w1', '0920', '12385791681f09adb3a08926c95471f332f538b6', 'v0.0.243'),
    ('mnasnet', '1174', 'e8ec017ca396dc7d39e03b383776b8cf9ad20a4d', 'v0.0.117'),
    ('darts', '0874', '74f0c7b690cf8bef9b54cc5afc2cb0f2a2a83630', 'v0.0.118'),
    ('proxylessnas_cpu', '0761', 'fe9572b11899395acbeef9374827dcc04e103ce3', 'v0.0.304'),
    ('proxylessnas_gpu', '0745', 'acca5941c454d896410060434b8f983d2db80727', 'v0.0.333'),
    ('proxylessnas_mobile', '0780', '639a90c27de088402db76b09e410326795b6fbdd', 'v0.0.304'),
    ('proxylessnas_mobile14', '0662', '0c0ad983f4fb88470d0f3e557d0b23f15e16624f', 'v0.0.331'),
    ('fbnet_cb', '0762', '2edb61f8e4b5c45d958d0e57beff41fbfacd6061', 'v0.0.415'),
    ('xception', '0549', 'e4f0232c99fa776e630189d62fea18e248a858b2', 'v0.0.115'),
    ('inceptionv3', '0565', 'cf4061800bc1dc3b090920fc9536d8ccc15bb86e', 'v0.0.92'),
    ('inceptionv4', '0529', '5cb7b4e4b8f62d6b4346855d696b06b426b44f3d', 'v0.0.105'),
    ('inceptionresnetv2', '0490', '1d1b4d184e6d41091c5ac3321d99fa554b498dbe', 'v0.0.107'),
    ('polynet', '0452', '6a1b295dad3f261b48e845f1b283e4eef3ab5a0b', 'v0.0.96'),
    ('nasnet_4a1056', '0816', 'd21bbaf5e937c2e06134fa40e7bdb1f501423b86', 'v0.0.97'),
    ('nasnet_6a4032', '0421', 'f354d28f4acdde399e081260c3f46152eca5d27e', 'v0.0.101'),
    ('pnasnet5large', '0428', '65de46ebd049e494c13958d5671aba5abf803ff3', 'v0.0.114'),
    ('spnasnet', '0817', '290a4fd99674b8f32d5143774ca0141f1b511733', 'v0.0.416'),
    ('efficientnet_b0', '0752', '0e3861300b8f1d1d0fb1bd15f0e06bba1ad6309b', 'v0.0.364'),
    ('efficientnet_b1', '0638', 'ac77bcd722dc4f3edfa24b9fb7b8f9cece3d85ab', 'v0.0.376'),
    ('efficientnet_b0b', '0702', 'ecf61b9b50666a6b444a9d789a5ff1087c65d0d8', 'v0.0.403'),
    ('efficientnet_b1b', '0594', '614e81663902850a738fa6c862fe406ecf205f73', 'v0.0.403'),
    ('efficientnet_b2b', '0527', '531f10e6898778b7c3a82c2c149f8b3e6393a892', 'v0.0.403'),
    ('efficientnet_b3b', '0445', '3c5fbba8c86121d4bc3bbc169804f24dd4c3d1f6', 'v0.0.403'),
    ('efficientnet_b4b', '0389', '6305bfe688b261f0d4fef6829f520d5c98c46301', 'v0.0.403'),
    ('efficientnet_b5b', '0337', 'e1c2ffcf710cbd3c53b9c08723282a370906731c', 'v0.0.403'),
    ('efficientnet_b6b', '0323', 'e5c1d7c35fcff5fac07921a7696f7c04aba84012', 'v0.0.403'),
    ('efficientnet_b7b', '0322', 'b9c5965a1e2572aaa772e20e8a2e3af7b4bee9a6', 'v0.0.403'),
    ('mixnet_s', '0719', 'aeafe8432c11ffafbe72b9456d0c040151a5465c', 'v0.0.412'),
    ('mixnet_m', '0660', '5aab9fbd5a1d53cca58cdab4e1c644cacb6e0d8c', 'v0.0.413'),
    ('mixnet_l', '0582', '6cf2c97538d4173d9f6bc80a6ec299463df2d1f3', 'v0.0.414'),
    ('resnetd50b', '0565', 'ec03d815c0f016c6517ed7b4b40126af46ceb8a4', 'v0.0.296'),
    ('resnetd101b', '0473', 'f851c920ec1fe4f729d339c933535d038bf2903c', 'v0.0.296'),
    ('resnetd152b', '0482', '112e216da50eb20d52c509a28c97b05ef819cefe', 'v0.0.296'),
    ('nin_cifar10', '0743', '795b082470b58c1aa94e2f861514b7914f6e2f58', 'v0.0.175'),
    ('nin_cifar100', '2839', '627a11c064eb44c6451fe53e0becfc21a6d57d7f', 'v0.0.183'),
    ('nin_svhn', '0376', '1205dc06a4847bece8159754033f325f75565c02', 'v0.0.270'),
    ('resnet20_cifar10', '0597', '9b0024ac4c2f374cde2c5052e0d0344a75871cdb', 'v0.0.163'),
    ('resnet20_cifar100', '2964', 'a5322afed92fa96cb7b3453106f73cf38e316151', 'v0.0.180'),
    ('resnet20_svhn', '0343', '8232e6e4c2c9fac1200386b68311c3bd56f483f5', 'v0.0.265'),
    ('resnet56_cifar10', '0452', '628c42a26fe347b84060136212e018df2bb35e0f', 'v0.0.163'),
    ('resnet56_cifar100', '2488', 'd65f53b10ad5d124698e728432844c65261c3107', 'v0.0.181'),
    ('resnet56_svhn', '0275', '6e08ed929b8f0ee649f75464f06b557089023290', 'v0.0.265'),
    ('resnet110_cifar10', '0369', '4d6ca1fc02eaeed724f4f596011e391528536049', 'v0.0.163'),
    ('resnet110_cifar100', '2280', 'd8d397a767db6d22af040223ec8ae342a088c3e5', 'v0.0.190'),
    ('resnet110_svhn', '0245', 'c971f0a38943d8a75386a60c835cc0843c2f6c1c', 'v0.0.265'),
    ('resnet164bn_cifar10', '0368', '74ae9f4bccb7fb6a8f3f603fdabe8d8632c46b2f', 'v0.0.179'),
    ('resnet164bn_cifar100', '2044', '8fa07b7264a075fa5add58f4c676b99a98fb1c89', 'v0.0.182'),
    ('resnet164bn_svhn', '0242', '549413723d787cf7e96903427a7a14fb3ea1a4c1', 'v0.0.267'),
    ('resnet272bn_cifar10', '0333', '84f28e0ca97eaeae0eb07e9f76054c1ba0c77c0e', 'v0.0.368'),
    ('resnet272bn_cifar100', '2007', 'a80d2b3ce14de6c90bf22d210d76ebd4a8c91928', 'v0.0.368'),
    ('resnet272bn_svhn', '0243', 'ab1d7da51f52cc6acb2e759736f2d58a77ce895e', 'v0.0.368'),
    ('resnet542bn_cifar10', '0343', '0fd36dd16587f49d33e0e36f1e8596d021a11439', 'v0.0.369'),
    ('resnet542bn_cifar100', '1932', 'a631d3ce5f12e145637a7b2faee663cddc94c354', 'v0.0.369'),
    ('resnet542bn_svhn', '0234', '04396c973121e356f2efda9a28c4e4086f1511b2', 'v0.0.369'),
    ('resnet1001_cifar10', '0328', '77a179e240808b7aa3534230d39b845a62413ca2', 'v0.0.201'),
    ('resnet1001_cifar100', '1979', '2728b558748f9c3e70db179afb6c62358020858b', 'v0.0.254'),
    ('resnet1001_svhn', '0241', '9e3d4bb55961db4c0f46a961b5323a4e03aea602', 'v0.0.408'),
    ('resnet1202_cifar10', '0353', '1d5a21290117903fb5fd6ba59f3f7e7da7c08836', 'v0.0.214'),
    ('resnet1202_cifar100', '2156', '86ecd091e5ac9677bf4518c644d08eb3e1d1708a', 'v0.0.410'),
    ('preresnet20_cifar10', '0651', '76cec68d11de5b25be2ea5935681645b76195f1d', 'v0.0.164'),
    ('preresnet20_cifar100', '3022', '3dbfa6a2b850572bccb28cc2477a0e46c24abcb8', 'v0.0.187'),
    ('preresnet20_svhn', '0322', 'c3c00fed49c1d6d9deda6436d041c5788d549299', 'v0.0.269'),
    ('preresnet56_cifar10', '0449', 'e9124fcf167d8ca50addef00c3afa4da9f828f29', 'v0.0.164'),
    ('preresnet56_cifar100', '2505', 'ca90a2be6002cd378769b9d4e7c497dd883d31d9', 'v0.0.188'),
    ('preresnet56_svhn', '0280', 'b51b41476710c0e2c941356ffe992ff883a3ee87', 'v0.0.269'),
    ('preresnet110_cifar10', '0386', 'cc08946a2126a1224d1d2560a47cf766a763c52c', 'v0.0.164'),
    ('preresnet110_cifar100', '2267', '3954e91581b7f3e5f689385d15f618fe16e995af', 'v0.0.191'),
    ('preresnet110_svhn', '0279', 'aa49e0a3c4a918e227ca2d5a5608704f026134c3', 'v0.0.269'),
    ('preresnet164bn_cifar10', '0364', '429012d412e82df7961fa071f97c938530e1b005', 'v0.0.196'),
    ('preresnet164bn_cifar100', '2018', 'a8e67ca6e14f88b009d618b0e9b554312d862174', 'v0.0.192'),
    ('preresnet164bn_svhn', '0258', '94d42de440d5f057a38f4c8cdbdb24acfee3981c', 'v0.0.269'),
    ('preresnet272bn_cifar10', '0325', '1a6a016eb4e4a5549c1fcb89ed5af4c1e5715b72', 'v0.0.389'),
    ('preresnet272bn_cifar100', '1963', '6fe0d2e24a60d12ab6b3d0e46065e2f14a46bc0b', 'v0.0.389'),
    ('preresnet272bn_svhn', '0234', 'c04ef5c20a53f76824339fe75185d181be4bce61', 'v0.0.389'),
    ('preresnet542bn_cifar10', '0314', '66fd6f2033dff08428e586bcce3e5151ed4274f9', 'v0.0.391'),
    ('preresnet542bn_cifar100', '1871', '07f1fb258207d295789981519e8dab892fc08f8d', 'v0.0.391'),
    ('preresnet542bn_svhn', '0236', '6bdf92368873ce1288526dc405f15e689a1d3117', 'v0.0.391'),
    ('preresnet1001_cifar10', '0265', '9fedfe5fd643e7355f1062a6db68da310c8962be', 'v0.0.209'),
    ('preresnet1001_cifar100', '1841', '88f14ed9df1573e98b0ec2a07009a15066855fda', 'v0.0.283'),
    ('preresnet1202_cifar10', '0339', '6fc686b02191226f39e25a76fc5da26857f7acd9', 'v0.0.246'),
    ('resnext29_32x4d_cifar10', '0315', '30413525cd4466dbef759294eda9b702bc39648f', 'v0.0.169'),
    ('resnext29_32x4d_cifar100', '1950', '13ba13d92f6751022549a3b370ae86d3b13ae2d1', 'v0.0.200'),
    ('resnext29_32x4d_svhn', '0280', 'e85c5217944cdfafb0a538dd7cc817cffaada7c4', 'v0.0.275'),
    ('resnext29_16x64d_cifar10', '0241', '4133d3d04f9b10b132dcb959601d36f10123f8c2', 'v0.0.176'),
    ('resnext29_16x64d_cifar100', '1693', '05e9a7f113099a98b219cad622ecfad5517a3b54', 'v0.0.322'),
    ('resnext29_16x64d_svhn', '0268', '74332b714cd278bfca3f09dafe2a9d117510e9a4', 'v0.0.358'),
    ('resnext272_1x64d_cifar10', '0255', '070ccc35c2841b7715b9eb271197c9bb316f3093', 'v0.0.372'),
    ('resnext272_1x64d_cifar100', '1911', '114eb0f8a0d471487e819b8fd156c1286ef91b7a', 'v0.0.372'),
    ('resnext272_1x64d_svhn', '0235', 'ab0448469bbd7d476f8bed1bf86403304b028e7c', 'v0.0.372'),
    ('resnext272_2x32d_cifar10', '0274', 'd2ace03c413be7e42c839c84db8dd0ebb5d69512', 'v0.0.375'),
    ('resnext272_2x32d_cifar100', '1834', '0b30c4701a719995412882409339f3553a54c9d1', 'v0.0.375'),
    ('resnext272_2x32d_svhn', '0244', '39b8a33612d335a0193b867b38c0b09d168de6c3', 'v0.0.375'),
    ('seresnet20_cifar10', '0601', '935d89433e803c8a3027c81f1267401e7caccce6', 'v0.0.362'),
    ('seresnet20_cifar100', '2854', '8c7abf66d8c1418cb3ca760f5d1efbb42738036b', 'v0.0.362'),
    ('seresnet20_svhn', '0323', 'd77df31c62d1504209a5ba47e59ccb0ae84500b2', 'v0.0.362'),
    ('seresnet56_cifar10', '0413', 'b61c143989cb2901bec48dded4c6ddcae91aabc4', 'v0.0.362'),
    ('seresnet56_cifar100', '2294', '7fa54f4593f364c2363cb3ee8d5bc1285af1ade5', 'v0.0.362'),
    ('seresnet56_svhn', '0264', '93839c762a97bd0b5bd27c71fd64c227afdae3ed', 'v0.0.362'),
    ('seresnet110_cifar10', '0363', '1ddec2309ff61c2c0e14c96d51a1b846afdc2acc', 'v0.0.362'),
    ('seresnet110_cifar100', '2086', 'a82c30938028a172dd6a124152bc0952b55a2f49', 'v0.0.362'),
    ('seresnet110_svhn', '0235', '9572ba7394c774b8d056b24a7631ef47e53024b8', 'v0.0.362'),
    ('seresnet164bn_cifar10', '0339', '1085dab6467cb18e554123663816094f080fc626', 'v0.0.362'),
    ('seresnet164bn_cifar100', '1995', '97dd4ab630f6277cf7b07cbdcbf4ae8ddce4d401', 'v0.0.362'),
    ('seresnet164bn_svhn', '0245', 'af0a90a50fb3c91eef039178a681e69aae703f3a', 'v0.0.362'),
    ('seresnet272bn_cifar10', '0339', '812db5187bab9aa5203611c1c174d0e51c81761c', 'v0.0.390'),
    ('seresnet272bn_cifar100', '1907', '179e1c38ba714e1babf6c764ca735f256d4cd122', 'v0.0.390'),
    ('seresnet272bn_svhn', '0238', '0e16badab35b483b1a1b0e7ea2a615de714f7424', 'v0.0.390'),
    ('seresnet542bn_cifar10', '0347', 'd1542214765f1923f2fdce810aef5dc2e523ffd2', 'v0.0.385'),
    ('seresnet542bn_cifar100', '1887', '9c4e7623dc06a56edabf04f4427286916843df85', 'v0.0.385'),
    ('seresnet542bn_svhn', '0226', '71a8f2986cbc1146f9a41d1a08ecba52649b8efd', 'v0.0.385'),
    ('sepreresnet20_cifar10', '0618', 'eabb3fce8373cbeb412ced9a79a1e2f9c6c3689c', 'v0.0.379'),
    ('sepreresnet20_cifar100', '2831', 'fe7558e0ae554d39d8761f234e8328262ee31efd', 'v0.0.379'),
    ('sepreresnet20_svhn', '0324', '061daa587dd483744d5b60d2fd3b2750130dd8a1', 'v0.0.379'),
    ('sepreresnet56_cifar10', '0451', 'fc23e153ccfaddd52de61d77570a0befeee1e687', 'v0.0.379'),
    ('sepreresnet56_cifar100', '2305', 'c4bdc5d7bbaa0d9f6e2ffdf2abe4808ad26d0f66', 'v0.0.379'),
    ('sepreresnet56_svhn', '0271', 'c91e922f1b3d0ea634db8e467e9ab4a6b8dc7722', 'v0.0.379'),
    ('sepreresnet110_cifar10', '0454', '418daea9d2253a3e9fbe4eb80eb4dcc6f29d5925', 'v0.0.379'),
    ('sepreresnet110_cifar100', '2261', 'ed7d3c3e51ed2ea9a827ed942e131c78784813b7', 'v0.0.379'),
    ('sepreresnet110_svhn', '0259', '556909fd942d3a42e424215374b340680b705424', 'v0.0.379'),
    ('sepreresnet164bn_cifar10', '0373', 'ff353a2910f85db66d8afca0a4150176bcdc7a69', 'v0.0.379'),
    ('sepreresnet164bn_cifar100', '2005', 'df1163c4d9de72c53efc37758773cc943be7f055', 'v0.0.379'),
    ('sepreresnet164bn_svhn', '0256', 'f8dd4e06596841f0c7f9979fb566b9e57611522f', 'v0.0.379'),
    ('sepreresnet272bn_cifar10', '0339', '606d096422394857cb1f45ecd7eed13508158a60', 'v0.0.379'),
    ('sepreresnet272bn_cifar100', '1913', 'cb71511346e441cbd36bacc93c821e8b6101456a', 'v0.0.379'),
    ('sepreresnet272bn_svhn', '0249', '904d74a2622d870f8a2384f9e50a84276218acc3', 'v0.0.379'),
    ('sepreresnet542bn_cifar10', '0308', '652bc8846cfac7a2ec6625789531897339800202', 'v0.0.382'),
    ('sepreresnet542bn_cifar100', '1945', '9180f8632657bb8f7b6583e47d04ce85defa956c', 'v0.0.382'),
    ('sepreresnet542bn_svhn', '0247', '318a8325afbfbaa8a35d54cbd1fa7da668ef1389', 'v0.0.382'),
    ('pyramidnet110_a48_cifar10', '0372', 'eb185645cda89e0c3c47b11c4b2d14ff18fa0ae1', 'v0.0.184'),
    ('pyramidnet110_a48_cifar100', '2095', '95da1a209916b3cf4af7e8dc44374345a88c60f4', 'v0.0.186'),
    ('pyramidnet110_a48_svhn', '0247', 'd48bafbebaabe9a68e5924571752b3d7cd95d311', 'v0.0.281'),
    ('pyramidnet110_a84_cifar10', '0298', '7b835a3cf19794478d478aced63ca9e855c3ffeb', 'v0.0.185'),
    ('pyramidnet110_a84_cifar100', '1887', 'ff711084381f217f84646c676e4dcc90269dc516', 'v0.0.199'),
    ('pyramidnet110_a84_svhn', '0243', '971576c61cf30e02f13da616afc9848b2a609e0e', 'v0.0.392'),
    ('pyramidnet110_a270_cifar10', '0251', '31bdd9d51ec01388cbb2adfb9f822c942de3c4ff', 'v0.0.194'),
    ('pyramidnet110_a270_cifar100', '1710', '7417dd99069d6c8775454475968ae226b9d5ac83', 'v0.0.319'),
    ('pyramidnet110_a270_svhn', '0238', '3047a9bb7c92a09adf31590e3fe6c9bcd36c7a67', 'v0.0.393'),
    ('pyramidnet164_a270_bn_cifar10', '0242', 'daa2a402c1081323b8f2239f2201246953774e84', 'v0.0.264'),
    ('pyramidnet164_a270_bn_cifar100', '1670', '54d99c834bee0ed7402ba46e749e48182ad1599a', 'v0.0.312'),
    ('pyramidnet164_a270_bn_svhn', '0233', '42d4c03374f32645924fc091d599ef7b913e2d32', 'v0.0.396'),
    ('pyramidnet200_a240_bn_cifar10', '0244', '44433afdd2bc32c55dfb1e8347bc44d1c2bf82c7', 'v0.0.268'),
    ('pyramidnet200_a240_bn_cifar100', '1609', '087c02d6882e274054f44482060f193b9fc208bb', 'v0.0.317'),
    ('pyramidnet200_a240_bn_svhn', '0232', 'f9660c25f1bcff9d361aeca8fb3efaccdc0546e7', 'v0.0.397'),
    ('pyramidnet236_a220_bn_cifar10', '0247', 'daa91d74979c451ecdd8b59e4350382966f25831', 'v0.0.285'),
    ('pyramidnet236_a220_bn_cifar100', '1634', 'a45816ebe1d6a67468b78b7a93334a41aca1c64b', 'v0.0.312'),
    ('pyramidnet236_a220_bn_svhn', '0235', 'f74fe248b6189699174c90bc21e7949d3cca8130', 'v0.0.398'),
    ('pyramidnet272_a200_bn_cifar10', '0239', '586b1ecdc8b34b69dcae4ba57f71c24583cca9b1', 'v0.0.284'),
    ('pyramidnet272_a200_bn_cifar100', '1619', '98bc2f48da0f2c68bc5376c17b0aefc734a64881', 'v0.0.312'),
    ('pyramidnet272_a200_bn_svhn', '0240', '96f6e740dcdc917d776f6df855e3437c93d0da4f', 'v0.0.404'),
    ('densenet40_k12_cifar10', '0561', '8b8e819467a2e4c450e4ff72ced80582d0628b68', 'v0.0.193'),
    ('densenet40_k12_cifar100', '2490', 'd182c224d6df2e289eef944d54fea9fd04890961', 'v0.0.195'),
    ('densenet40_k12_svhn', '0305', 'ac0de84a1a905b768c66f0360f1fb9bd918833bf', 'v0.0.278'),
    ('densenet40_k12_bc_cifar10', '0643', '6dc86a2ea1d088f088462f5cbac06cc0f37348c0', 'v0.0.231'),
    ('densenet40_k12_bc_cifar100', '2841', '1e9db7651a21e807c363c9f366bd9e91ce2f296f', 'v0.0.232'),
    ('densenet40_k12_bc_svhn', '0320', '320760528b009864c68ff6c5b874e9f351ea7a07', 'v0.0.279'),
    ('densenet40_k24_bc_cifar10', '0452', '669c525548a4a2392c5e3c380936ad019f2be7f9', 'v0.0.220'),
    ('densenet40_k24_bc_cifar100', '2267', '411719c0177abf58eddaddd05511c86db0c9d548', 'v0.0.221'),
    ('densenet40_k24_bc_svhn', '0290', 'f4440d3b8c974c9e1014969f4d5832c6c90195d5', 'v0.0.280'),
    ('densenet40_k36_bc_cifar10', '0404', 'b1a4cc7e67db1ed8c5583a59dc178cc7dc2c572e', 'v0.0.224'),
    ('densenet40_k36_bc_cifar100', '2050', 'cde836fafec1e5d6c8ed69fd3cfe322e8e71ef1d', 'v0.0.225'),
    ('densenet40_k36_bc_svhn', '0260', '8c7db0a291a0797a8bc3c709bff7917bc41471cc', 'v0.0.311'),
    ('densenet100_k12_cifar10', '0366', '26089c6e70236e8f25359de6fda67b84425888ab', 'v0.0.205'),
    ('densenet100_k12_cifar100', '1964', '5e10cd830c06f6ab178e9dd876c83c754ca63f00', 'v0.0.206'),
    ('densenet100_k12_svhn', '0260', '57fde50e9f44edc0486b62a1144565bc77d5bdfe', 'v0.0.311'),
    ('densenet100_k24_cifar10', '0313', '397f0e39b517c05330221d4f3a9755eb5e561be1', 'v0.0.252'),
    ('densenet100_k24_cifar100', '1808', '1c0a8067283952709d8e09c774c3a404f51e0079', 'v0.0.318'),
    ('densenet100_k12_bc_cifar10', '0416', 'b9232829b13c3f3f2ea15f4be97f500b7912c3c2', 'v0.0.189'),
    ('densenet100_k12_bc_cifar100', '2119', '05a6f02772afda51a612f5b92aadf19ffb60eb72', 'v0.0.208'),
    ('densenet190_k40_bc_cifar10', '0252', '2896fa088aeaef36fcf395d404d97ff172d78943', 'v0.0.286'),
    ('densenet250_k24_bc_cifar10', '0267', 'f8f9d3052bae1fea7e33bb1ce143c38b4aa5622b', 'v0.0.290'),
    ('densenet250_k24_bc_cifar100', '1739', '09ac3e7d9fbe6b97b170bd838dac20ec144b4e49', 'v0.0.303'),
    ('xdensenet40_2_k24_bc_cifar10', '0531', 'b91a9dc35877c4285fe86f49953d1118f6b69e57', 'v0.0.226'),
    ('xdensenet40_2_k24_bc_cifar100', '2396', '0ce8f78ab9c6a4786829f816ae0615c7905f292c', 'v0.0.227'),
    ('xdensenet40_2_k24_bc_svhn', '0287', 'fd9b6def10f154378a76383cf023d7f2f5ae02ab', 'v0.0.306'),
    ('xdensenet40_2_k36_bc_cifar10', '0437', 'ed264a2060836c7440f0ccde57315e1ec6263ff0', 'v0.0.233'),
    ('xdensenet40_2_k36_bc_cifar100', '2165', '6f68f83dc31dea5237e6362e6c6cfeed48a8d9e3', 'v0.0.234'),
    ('xdensenet40_2_k36_bc_svhn', '0274', '540a69f13a6ce70bfef13657e70dfa414d966581', 'v0.0.306'),
    ('wrn16_10_cifar10', '0293', 'ce810d8a17a2deb73eddb5bec8709f93278bc53e', 'v0.0.166'),
    ('wrn16_10_cifar100', '1895', 'bef9809c845deb1b2bb0c9aaaa7c58bd97740504', 'v0.0.204'),
    ('wrn16_10_svhn', '0278', '5ab2a4edd5398a03d2e28db1b075bf0313ae5828', 'v0.0.271'),
    ('wrn28_10_cifar10', '0239', 'fe97dcd6d0dd8dda8e9e38e6cfa320cffb9955ce', 'v0.0.166'),
    ('wrn28_10_cifar100', '1788', '8c3fe8185d3af9cc3813fe376cab895f6780ac18', 'v0.0.320'),
    ('wrn28_10_svhn', '0271', 'd62b6bbaef7228706a67c2c8416681f97c6d4688', 'v0.0.276'),
    ('wrn40_8_cifar10', '0237', '8dc84ec730f35c4b8968a022bc045c0665410840', 'v0.0.166'),
    ('wrn40_8_cifar100', '1803', '0d18bfbff85951d88a881dc6a15ad46f56ea8c28', 'v0.0.321'),
    ('wrn40_8_svhn', '0254', 'dee59602c10e5d56bd9c168e8e8400792b9a8b08', 'v0.0.277'),
    ('wrn20_10_1bit_cifar10', '0326', 'e6140f8a5eacd5227e8748457b5ee9f5f519d2d5', 'v0.0.302'),
    ('wrn20_10_1bit_cifar100', '1904', '149860c829a812224dbf2086c8ce95c2eba322fe', 'v0.0.302'),
    ('wrn20_10_1bit_svhn', '0273', 'ffe96cb78cd304d5207fff0cf08835ba2a71f666', 'v0.0.302'),
    ('wrn20_10_32bit_cifar10', '0314', 'a18146e8b0f99a900c588eb8995547393c2d9d9e', 'v0.0.302'),
    ('wrn20_10_32bit_cifar100', '1812', '70d8972c7455297bc21fdbe4fc040c2f6b3593a3', 'v0.0.302'),
    ('wrn20_10_32bit_svhn', '0259', 'ce402a58887cbae3a38da1e845a1c1479a6d7213', 'v0.0.302'),
    ('ror3_56_cifar10', '0543', '44f0f47d2e1b609880ee1b623014c52a9276e2ea', 'v0.0.228'),
    ('ror3_56_cifar100', '2549', '34be6719cd128cfe60ba93ac6d250ac4c1acf0a5', 'v0.0.229'),
    ('ror3_56_svhn', '0269', '5a9ad66c8747151be1d2fb9bc854ae382039bdb9', 'v0.0.287'),
    ('ror3_110_cifar10', '0435', 'fb2a2b0499e4a4d92bdc1d6792bd5572256d5165', 'v0.0.235'),
    ('ror3_110_cifar100', '2364', 'd599e3a93cd960c8bfc5d05c721cd48fece5fa6f', 'v0.0.236'),
    ('ror3_110_svhn', '0257', '155380add8d351d2c12026d886a918f1fc3f9fd0', 'v0.0.287'),
    ('ror3_164_cifar10', '0393', 'de7b6dc60ad6a297bd55ab65b6d7b1225b0ef6d1', 'v0.0.294'),
    ('ror3_164_cifar100', '2234', 'd37483fccc7fc1a25ff90ef05ecf1b8eab3cc1c4', 'v0.0.294'),
    ('ror3_164_svhn', '0273', 'ff0d9af0d40ef204393ecc904b01a11aa63acc01', 'v0.0.294'),
    ('rir_cifar10', '0328', '414c3e6088ae1e83aa1a77c43e38f940c18a0ce2', 'v0.0.292'),
    ('rir_cifar100', '1923', 'de8ec24a232b94be88f4208153441f66098a681c', 'v0.0.292'),
    ('rir_svhn', '0268', '12fcbd3bfc6b4165e9b23f3339a1b751b4b8f681', 'v0.0.292'),
    ('shakeshakeresnet20_2x16d_cifar10', '0515', 'ef71ec0d5ef928ef8654294114a013895abe3f9a', 'v0.0.215'),
    ('shakeshakeresnet20_2x16d_cifar100', '2922', '4d07f14234b1c796b3c1dfb24d4a3220a1b6b293', 'v0.0.247'),
    ('shakeshakeresnet20_2x16d_svhn', '0317', 'a693ec24fb8fe2c9f15bcc6b1050943c0c5d595a', 'v0.0.295'),
    ('shakeshakeresnet26_2x32d_cifar10', '0317', 'ecd1f8337cc90b5378b4217fb2591f2ed0f02bdf', 'v0.0.217'),
    ('shakeshakeresnet26_2x32d_cifar100', '1880', 'b47e371f60c9fed9eaac960568783fb6f83a362f', 'v0.0.222'),
    ('shakeshakeresnet26_2x32d_svhn', '0262', 'c1b8099ece97e17ce85213e4ecc6e50a064050cf', 'v0.0.295'),
    ('diaresnet20_cifar10', '0622', '5e1a02bf2347d48651a5feabe97f7caf215bacc9', 'v0.0.340'),
    ('diaresnet20_cifar100', '2771', '28aa1a18d91334e274d3157114fc5c72e47c6c65', 'v0.0.342'),
    ('diaresnet20_svhn', '0323', 'b8ee92c9d86de6a6adc80988518fe0544759ca4f', 'v0.0.342'),
    ('diaresnet56_cifar10', '0505', '8ac8680448b2999bd1e03eed60373ea78eba9a44', 'v0.0.340'),
    ('diaresnet56_cifar100', '2435', '19085975afc7ee902a6d663eb371554c9519b467', 'v0.0.342'),
    ('diaresnet56_svhn', '0268', 'bd2ec7558697aff1e0fd229d3e933a08c4c302e9', 'v0.0.342'),
    ('diaresnet110_cifar10', '0410', '0c00a7daec69b57ab41d4a55e1026da33ecf4539', 'v0.0.340'),
    ('diaresnet110_cifar100', '2211', '7096ddb3a393ad28b27ece19263c203068a11b6d', 'v0.0.342'),
    ('diaresnet110_svhn', '0247', '635e42cfac6ed67e15b8a5526c8232f768d11201', 'v0.0.342'),
    ('diaresnet164bn_cifar10', '0350', 'd31f2ebce3acb419b07dc4d298018ffea2599fea', 'v0.0.340'),
    ('diaresnet164bn_cifar100', '1953', 'b1c474d27de3a291a45856a3e3d256b7fda90dd0', 'v0.0.342'),
    ('diaresnet164bn_svhn', '0244', '0b8f67132b3911e6328733b666bf6a0fed133eeb', 'v0.0.342'),
    ('diapreresnet20_cifar10', '0642', '14a1eb85c6346c81336b490cc49f2e6b809c193e', 'v0.0.343'),
    ('diapreresnet20_cifar100', '2837', 'f7675c09ca5f742376a102e3c8c5156aea4e24b9', 'v0.0.343'),
    ('diapreresnet20_svhn', '0303', 'dc3e3a453ffc8aff7d014bc15867db4ce2d8e1e9', 'v0.0.343'),
    ('diapreresnet56_cifar10', '0483', '41cae958be1bec3f839126cd167051de6a981d0a', 'v0.0.343'),
    ('diapreresnet56_cifar100', '2505', '5d357985236c021ab965101b94980cdc4722a70d', 'v0.0.343'),
    ('diapreresnet56_svhn', '0280', '537ebc66fe32f9bb6fb6bb8f9ac6402f8ec93e09', 'v0.0.343'),
    ('diapreresnet110_cifar10', '0425', '5638501600355b8b195179fb2be5d5989e93b0e0', 'v0.0.343'),
    ('diapreresnet110_cifar100', '2269', 'c993cc296c39bc9c8c0fc6115bfe6c7d720a0903', 'v0.0.343'),
    ('diapreresnet110_svhn', '0242', 'a156cfb58ffda89c0e87cd8aef82f56f79b40ea5', 'v0.0.343'),
    ('diapreresnet164bn_cifar10', '0356', '6ec898c89c66eb32b0e42b78a027af4920b24366', 'v0.0.343'),
    ('diapreresnet164bn_cifar100', '1999', '00872f989c33321f7938a40c0fd9f44669c4c483', 'v0.0.343'),
    ('diapreresnet164bn_svhn', '0256', '134048810bd2e12dc68035d4ecad6af525639db0', 'v0.0.343'),
    ('resnet10_cub', '2777', '4525b5932665698b3f4551dde99d22ce03878172', 'v0.0.335'),
    ('resnet12_cub', '2727', 'c15248832d2fe88c58fb603df3925e09b3d797e7', 'v0.0.336'),
    ('resnet14_cub', '2477', '5051bbc659c0303c1860114f1a32a18942de9099', 'v0.0.337'),
    ('resnet16_cub', '2365', 'b831356c696db80fec8deb2381875f37bf60dd93', 'v0.0.338'),
    ('resnet18_cub', '2333', '200d8b9c48baf073a4c2ea0cbba4d7f81288e684', 'v0.0.344'),
    ('resnet26_cub', '2316', '599ab467f396e979028f2ae5d65330949c9ddc86', 'v0.0.345'),
    ('seresnet10_cub', '2772', 'f52526ec21bbb534a6d51be42bdb5322fbda919b', 'v0.0.361'),
    ('seresnet12_cub', '2651', '5c0e7f835c65d1f2f85048d0169788377490b819', 'v0.0.361'),
    ('seresnet14_cub', '2416', 'a4cda9012ec2380fa74f3d74879f0d206fcaf5b5', 'v0.0.361'),
    ('seresnet16_cub', '2332', '43a819b7e226d65aa77a4c90fdb7c70eb5093505', 'v0.0.361'),
    ('seresnet18_cub', '2352', '414fa2775de28ce3a1a0bc142ab674fa3a6638e3', 'v0.0.361'),
    ('seresnet26_cub', '2299', '5aa0a7d1ef9c33f8dbf3ff1cb1a1a855627163f4', 'v0.0.361'),
    ('mobilenet_w1_cub', '2377', '8428471f4ae08709b71ff2f69cf1a6fd286004c9', 'v0.0.346'),
    ('proxylessnas_mobile_cub', '2266', 'e4b5098a17425c97740fc564460aa95d9eb2a41e', 'v0.0.347'),
    ('ntsnet_cub', '1277', 'f6f330abfabcc2ea17a8d4b8977a6ea322ddf532', 'v0.0.334'),
    ('pspnet_resnetd101b_voc', '8144', 'c22f021948461a7b7ab1ef1265a7948762770c83', 'v0.0.297'),
    ('pspnet_resnetd50b_ade20k', '3687', '13f22137d7dd06c6de2ffc47e6ed33403d3dd2cf', 'v0.0.297'),
    ('pspnet_resnetd101b_ade20k', '3797', '115d62bf66477221b83337208aefe0f2f0266da2', 'v0.0.297'),
    ('pspnet_resnetd101b_cityscapes', '7172', '0a6efb497bd4fc763d27e2121211e06f72ada7ed', 'v0.0.297'),
    ('pspnet_resnetd101b_coco', '6741', 'c8b13be65cb43402fce8bae945f6e0d0a3246b92', 'v0.0.297'),
    ('deeplabv3_resnetd101b_voc', '8024', 'fd8bf74ffc96c97b30bcd3b6ce194a2daed68098', 'v0.0.298'),
    ('deeplabv3_resnetd152b_voc', '8120', 'f2dae198b3cdc41920ea04f674b665987c68d7dc', 'v0.0.298'),
    ('deeplabv3_resnetd50b_ade20k', '3713', 'bddbb458e362e18f5812c2307b322840394314bc', 'v0.0.298'),
    ('deeplabv3_resnetd101b_ade20k', '3784', '977446a5fb32b33f168f2240fb6b7ef9f561fc1e', 'v0.0.298'),
    ('deeplabv3_resnetd101b_coco', '6773', 'e59c1d8f7ed5bcb83f927d2820580a2f4970e46f', 'v0.0.298'),
    ('deeplabv3_resnetd152b_coco', '6899', '7e946d7a63ed255dd38afacebb0a0525e735da64', 'v0.0.298'),
    ('fcn8sd_resnetd101b_voc', '8040', '66edc0b073f0dec66c18bb163c7d6de1ddbc32a3', 'v0.0.299'),
    ('fcn8sd_resnetd50b_ade20k', '3339', 'e1dad8a15c2a1be1138bd3ec51ba1b100bb8d9c9', 'v0.0.299'),
    ('fcn8sd_resnetd101b_ade20k', '3588', '30d05ca42392a164ea7c93a9cbd7f33911d3c1af', 'v0.0.299'),
    ('fcn8sd_resnetd101b_coco', '6011', 'ebe2ad0bc1de5b4cecade61d17d269aa8bf6df7f', 'v0.0.299'),
]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.

    Returns
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

    Parameters
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
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

    Parameters
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

    Parameters
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

    Parameters
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
