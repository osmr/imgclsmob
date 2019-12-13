"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '1789', 'ecc4bb4e46e05dde17809978d2900f4fe14ea590', 'v0.0.422'),
    ('alexnetb', '1859', '9e390537e070ee42c5deeb6c456f81c991efbb49', 'v0.0.422'),
    ('zfnet', '1717', '9500db3008e9ca8bc8f8de8101ec760e5ac8c05a', 'v0.0.422'),
    ('zfnetb', '1480', '47533f6a367312c8b2f56202aeae0be366013116', 'v0.0.422'),
    ('vgg11', '1017', 'c20556f4179e9311f28baa310702b6ea9265fee8', 'v0.0.422'),
    ('vgg13', '0951', '9fa609fcb5cb44caf2737d13c0accc07cdea0c9d', 'v0.0.422'),
    ('vgg16', '0834', 'ce78831f5d0640bd2fd619ba7d8d5027e62eb4f2', 'v0.0.422'),
    ('vgg19', '0768', 'ec5ac0baa5d49c041af48e67d34d1a89f1a72e7f', 'v0.0.422'),
    ('bn_vgg11', '0936', 'ef31b86687e83d413cb9c95c9ead657c3de9f21b', 'v0.0.422'),
    ('bn_vgg13', '0887', '2cccc7252ab4798fd9a6c3ce9d0b59717c47e40b', 'v0.0.422'),
    ('bn_vgg16', '0759', '1ca9dee8ef41ed84a216636d3c21380988ea1bf8', 'v0.0.422'),
    ('bn_vgg19', '0688', '81d25be84932c1c2848cabd4533423e3fd2cdbec', 'v0.0.422'),
    ('bn_vgg11b', '0975', 'aeaccfdc4a655d895e280165cf5be856472ca91f', 'v0.0.422'),
    ('bn_vgg13b', '1019', '1102ffb7817ff11a8db85f1b9b8519b100da26a0', 'v0.0.422'),
    ('bn_vgg16b', '0862', '137178f78ace3943333a98d980dd88b4746e66af', 'v0.0.422'),
    ('bn_vgg19b', '0817', 'cd68a741183cbbab52562c4b7330d721e8ffa739', 'v0.0.422'),
    ('resnet10', '1390', '9e787f637312e04d3ec85136bf0ceca50acf8c80', 'v0.0.422'),
    ('resnet12', '1301', '8bc41d1b1da87463857bb5ca03fe252ef03116ad', 'v0.0.422'),
    ('resnet14', '1224', '7573d98872e622ef74e036c8a436a39ab75e9378', 'v0.0.422'),
    ('resnetbc14b', '1115', '5f30b7985b5a57d34909f3db08c52dfe1da065ac', 'v0.0.422'),
    ('resnet16', '1088', '14ce0d64680c3fe52f43b407a00d1a23b6cfd81c', 'v0.0.422'),
    ('resnet18_wd4', '1745', '6e80041645de7ccbe156ce5bc3cbde909cee6b41', 'v0.0.422'),
    ('resnet18_wd2', '1283', '85a7caff1b2f8e355a1b8cb559e836d5b0c22d12', 'v0.0.422'),
    ('resnet18_w3d4', '1067', 'c1735b7de29016779c95e8e1481e5ded955b2b63', 'v0.0.422'),
    ('resnet18', '0956', '6645845a7614afd265e997223d38e00433f00182', 'v0.0.422'),
    ('resnet26', '0837', 'a8f20f7194cdfcb6fd514a8dc9546105fd7a562a', 'v0.0.422'),
    ('resnetbc26b', '0757', 'd70a2cadfb648f4c528704f1b9983f35af94de6f', 'v0.0.422'),
    ('resnet34', '0744', '7f7d70e7780e24b4cb60cefc895198cdb2b94665', 'v0.0.422'),
    ('resnetbc38b', '0677', '75e405a71f7227de5abb6a3c3c44d807b5963c44', 'v0.0.422'),
    ('resnet50', '0604', '728800bf57bd49f79671399fd4fd2b7fe9883f07', 'v0.0.422'),
    ('resnet50b', '0614', 'b2a49da61dce6309c75e77226bb047b43247da24', 'v0.0.422'),
    ('resnet101', '0601', 'b6befeb4c8cf6d72d9c325c22df72ac792b51706', 'v0.0.422'),
    ('resnet101b', '0511', 'e3076227a06b394aebcce6260c4afc665224c987', 'v0.0.422'),
    ('resnet152', '0534', '2d8e394abcb9d35d2a853bb4dacb58460ff13551', 'v0.0.422'),
    ('resnet152b', '0480', 'b77f1e2c9158cc49deba2cf60b8a8e8d6605d654', 'v0.0.422'),
    ('preresnet10', '1402', '541bf0e17a576b1676069563a1ed0de0fde4090f', 'v0.0.422'),
    ('preresnet12', '1320', '349c0df4a835699bdb045bedc3d38a7747cd21d4', 'v0.0.422'),
    ('preresnet14', '1224', '194b876203e467fbad2ccd2e03b90a79bfec8dac', 'v0.0.422'),
    ('preresnetbc14b', '1152', 'bc4e06ff3df99e7ffa0b2bdafa224796fa46f5a9', 'v0.0.422'),
    ('preresnet16', '1080', 'e00c40ee6d211f553bff0274771e5461150c69f4', 'v0.0.422'),
    ('preresnet18_wd4', '1780', '6ac7bc592983ced18c863f203db80bbd30e87a0b', 'v0.0.422'),
    ('preresnet18_wd2', '1314', '0c0528c8ae4943aa68ba0298209f2ed418e4f644', 'v0.0.422'),
    ('preresnet18_w3d4', '1070', '056b46c6e8ee2c86ebee560efea81dd43bbd5de6', 'v0.0.422'),
    ('preresnet18', '0955', '621ead9297b93673ec1c040e091efff9142313b5', 'v0.0.422'),
    ('preresnet26', '0837', '1a92a73217b1611c27b0c7082a018328264a65ff', 'v0.0.422'),
    ('preresnetbc26b', '0788', '1f737cd6c173ed8e5d9a8a69b35e1cf696ba622e', 'v0.0.422'),
    ('preresnet34', '0754', '3cc5ae1481512a8b206fb96ac8b632bcc5ee2db9', 'v0.0.422'),
    ('preresnetbc38b', '0636', '3396b49b5d20e7d362f9bd8879c00a21e8d67df1', 'v0.0.422'),
    ('preresnet50', '0625', '208605629d347a64b9a354f5ad7f441f736eb418', 'v0.0.422'),
    ('preresnet50b', '0634', '711227b1a93dd721dd3e37709456acfde969ba18', 'v0.0.422'),
    ('preresnet101', '0573', 'd45ea488f72fb99af1c46e4064b12c5014a7b626', 'v0.0.422'),
    ('preresnet101b', '0539', '54d23aff956752be614c2ba66d8bff5477cf0367', 'v0.0.422'),
    ('preresnet152', '0532', '0ad4b58f2365028db9216f1e080898284328cc3e', 'v0.0.422'),
    ('preresnet152b', '0500', '119062d97d30f6636905c824c6d1b4e21be2c3f2', 'v0.0.422'),
    ('preresnet200b', '0563', '2f9c761d78714c33d3b260add782e3851b0078f4', 'v0.0.422'),
    ('preresnet269b', '0557', '7003b3c4a1dea496f915750b4411cc67042a111d', 'v0.0.422'),
    ('resnext14_16x4d', '1222', 'bff90c1d3dbde7ea4a6972bbacb619e252d344ea', 'v0.0.422'),
    ('resnext14_32x2d', '1247', '06aa6709cfb4cf23793eb0eee5d5fce42cfcb9cb', 'v0.0.422'),
    ('resnext14_32x4d', '1115', '3acdaec14a6c74284c03bc79ed47e9ecb394e652', 'v0.0.422'),
    ('resnext26_32x2d', '0851', '827791ccefaef07e5837f8fb1dae8733c871c029', 'v0.0.422'),
    ('resnext26_32x4d', '0718', '4f05525e34b9aeb82db2339f714b25055d94657b', 'v0.0.422'),
    ('resnext50_32x4d', '0547', '45234d14f0e80700afc5c61e1bd148d848d8d089', 'v0.0.422'),
    ('resnext101_32x4d', '0494', '3990ddd1e776c7e90625db9a8f683e1d6a6fb301', 'v0.0.422'),
    ('resnext101_64x4d', '0484', 'f8cf1580943cf3c6d6019f2fcc44f8adb857cb20', 'v0.0.422'),
    ('seresnet10', '1332', '33a592e1497d37a427920c1408be908ba28d2a6d', 'v0.0.422'),
    ('seresnet18', '0921', '46c847abfdbd82c41a096e385163f21ae29ee200', 'v0.0.422'),
    ('seresnet26', '0807', '5178b3b1ea71bb118ffcc5d471f782f4ae6150d4', 'v0.0.422'),
    ('seresnetbc26b', '0684', '1460a381603c880f24fb0a42bfb6b79b850e2b28', 'v0.0.422'),
    ('seresnetbc38b', '0575', '18fcfcc1fee078382ad957e0f7d139ff596732e7', 'v0.0.422'),
    ('seresnet50', '0642', '21b366af438527aa7667e3d89433ced7cac997ac', 'v0.0.422'),
    ('seresnet50b', '0533', '256002c3b489d5b685ee1ab6b62303d7768c5816', 'v0.0.422'),
    ('seresnet101', '0589', '2a22ba87f5b0d56d51063898161d4c42cac45325', 'v0.0.422'),
    ('seresnet152', '0576', '8023259a13a53aa0a72d9df6468721314e702872', 'v0.0.422'),
    ('sepreresnet10', '1309', 'af20d06c486dc97cff0f6d9bc52a7c7458040514', 'v0.0.422'),
    ('sepreresnet18', '0940', 'fe403280f68a5dfa93366437b9ff37ce3a419cf8', 'v0.0.422'),
    ('sepreresnetbc26b', '0640', 'a72bf8765efb1024bdd33eebe9920fd3e22d0bd6', 'v0.0.422'),
    ('sepreresnetbc38b', '0567', '17d10c63f096db1b7bfb59b6c6ffe14b9c669676', 'v0.0.422'),
    ('seresnext50_32x4d', '0509', '4244900a583098a5fb6c174c834f44a7471305c2', 'v0.0.422'),
    ('seresnext101_32x4d', '0459', '13a9b2fd699a3e25ee18d93a408dbaf3dee74428', 'v0.0.422'),
    ('seresnext101_64x4d', '0465', 'ec0a3b132256c8a7d0f92c45775d201a456f25fb', 'v0.0.422'),
    ('senet16', '0805', 'f5f576568d02a572be5276b0b64e71ce4d1c4531', 'v0.0.422'),
    ('senet28', '0590', '667d56873564cc22b2f10478d5f3d55cda580c61', 'v0.0.422'),
    ('senet154', '0466', 'f1b79a9bf0f7073bacf534d846c03d1b71dc404b', 'v0.0.422'),
    ('densenet121', '0684', 'e9196a9c93534ca7b71ef136e5cc27f240370481', 'v0.0.422'),
    ('densenet161', '0618', 'e77cf292bfd791403f5ab5b16eeedd900ed91d17', 'v0.0.422'),
    ('densenet169', '0606', 'f708dc3310008e59814745ffc22ddf829fb2d25a', 'v0.0.422'),
    ('densenet201', '0637', 'f45e9450de86d67733a0fd3dce871343c9ca6e23', 'v0.0.422'),
    ('darknet_tiny', '1745', 'd30be41aad15edf40dfed0bbf53d0e68c520f9f3', 'v0.0.422'),
    ('darknet_ref', '1671', 'b4991f6b58ae95118aa9ea84cae4a27e328196b5', 'v0.0.422'),
    ('darknet53', '0558', '4a63ab3005e5138445da5fac4247c460de02a41b', 'v0.0.422'),
    ('squeezenet_v1_0', '1760', 'd13ba73265325f21eb34e782989a7269cad406c6', 'v0.0.422'),
    ('squeezenet_v1_1', '1742', '95b614487f1f0572bd0dba18e0fc6d63df3a6bfc', 'v0.0.422'),
    ('squeezeresnet_v1_0', '1783', 'db620d998257c84fd6d5e80bba48cc1022febda3', 'v0.0.422'),
    ('squeezeresnet_v1_1', '1789', '13d6bc6bd85adf83ef55325443495feb07c5788f', 'v0.0.422'),
    ('sqnxt23_w1', '1861', '379975ebe54b180f52349c3737b17ea7b2613953', 'v0.0.422'),
    ('sqnxt23v5_w1', '1762', '153b4ce73714d2ecdca294efb365ab9c026e2f41', 'v0.0.422'),
    ('sqnxt23_w3d2', '1334', 'a2ba956cfeed0b4bbfc37776c6a1cd5ca13d9345', 'v0.0.422'),
    ('sqnxt23v5_w3d2', '1284', '72efaa710f0f1645cb220cb9950b3660299f2bed', 'v0.0.422'),
    ('sqnxt23_w2', '1069', 'f43dee19c527460f9815fc4e5eeeaef99fae4df3', 'v0.0.422'),
    ('sqnxt23v5_w2', '1026', 'da80c6407a4c18be31bcdd08356666942a9ef2b4', 'v0.0.422'),
    ('shufflenet_g1_wd4', '3681', '04a9e2d4ada22b3d317e2fc8b7d4ec11865c414f', 'v0.0.422'),
    ('shufflenet_g3_wd4', '3618', 'c9aad0f08d129726bbc19219c9773b38cf38825e', 'v0.0.422'),
    ('shufflenet_g1_wd2', '2236', '082db702c422d8bce12d4d79228de56f088a420d', 'v0.0.422'),
    ('shufflenet_g3_wd2', '2059', 'e3aefeeb36c20e325d0c7fe46afc60484167609d', 'v0.0.422'),
    ('shufflenet_g1_w3d4', '1679', 'a1cc5da3a288299a33353f697ed0297328dc3e95', 'v0.0.422'),
    ('shufflenet_g3_w3d4', '1611', '89546a05f499f0fdf96dade0f3db430f92c5920d', 'v0.0.422'),
    ('shufflenet_g1_w1', '1348', '52ddb20fd7ff288ae30a17757efda4653c09d5ca', 'v0.0.422'),
    ('shufflenet_g2_w1', '1333', '2a8ba6928e6fac05a5fe8911a9a175268eb18382', 'v0.0.422'),
    ('shufflenet_g3_w1', '1326', 'daaec8b84572023c1352e11830d296724123408e', 'v0.0.422'),
    ('shufflenet_g4_w1', '1313', '35dbd6b9fb8bc3e97367ea210abbd61da407f226', 'v0.0.422'),
    ('shufflenet_g8_w1', '1322', '449fb27659101a2cf0a87c90e33f4632d1c5e9f2', 'v0.0.422'),
    ('shufflenetv2_wd2', '1843', 'd492d721d3167cd64ab1c2a1f33f3ca5f6dec7c3', 'v0.0.422'),
    ('shufflenetv2_w1', '1135', 'dae13ee9f24c89cd1ea12a58fb90b967223c8e2e', 'v0.0.422'),
    ('shufflenetv2_w3d2', '0923', 'ea615baab737fca3a3d90303844b4a2922ea2c62', 'v0.0.422'),
    ('shufflenetv2_w2', '0821', '6ccac868f595e4618ca7e5f67f7c113f021ffad4', 'v0.0.422'),
    ('shufflenetv2b_wd2', '1784', 'd5644a6ab8fcb6ff04f30a2eb862ebd2de92b94c', 'v0.0.422'),
    ('shufflenetv2b_w1', '1104', 'b7db0ca041e996ee76fec7f126dc39c4e5120e82', 'v0.0.422'),
    ('shufflenetv2b_w3d2', '0877', '9efb13f7d795d63c8fbee736622b9f1940dd5dd5', 'v0.0.422'),
    ('shufflenetv2b_w2', '0808', 'ba5c7ddcd8f7da3719f5d1de71d5fd30130d59d9', 'v0.0.422'),
    ('menet108_8x1_g3', '2039', '1a8cfc9296011cd994eb48e75e24c33ecf6580f5', 'v0.0.422'),
    ('menet128_8x1_g4', '1918', '7fb59f0a8d3e1f490c26546dfe93ea29ebd79c2b', 'v0.0.422'),
    ('menet160_8x1_g8', '2034', '3cf9eb2aa2d4e067aa49ce32e7a41e9db5262493', 'v0.0.422'),
    ('menet228_12x1_g3', '1291', '21bd19bf0adb73b10cb04ccce8688f119467a114', 'v0.0.422'),
    ('menet256_12x1_g4', '1217', 'd9f2e10e6402e5ee2aec485da07da72edf25f790', 'v0.0.422'),
    ('menet348_12x1_g3', '0937', 'cee7691c710f5c453b63ef9e8c3e15e699b004bb', 'v0.0.422'),
    ('menet352_12x1_g8', '1167', '54a916bcc3920c6ef24243c8c73604b25d728a6d', 'v0.0.422'),
    ('menet456_24x1_g3', '0779', '2a70b14bd17e8d4692f15f2f8e9d181e7d95b971', 'v0.0.422'),
    ('mobilenet_wd4', '2213', 'ad04596aa730e5bb4429115df70504c5a7dd5969', 'v0.0.422'),
    ('mobilenet_wd2', '1333', '01395e1b9e2a54065aafcc8b4c419644e7f6a655', 'v0.0.422'),
    ('mobilenet_w3d4', '1051', '7832561b956f0d763b002fbd9f2f880bbb712885', 'v0.0.422'),
    ('mobilenet_w1', '0866', '6939232b46fb98c8a9209d66368d630bb50941ed', 'v0.0.422'),
    ('fdmobilenet_wd4', '3062', '36aa16df43b344f42d6318cc840a81702951a033', 'v0.0.422'),
    ('fdmobilenet_wd2', '1977', '34541b84660b4e812830620c5d48df7c7a142078', 'v0.0.422'),
    ('fdmobilenet_w3d4', '1597', '0123c0313194a3094ec006f757d93f59aad73c2b', 'v0.0.422'),
    ('fdmobilenet_w1', '1312', 'fa99fb8d728f66f68464221e049a33cd2b8bfc6a', 'v0.0.422'),
    ('mobilenetv2_wd4', '2413', 'c3705f55b0df68919fba7ed79204c5651f6f71b1', 'v0.0.422'),
    ('mobilenetv2_wd2', '1446', 'b0c9a98b85b579ba77c17d228ace399809c6ab43', 'v0.0.422'),
    ('mobilenetv2_w3d4', '1044', 'e122c73eae885d204bc2ba46fb013a9da5cb282f', 'v0.0.422'),
    ('mobilenetv2_w1', '0863', 'b32cede3b68f40f2ed0552dcdf238c70f82e5705', 'v0.0.422'),
    ('mobilenetv3_large_w1', '0769', 'f66596ae10c8eaa1ea3cb79b2645bd93f946b059', 'v0.0.422'),
    ('igcv3_wd4', '2828', '309359dc5a0cd0439f2be5f629534aa3bdf2b4f9', 'v0.0.422'),
    ('igcv3_wd2', '1701', 'b952333ab2024f879d4bb9895331a617f2b957b5', 'v0.0.422'),
    ('igcv3_w3d4', '1100', '00294c7b1ab9dddf7ab2cef3e7ec0a627bd67b29', 'v0.0.422'),
    ('igcv3_w1', '0899', 'a0cb775dd5bb2c13dce35a21d6fd53a783959702', 'v0.0.422'),
    ('mnasnet_b1', '0802', '763d6849142ce86f46cb7ec4c003ccf15542d6eb', 'v0.0.422'),
    ('mnasnet_a1', '0756', '8e0f49481a3473b9457d0987c9c6f7e51ff57576', 'v0.0.422')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".tensorflow", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TENSORFLOW_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.tf2.h5".format(
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
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
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
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash
