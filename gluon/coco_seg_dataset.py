import os
import logging
import numpy as np
import mxnet as mx
from PIL import Image
import pickle
from tqdm import trange
from .seg_dataset import SegDataset


class COCOSegDataset(SegDataset):
    """
    Pascal COCO semantic segmentation dataset.

    Parameters
    ----------
    root : string
        Path to `annotations`, `train2017`, and `val2017` folders.
    mode: string, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 **kwargs):
        super(COCOSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)

        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "instances_" + mode_name + "2017.json")
        idx_file_path = os.path.join(annotations_dir_path, mode_name + ".idx")
        self.image_dir_path = os.path.join(root, mode_name + "2017")

        try_import_pycocotools()
        from pycocotools.coco import COCO
        from pycocotools import mask
        self.coco = COCO(annotations_file_path)
        self.coco_mask = mask
        if os.path.exists(idx_file_path):
            with open(idx_file_path, "rb") as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, idx_file_path)
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        image_idx = self.ids[index]
        img_metadata = coco.loadImgs(image_idx)[0]
        image_file_name = img_metadata["file_name"]

        image_file_path = os.path.join(self.image_dir_path, image_file_name)
        image = Image.open(image_file_path).convert("RGB")
        if self.mode == "demo":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(self.images[index])

        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=image_idx))
        mask = Image.fromarray(self._gen_seg_mask(
            cocotarget,
            img_metadata["height"],
            img_metadata["width"]))

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert self.mode == "test"
            image, mask = self._img_transform(image), self._mask_transform(mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    def _gen_seg_mask(self, target, h, w):
        CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in CAT_LIST:
                c = CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        logging.info("Preprocessing mask, this will take a while. But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(
                cocotarget,
                img_metadata["height"],
                img_metadata["width"])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description("Doing: {}/{}, got {} qualified images".format(i, len(ids), len(new_ids)))
        logging.info("Found number of qualified images: ", len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    classes = 21
    vague_idx = 255
    use_vague = True
    background_idx = 0
    ignore_bg = True

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        # np_mask[np_mask == 255] = VOCSegDataset.vague_idx
        return mx.nd.array(np_mask, mx.cpu())

    def __len__(self):
        return len(self.ids)


def try_import_pycocotools():
    """
    Optionally install and import pycocotools.
    """
    try:
        import pycocotools as cocot
        assert (cocot is not None)
    except ImportError:
        import os
        import_try_install("cython")
        try:
            if os.name == "nt":
                win_url = "git+https://github.com/zhreshold/cocoapi.git#subdirectory=PythonAPI"
                import_try_install("pycocotools", win_url)
            else:
                import_try_install("pycocotools")
        except ImportError:
            raise ImportError("Cannot import or install pycocotools, please refer to cocoapi FAQ.")


def import_try_install(package, extern_url=None):
    """
    Try import the specified package.
    If the package not installed, try use pip to install and import if success.

    Parameters
    ----------
    package : str
        The name of the package trying to import.
    extern_url : str or None, optional
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
         "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.

    Returns
    -------
    <class 'Module'>
        The imported python module.

    """
    try:
        return __import__(package)
    except ImportError:
        try:
            from pip import main as pipmain
        except ImportError:
            from pip._internal import main as pipmain

        url = package if extern_url is None else extern_url
        pipmain(["install", "--user", url])  # will raise SystemExit Error if fails

        try:
            return __import__(package)
        except ImportError:
            import sys
            import site
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            return __import__(package)
    return __import__(package)
