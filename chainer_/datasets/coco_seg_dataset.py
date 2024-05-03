"""
    COCO semantic segmentation dataset.
"""

import os
import logging
import numpy as np
from PIL import Image
from tqdm import trange
from .seg_dataset import SegDataset
from .voc_seg_dataset import VOCMetaInfo


class CocoSegDataset(SegDataset):
    """
    COCO semantic segmentation dataset.

    Parameters
    ----------
    root: str
        Path to `annotations`, `train2017`, and `val2017` folders.
    mode: str, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 **kwargs):
        super(CocoSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)

        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "instances_" + mode_name + "2017.json")
        idx_file_path = os.path.join(annotations_dir_path, mode_name + "_idx.npy")
        self.image_dir_path = os.path.join(root, mode_name + "2017")

        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask
        self.coco = COCO(annotations_file_path)
        self.coco_mask = coco_mask
        if os.path.exists(idx_file_path):
            self.idx = np.load(idx_file_path)
        else:
            idx_list = list(self.coco.imgs.keys())
            self.idx = self._filter_idx(idx_list, idx_file_path)

        self.transform = transform

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def _get_image(self, i):
        image_idx = int(self.idx[i])
        img_metadata = self.coco.loadImgs(image_idx)[0]
        image_file_name = img_metadata["file_name"]

        image_file_path = os.path.join(self.image_dir_path, image_file_name)
        image = Image.open(image_file_path).convert("RGB")

        assert (self.mode in ("test", "demo"))
        image = self._img_transform(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_label(self, i):
        if self.mode == "demo":
            image_idx = int(self.idx[i])
            img_metadata = self.coco.loadImgs(image_idx)[0]
            image_file_name = img_metadata["file_name"]
            image_file_path = os.path.join(self.image_dir_path, image_file_name)
            return os.path.basename(image_file_path)
        assert (self.mode == "test")

        image_idx = int(self.idx[i])
        img_metadata = self.coco.loadImgs(image_idx)[0]

        coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_idx))
        mask = Image.fromarray(self._gen_seg_mask(
            coco_target,
            img_metadata["height"],
            img_metadata["width"]))

        mask = self._mask_transform(mask)
        return mask

    def _gen_seg_mask(self, target, h, w):
        cat_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        mask = np.zeros((h, w), dtype=np.uint8)
        for instance in target:
            rle = self.coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = self.coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in cat_list:
                c = cat_list.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _filter_idx(self,
                    idx,
                    idx_file,
                    pixels_thr=1000):
        logging.info("Filtering mask index")
        tbar = trange(len(idx))
        filtered_idx = []
        for i in tbar:
            img_id = idx[i]
            coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(
                coco_target,
                img_metadata["height"],
                img_metadata["width"])
            if (mask > 0).sum() > pixels_thr:
                filtered_idx.append(img_id)
            tbar.set_description("Doing: {}/{}, got {} qualified images".format(i, len(idx), len(filtered_idx)))
        logging.info("Found number of qualified images: {}".format(len(filtered_idx)))
        np.save(idx_file, np.array(filtered_idx, np.int32))
        return filtered_idx

    classes = 21
    vague_idx = -1
    use_vague = False
    background_idx = 0
    ignore_bg = True

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        return np_mask

    def __len__(self):
        return len(self.idx)


class CocoSegMetaInfo(VOCMetaInfo):
    def __init__(self):
        super(CocoSegMetaInfo, self).__init__()
        self.label = "COCO"
        self.short_label = "coco"
        self.root_dir_name = "coco"
        self.dataset_class = CocoSegDataset
        self.num_classes = CocoSegDataset.classes
        self.test_metric_extra_kwargs = [
            {"vague_idx": CocoSegDataset.vague_idx,
             "use_vague": CocoSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": CocoSegDataset.classes,
             "vague_idx": CocoSegDataset.vague_idx,
             "use_vague": CocoSegDataset.use_vague,
             "bg_idx": CocoSegDataset.background_idx,
             "ignore_bg": CocoSegDataset.ignore_bg,
             "macro_average": False}]
