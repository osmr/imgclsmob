"""
    COCO semantic segmentation dataset.
"""

import os
import logging
import numpy as np
import mxnet as mx
from PIL import Image
from tqdm import trange
from .seg_dataset import SegDataset
from .voc_seg_dataset import VOCMetaInfo


class CocoSegDataset(SegDataset):
    """
    COCO semantic segmentation dataset.

    Parameters
    ----------
    root : string
        Path to `annotations`, `train2017`, and `val2017` folders.
    mode : string, default 'train'
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

        year = "2017"
        mode_name = "train" if mode == "train" else "val"
        annotations_dir_path = os.path.join(root, "annotations")
        annotations_file_path = os.path.join(annotations_dir_path, "instances_" + mode_name + year + ".json")
        idx_file_path = os.path.join(annotations_dir_path, mode_name + "_idx.npy")
        self.image_dir_path = os.path.join(root, mode_name + year)

        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask
        self.coco = COCO(annotations_file_path)
        self.coco_mask = coco_mask
        if os.path.exists(idx_file_path):
            self.idx = np.load(idx_file_path)
        else:
            idx_list = list(self.coco.imgs.keys())
            self.idx = self._filter_idx(idx_list, idx_file_path)

    def __getitem__(self, index):
        image_id = int(self.idx[index])
        image_metadata = self.coco.loadImgs(image_id)[0]
        image_file_name = image_metadata["file_name"]

        image_file_path = os.path.join(self.image_dir_path, image_file_name)
        image = Image.open(image_file_path).convert("RGB")
        if self.mode == "demo":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(image_file_path)

        coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        mask = Image.fromarray(self._gen_seg_mask(
            target=coco_target,
            height=image_metadata["height"],
            width=image_metadata["width"]))

        if self.mode == "train":
            image, mask = self._train_sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert (self.mode == "test")
            image, mask = self._img_transform(image), self._mask_transform(mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    def _gen_seg_mask(self, target, height, width):
        cat_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        mask = np.zeros((height, width), dtype=np.uint8)
        for instance in target:
            rle = self.coco_mask.frPyObjects(instance["segmentation"], height, width)
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
                    idx_list,
                    idx_file_path,
                    pixels_thr=1000):
        logging.info("Filtering mask index:")
        tbar = trange(len(idx_list))
        filtered_idx = []
        for i in tbar:
            img_id = idx_list[i]
            coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(
                coco_target,
                img_metadata["height"],
                img_metadata["width"])
            if (mask > 0).sum() > pixels_thr:
                filtered_idx.append(img_id)
            tbar.set_description("Doing: {}/{}, got {} qualified images".format(i, len(idx_list), len(filtered_idx)))
        logging.info("Found number of qualified images: {}".format(len(filtered_idx)))
        np.save(idx_file_path, np.array(filtered_idx, np.int32))
        return filtered_idx

    classes = 21
    vague_idx = -1
    use_vague = False
    background_idx = 0
    ignore_bg = True

    @staticmethod
    def _mask_transform(mask, ctx=mx.cpu()):
        np_mask = np.array(mask).astype(np.int32)
        # print("min={}, max={}".format(np_mask.min(), np_mask.max()))
        return mx.nd.array(np_mask, ctx=ctx)

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
        self.train_metric_extra_kwargs = [
            {"vague_idx": CocoSegDataset.vague_idx,
             "use_vague": CocoSegDataset.use_vague,
             "macro_average": False,
             "aux": self.train_aux}]
        self.val_metric_extra_kwargs = [
            {"vague_idx": CocoSegDataset.vague_idx,
             "use_vague": CocoSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": CocoSegDataset.classes,
             "vague_idx": CocoSegDataset.vague_idx,
             "use_vague": CocoSegDataset.use_vague,
             "bg_idx": CocoSegDataset.background_idx,
             "ignore_bg": CocoSegDataset.ignore_bg,
             "macro_average": False}]
        self.test_metric_extra_kwargs = self.val_metric_extra_kwargs
