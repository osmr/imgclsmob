"""
    ADE20K semantic segmentation dataset.
"""

import os
import numpy as np
import mxnet as mx
from PIL import Image
from .seg_dataset import SegDataset
from .voc_seg_dataset import VOCMetaInfo


class ADE20KSegDataset(SegDataset):
    """
    ADE20K semantic segmentation dataset.

    Parameters
    ----------
    root : str
        Path to a folder with `ADEChallengeData2016` subfolder.
    mode : str, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 **kwargs):
        super(ADE20KSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)

        base_dir_path = os.path.join(root, "ADEChallengeData2016")
        assert os.path.exists(base_dir_path), "Please prepare dataset"

        image_dir_path = os.path.join(base_dir_path, "images")
        mask_dir_path = os.path.join(base_dir_path, "annotations")

        mode_dir_name = "training" if mode == "train" else "validation"
        image_dir_path = os.path.join(image_dir_path, mode_dir_name)
        mask_dir_path = os.path.join(mask_dir_path, mode_dir_name)

        self.images = []
        self.masks = []
        for image_file_name in os.listdir(image_dir_path):
            image_file_stem, _ = os.path.splitext(image_file_name)
            if image_file_name.endswith(".jpg"):
                image_file_path = os.path.join(image_dir_path, image_file_name)
                mask_file_name = image_file_stem + ".png"
                mask_file_path = os.path.join(mask_dir_path, mask_file_name)
                if os.path.isfile(mask_file_path):
                    self.images.append(image_file_path)
                    self.masks.append(mask_file_path)
                else:
                    print("Cannot find the mask: {}".format(mask_file_path))

        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: {}\n".format(base_dir_path))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        # image = mx.image.imread(self.images[index])
        if self.mode == "demo":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # mask = mx.image.imread(self.masks[index])

        if self.mode == "train":
            image, mask = self._train_sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert (self.mode == "test")
            image = self._img_transform(image)
            mask = self._mask_transform(mask)

        if self.transform is not None:
            image = self.transform(image)
        return image, mask

    classes = 150
    vague_idx = 150
    use_vague = True
    background_idx = -1
    ignore_bg = False

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        np_mask[np_mask == 0] = ADE20KSegDataset.vague_idx + 1
        np_mask -= 1
        return mx.nd.array(np_mask, mx.cpu())

    def __len__(self):
        return len(self.images)


class ADE20KMetaInfo(VOCMetaInfo):
    def __init__(self):
        super(ADE20KMetaInfo, self).__init__()
        self.label = "ADE20K"
        self.short_label = "voc"
        self.root_dir_name = "ade20k"
        self.dataset_class = ADE20KSegDataset
        self.num_classes = ADE20KSegDataset.classes
        self.test_metric_extra_kwargs = [
            {"vague_idx": ADE20KSegDataset.vague_idx,
             "use_vague": ADE20KSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": ADE20KSegDataset.classes,
             "vague_idx": ADE20KSegDataset.vague_idx,
             "use_vague": ADE20KSegDataset.use_vague,
             "bg_idx": ADE20KSegDataset.background_idx,
             "ignore_bg": ADE20KSegDataset.ignore_bg,
             "macro_average": False}]
