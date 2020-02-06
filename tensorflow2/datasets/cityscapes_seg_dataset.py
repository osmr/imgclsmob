"""
    Cityscapes semantic segmentation dataset.
"""

import os
import numpy as np
from PIL import Image
from .seg_dataset import SegDataset
from .voc_seg_dataset import VOCMetaInfo


class CityscapesSegDataset(SegDataset):
    """
    Cityscapes semantic segmentation dataset.

    Parameters
    ----------
    root : str
        Path to a folder with `leftImg8bit` and `gtFine` subfolders.
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
        super(CityscapesSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)

        image_dir_path = os.path.join(root, "leftImg8bit")
        mask_dir_path = os.path.join(root, "gtFine")
        assert os.path.exists(image_dir_path) and os.path.exists(mask_dir_path), "Please prepare dataset"

        mode_dir_name = "train" if mode == "train" else "val"
        image_dir_path = os.path.join(image_dir_path, mode_dir_name)
        # mask_dir_path = os.path.join(mask_dir_path, mode_dir_name)

        self.images = []
        self.masks = []
        for image_subdir_path, _, image_file_names in os.walk(image_dir_path):
            for image_file_name in image_file_names:
                if image_file_name.endswith(".png"):
                    image_file_path = os.path.join(image_subdir_path, image_file_name)
                    mask_file_name = image_file_name.replace("leftImg8bit", "gtFine_labelIds")
                    mask_subdir_path = image_subdir_path.replace("leftImg8bit", "gtFine")
                    mask_file_path = os.path.join(mask_subdir_path, mask_file_name)
                    if os.path.isfile(mask_file_path):
                        self.images.append(image_file_path)
                        self.masks.append(mask_file_path)
                    else:
                        print("Cannot find the mask: {}".format(mask_file_path))

        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: {}\n".format(image_dir_path))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.mode == "demo":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert (self.mode == "test")
            image = self._img_transform(image)
            mask = self._mask_transform(mask)

        if self.transform is not None:
            image = self.transform(image)
        return image, mask

    classes = 19
    vague_idx = 19
    use_vague = True
    background_idx = -1
    ignore_bg = False

    _key = np.array([-1, -1, -1, -1, -1, -1,
                     -1, -1, 0, 1, -1, -1,
                     2, 3, 4, -1, -1, -1,
                     5, -1, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15,
                     -1, -1, 16, 17, 18])
    _mapping = np.array(range(-1, len(_key) - 1)).astype(np.int32)

    @staticmethod
    def _class_to_index(mask):
        values = np.unique(mask)
        for value in values:
            assert(value in CityscapesSegDataset._mapping)
        index = np.digitize(mask.ravel(), CityscapesSegDataset._mapping, right=True)
        return CityscapesSegDataset._key[index].reshape(mask.shape)

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        np_mask = CityscapesSegDataset._class_to_index(np_mask)
        np_mask[np_mask == -1] = CityscapesSegDataset.vague_idx
        return np_mask

    def __len__(self):
        return len(self.images)


class CityscapesMetaInfo(VOCMetaInfo):
    def __init__(self):
        super(CityscapesMetaInfo, self).__init__()
        self.label = "Cityscapes"
        self.short_label = "voc"
        self.root_dir_name = "cityscapes"
        self.dataset_class = CityscapesSegDataset
        self.num_classes = CityscapesSegDataset.classes
        self.test_metric_extra_kwargs = [
            {"vague_idx": CityscapesSegDataset.vague_idx,
             "use_vague": CityscapesSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": CityscapesSegDataset.classes,
             "vague_idx": CityscapesSegDataset.vague_idx,
             "use_vague": CityscapesSegDataset.use_vague,
             "bg_idx": CityscapesSegDataset.background_idx,
             "ignore_bg": CityscapesSegDataset.ignore_bg,
             "macro_average": False}]
