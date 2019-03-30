import os
import numpy as np
from PIL import Image
from .seg_dataset import SegDataset


class ADE20KSegDataset(SegDataset):
    """
    ADE20K semantic segmentation dataset.

    Parameters
    ----------
    root : string
        Path to a folder with `ADEChallengeData2016` subfolder.
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

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def _get_image(self, i):
        image = Image.open(self.images[i]).convert("RGB")
        assert (self.mode in ("test", "demo"))
        image = self._img_transform(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_label(self, i):
        if self.mode == "demo":
            return os.path.basename(self.images[i])
        assert (self.mode == "test")
        mask = Image.open(self.masks[i])
        mask = self._mask_transform(mask)
        return mask

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
        return np_mask

    def __len__(self):
        return len(self.images)
