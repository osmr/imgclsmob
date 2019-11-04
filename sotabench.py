from torchbench.image_classification import ImageNet
from pytorch.pytorchcv.model_provider import trained_model_metainfo_list
from pytorch.pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision.transforms as transforms
import torch
import math
# import os

for model_metainfo in trained_model_metainfo_list:
    input_image_size = model_metainfo[3]
    resize_inv_factor = model_metainfo[4]
    ImageNet.benchmark(
        model=ptcv_get_model(model_metainfo[0], pretrained=True),
        model_description=model_metainfo[5],
        paper_model_name=model_metainfo[1],
        paper_arxiv_id=model_metainfo[2],
        input_transform=transforms.Compose([
            transforms.Resize(int(math.ceil(float(input_image_size) / resize_inv_factor))),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
        batch_size=200,
        num_gpu=1,
        # data_root=os.path.join("..", "imgclsmob_data", "imagenet")
    )
    torch.cuda.empty_cache()
