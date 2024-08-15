from torchbench.image_classification import ImageNet
from pytorchcv.models.common.model_store import get_model_metainfo_dict
from pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision.transforms as transforms
import torch
import math
from sys import version_info


model_metainfo_dict = get_model_metainfo_dict()
for model_name, model_metainfo in (model_metainfo_dict.items() if version_info[0] >= 3 else model_metainfo_dict.iteritems()):  # noqa
    caption, paper, ds, img_size, scale, batch, rem = model_metainfo[4:]
    net = ptcv_get_model(model_name, pretrained=True)
    if (ds != "in1k") or (img_size == 0) or ((len(rem) > 0) and (rem[-1] == "*")):
        continue
    paper_model_name = caption
    paper_arxiv_id = paper
    input_image_size = img_size
    resize_inv_factor = scale
    batch_size = batch
    model_description = "pytorch" + (rem if rem == "" else ", " + rem)
    assert (not hasattr(net, "in_size")) or (input_image_size == net.in_size[0])
    ImageNet.benchmark(
        model=net,
        model_description=model_description,
        paper_model_name=paper_model_name,
        paper_arxiv_id=paper_arxiv_id,
        input_transform=transforms.Compose([
            transforms.Resize(int(math.ceil(float(input_image_size) / resize_inv_factor))),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
        batch_size=batch_size,
        num_gpu=1,
        # data_root=os.path.join("..", "imgclsmob_data", "imagenet")
    )
    torch.cuda.empty_cache()
