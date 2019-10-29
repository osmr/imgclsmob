from torchbench.image_classification import ImageNet
from pytorch.pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision.transforms as transforms
# import os

input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ImageNet.benchmark(
    model=ptcv_get_model("resnet18", pretrained=True),
    paper_model_name="ResNet-18",
    paper_arxiv_id="1512.03385",
    input_transform=input_transform,
    batch_size=200,
    num_gpu=1,
    # data_root=os.path.join("..", "imgclsmob_data", "imagenet")
)
