from pathlib import Path
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os


DATASETS_PATH = Path("./data/raw")
IMAGENET_MEAN = tensor([0.485, 0.456, 0.406])
IMAGENET_STD = tensor([0.229, 0.224, 0.225])


class MVTecDataset:
    def __init__(self, class_name: str, size: int = 224):
        self.class_name = class_name
        self.size = size
        self.train_ds = MVTecTrainDataset(class_name, size)
        self.test_ds = MVTecTestDataset(class_name, size)

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


class MVTecTrainDataset(ImageFolder):
    def __init__(self, class_name: str, size: int):
        super().__init__(
            root=DATASETS_PATH / class_name / "train",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size


class MVTecTestDataset(ImageFolder):
    def __init__(self, class_name: str, size: int):
        super().__init__(
            root=DATASETS_PATH / class_name / "test",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
            target_transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size

    def __getitem__(self, index):  # type: ignore
        sample, target, sample_class, _ = self.get_item_with_details(index)
        return sample, target, sample_class

    def get_item_with_details(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        # Extract label name from path
        label_name = Path(path).parent.name

        if "good" in path:
            target = Image.new("L", (self.size, self.size))
            sample_class = 0
        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            # Some ground truth masks might not exist
            if os.path.exists(target_path):
                target = self.loader(target_path)
            else:
                target = Image.new("L", (self.size, self.size))
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class, label_name  # type: ignore
