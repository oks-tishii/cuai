from PIL import Image
from torch import tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from config import settings


class MVTecDataset:
    def __init__(self, class_name: str, size: int = settings.IMAGE_SIZE[0]):
        self.class_name = class_name
        self.size = size
        self.train_ds = MVTecTrainDataset(class_name, size)
        self.test_ds = MVTecTestDataset(class_name, size)

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


class MVTecTrainDataset(ImageFolder):
    def __init__(self, class_name: str, size: int = settings.IMAGE_SIZE[0]):
        super().__init__(
            root=settings.DATA_DIR / class_name / "train",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size


class MVTecTestDataset(ImageFolder):
    def __init__(self, class_name: str, size: int = settings.IMAGE_SIZE[0]):
        super().__init__(
            root=settings.DATA_DIR / class_name / "test",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD),
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
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:
            target = Image.new("L", (self.size, self.size))
            sample_class = 0
        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            target = self.loader(target_path)
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class  # type: ignore


class StreamingDataset:
    """このデータセットは、streamlit アプリ専用に作成されています。"""

    def __init__(self, size: int = settings.IMAGE_SIZE[0]):
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD),
            ]
        )
        self.samples = []

    def add_pil_image(self, image: Image):  # type: ignore
        image = image.convert("RGB")  # type: ignore
        self.samples.append(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (self.transform(sample), tensor(0.0))
