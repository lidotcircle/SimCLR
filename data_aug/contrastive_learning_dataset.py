from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from PIL import Image
import glob
import torch
import os


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transforms=None):
        self.img_dir = dir
        jpg = os.path.join(dir, '*.jpg')
        png = os.path.join(dir, '*.png')
        self.files = glob.glob(jpg) + glob.glob(png)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, aug_transforms='crop_flip_color_gray_blur'):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        trans = []
        if 'crop' in aug_transforms:
            trans.append(transforms.RandomResizedCrop(size=size))
        if 'flip' in aug_transforms:
            trans.append(transforms.RandomHorizontalFlip())
        if 'color' in aug_transforms:
            trans.append(transforms.RandomApply([color_jitter], p=0.8))
        if 'gray' in aug_transforms:
            trans.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in aug_transforms:
            trans.append(GaussianBlur(kernel_size=int(0.1 * size)))
        trans.append(transforms.ToTensor())

        data_transforms = transforms.Compose(trans)
        return data_transforms

    @staticmethod
    def get_to_tensor_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, n_views, identical: bool=False, aug_transforms='crop_flip_color_gray_blur'):
        return ImagePathDataset(self.root_folder,
                         transforms=ContrastiveLearningViewGenerator(
                            self.get_to_tensor_transform(256) if identical else self.get_simclr_pipeline_transform(256),
                            n_views))