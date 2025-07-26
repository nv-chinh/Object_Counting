import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
import os
from glob import glob
from PIL import Image
import numpy as np
from typing import Optional, Callable, Union, Tuple
from scipy.io import loadmat

from .utils import get_id, generate_density_map

curr_dir = os.path.dirname(os.path.abspath(__file__))

class Crowd(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        """
        Dataset for crowd counting.
        """
        assert split in ["train", "val"], f"Split {split} is not available."
        assert num_crops > 0, f"num_crops should be positive, got {num_crops}."

        self.dataset = dataset
        self.split = split

        self.root = os.path.join(curr_dir, "data", self.dataset)
        self.image_names = os.listdir(os.path.join(self.root, self.split, "images"))
        self.indices = list(range(len(self.image_names)))

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename
        self.num_crops = num_crops

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        image_name = self.image_names[idx]

        image_path = os.path.join(self.root, self.split, "images", image_name)
        label_path = os.path.join(self.root, self.split, "labels", image_name[:-4]+".npy")

        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        image = self.to_tensor(image)

        with open(label_path, "rb") as f:
            label = np.load(f)

        label = torch.from_numpy(label).float()

        if self.transforms is not None:
            images_labels = [self.transforms(image.clone(), label.clone()) for _ in range(self.num_crops)]
            images, labels = zip(*images_labels)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            labels = [label.clone() for _ in range(self.num_crops)]

        images = [self.normalize(img) for img in images]
        if idx in self.indices:
            density_maps = torch.stack([generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in zip(images, labels)], 0)
        else:
            labels = None
            density_maps = None

        image_names = [image_name] * len(images)
        images = torch.stack(images, 0)

        if self.return_filename:
            return images, labels, density_maps, image_names
        else:
            return images, labels, density_maps

class Crowd_Inference(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset: str,
        split: str,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        """
        Dataset for crowd counting.
        """
        assert split in ["train", "val", "test"], f"Split {split} is not available."
        assert num_crops > 0, f"num_crops should be positive, got {num_crops}."

        self.data_path = data_path
        self.dataset = dataset
        self.split = split
        self.root = self.data_path
        self.image_names = os.listdir(os.path.join(self.root, self.split, "images"))
        self.indices = list(range(len(self.image_names)))

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename
        self.num_crops = num_crops

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        image_name = self.image_names[idx]

        image_path = os.path.join(self.root, self.split, "images", image_name)
        label_path = os.path.join(self.root, self.split, "labels", image_name[:-4]+".txt")
        if os.path.exists(label_path):
            with open(os.path.join(label_path),'r') as f:
                gt = f.read().splitlines()
            num_gt = len(gt)
        else:
            mat_label_path = os.path.join(self.root, self.split, "labels", 'GT_'+image_name[:-4]+".mat")
            gt = loadmat(mat_label_path)["image_info"][0][0][0][0][0]
            num_gt = len(gt)

        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.return_filename:
            return image, num_gt, image_name
        else:
            return image, num_gt
