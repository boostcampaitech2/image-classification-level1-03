import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

import albumentations
from albumentations.pytorch.transforms import ToTensorV2


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class TrainDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, transform=None, resize=(300, 300), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.resize = resize

        self.setup()
        self.calc_statistics()
        if transform == 'train':
            self.transforms = albumentations.Compose([
                albumentations.CenterCrop(384, 384),
                albumentations.Resize(self.resize[0], self.resize[1]),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.ToGray(p=1),
                    albumentations.CoarseDropout(max_holes=7, max_height=50, max_width=50, min_height=30, min_width=30, p=1),
                ], p=1),
                albumentations.OneOf([
                    albumentations.GaussNoise(p=1),
                    albumentations.GaussianBlur(p=1),
                    albumentations.RandomBrightnessContrast(p=1),
                    albumentations.HueSaturationValue(p=1),
                    albumentations.CLAHE(p=1),
                    albumentations.RandomGamma(p=1),
                    albumentations.ImageCompression(p=1),
                ], p=1),
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        elif transform == 'valid':
            self.transforms = albumentations.Compose([
                albumentations.CenterCrop(384, 384),
                albumentations.Resize(self.resize[0], self.resize[1]),
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                cnt += 1

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        
        image_transform = self.transforms(image=image)['image']
        return image_transform, (mask_label, gender_label, age_label)

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = albumentations.Compose([
            albumentations.CenterCrop(384, 384),
            albumentations.Resize(resize[0], resize[1]),
            albumentations.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        # image = Image.open(self.img_paths[index])
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)
