import os
import pandas as pd
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_csv('/opt/ml/input/data/train/train.csv')
        self.path = path
        self.data = df['path']
        self.transform = transform

    def __len__(self):
        return len(self.data) * 7

    def __getitem__(self, idx):
        label_dict = {0:1, 1:0, 2:0, 3:0, 4:0, 5:0, 6:2}
        folder_name = self.data[idx // 7]
        folder_path = os.path.join(self.path, folder_name)
        image_list = sorted(glob(folder_path + '/*'))
        idx = idx % 7
        img = Image.open(image_list[idx])
        label = label_dict[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class GenderDataset(Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_csv('/opt/ml/input/data/train/train.csv')
        self.path = path
        self.data = df['path']
        self.transform = transform

    def __len__(self):
        return len(self.data) * 7

    def __getitem__(self, idx):
        label_dict = {'male':0, 'female':1}
        folder_name = self.data[idx // 7]
        folder_path = os.path.join(self.path, folder_name)
        image_list = sorted(glob(folder_path + '/*'))
        idx = idx % 7
        img = Image.open(image_list[idx])
        gender = folder_name.split('_')[1]
        label = label_dict[gender]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class AgeDataset(Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_csv('/opt/ml/input/data/train/train.csv')
        self.path = path
        self.data = df['path']
        self.transform = transform

    def __len__(self):
        return len(self.data) * 7

    def __getitem__(self, idx):
        folder_name = self.data[idx // 7]
        folder_path = os.path.join(self.path, folder_name)
        image_list = sorted(glob(folder_path + '/*'))
        idx = idx % 7
        img = Image.open(image_list[idx])
        age = int(folder_name.split('_')[3])
        if age < 30:
            label = 0
        elif 30 <= age and age < 60:
            label = 1
        elif age >= 60:
            label = 2

        if self.transform is not None:
            img = self.transform(img)

        return img, label
