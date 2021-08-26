import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

test_dir = '/opt/ml/input/data/eval'


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
result = pd.read_csv(os.path.join(test_dir, 'result.csv'))
image_dir = os.path.join(test_dir, 'images')

image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([ToTensor()])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

device = torch.device('cuda')

model = torchvision.models.resnet50(pretrained=True)
in_features = model.fc.weight.shape[1]
model.fc = torch.nn.Linear(in_features=in_features, out_features=3, bias=True)
model.load_state_dict(torch.load('/opt/ml/code/model_M.pth'))
model = model.to(device)
model.eval()

print('----------predict Mask----------')
all_predictions = []
for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
mask = all_predictions
result['mask'] = mask


model.fc = torch.nn.Linear(in_features=in_features, out_features=2, bias=True)
model.load_state_dict(torch.load('/opt/ml/code/model_G.pth'))
model = model.to(device)
model.eval()

print('----------predict Gender----------')
all_predictions = []
for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
gender = all_predictions
result['gender'] = gender


model.fc = torch.nn.Linear(in_features=in_features, out_features=3, bias=True)
model.load_state_dict(torch.load('/opt/ml/code/model_A.pth'))
model = model.to(device)
model.eval()

print('----------predict Age----------')
all_predictions = []
for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
age = all_predictions
result['age'] = age


all_predictions = [6 * M + 3 * G + A for M, G, A in zip(mask, gender, age)]
submission['ans'] = all_predictions
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
result.to_csv(os.path.join(test_dir, 'result.csv'), index=False)
print('test inference is done!')
