import os
import sys
import gzip
import random
import platform
import warnings
import collections
from tqdm import tqdm, tqdm_notebook

import re
import requests
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from glob import glob
import math

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

# torch.cuda.empty_cache()

# current_os = platform.system()
# print(f"Current OS: {current_os}")
# print(f"CUDA: {torch.cuda.is_available()}")
# print(f"Python Version: {platform.python_version()}")
# print(f"torch Version: {torch.__version__}")
# print(f"torchvision Version: {torchvision.__version__}")

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

train_image_dir = '/opt/ml/input/data/train/images'
data_transforms = transforms.Compose([transforms.Resize([384, 512]),
                                      transforms.ToTensor()])

MD = MaskDataset(train_image_dir, transform=data_transforms)



train_data_length = int(len(MD) * 0.8)
val_data_length = int(len(MD) * 0.2)

MD_train, MD_val = random_split(MD, [train_data_length, val_data_length])


BATCH_SIZE = 64
MD_train_loader = DataLoader(MD_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
MD_val_loader = DataLoader(MD_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


model = torchvision.models.resnet50(pretrained=True)
# print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
# print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc.weight.shape[0])
# print(model)

model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
torch.nn.init.xavier_uniform_(model.fc.weight)
stdv = 1 / math.sqrt(model.fc.weight.size(1))
model.fc.bias.data.uniform_(-stdv, stdv)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is using!")


LEARNING_RATE = 0.0001
NUM_EPOCH = 5

model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


### MD
print('----------MD----------')

best_val_accuracy = 0.
best_val_loss = 0.

dataloaders = {
    "train" : MD_train_loader,
    "val" : MD_val_loader
}

for epoch in range(NUM_EPOCH):
  for phase in ["train", "val"]:
    running_loss = 0.
    running_acc = 0.
    if phase == "train":
      model.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
    elif phase == "val":
      model.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함

    for ind, (images, labels) in enumerate(tqdm(dataloaders[phase])):
      # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
      # hint - with, pbar
      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

      with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
        logits = model(images)
        _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
        loss = loss_fn(logits, labels)

        if phase == "train":
          loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
          optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

      running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
      running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장

    # 한 epoch이 모두 종료되었을 때,
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_acc / len(dataloaders[phase].dataset)

    print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
    if phase == "val" and best_val_accuracy < epoch_acc: # phase가 val일 때, best accuracy 계산
      best_val_accuracy = epoch_acc
    if phase == "val" and best_val_loss < epoch_loss: # phase가 val일 때, best loss 계산
      best_val_loss = epoch_loss
print("학습 종료!")
print(f"최고 accuracy : {best_val_accuracy}, 최고 낮은 loss : {best_val_loss}")

torch.save(model.state_dict(), "model_M.pth")
print("Saved PyTorch Model State to model_M.pth")

### GD
print('----------GD----------')
print(f"{device} is using!")

GD = GenderDataset(train_image_dir, transform=data_transforms)
GD_train, GD_val = random_split(GD, [train_data_length, val_data_length])
GD_train_loader = DataLoader(GD_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
GD_val_loader = DataLoader(GD_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

# model = resnet50
# model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
# model.load_state_dict(torch.load("model_M.pth"))

model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
torch.nn.init.xavier_uniform_(model.fc.weight)
stdv = 1 / math.sqrt(model.fc.weight.size(1))
model.fc.bias.data.uniform_(-stdv, stdv)
model = model.to(device)

best_val_accuracy = 0.
best_val_loss = 0.

dataloaders = {
    "train" : GD_train_loader,
    "val" : GD_val_loader
}

for epoch in range(NUM_EPOCH):
  for phase in ["train", "val"]:
    running_loss = 0.
    running_acc = 0.
    if phase == "train":
      model.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
    elif phase == "val":
      model.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함

    for ind, (images, labels) in enumerate(tqdm(dataloaders[phase])):
      # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
      # hint - with, pbar
      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

      with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
        logits = model(images)
        _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
        loss = loss_fn(logits, labels)

        if phase == "train":
          loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
          optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

      running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
      running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장

    # 한 epoch이 모두 종료되었을 때,
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_acc / len(dataloaders[phase].dataset)

    print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
    if phase == "val" and best_val_accuracy < epoch_acc: # phase가 val일 때, best accuracy 계산
      best_val_accuracy = epoch_acc
    if phase == "val" and best_val_loss < epoch_loss: # phase가 val일 때, best loss 계산
      best_val_loss = epoch_loss
print("학습 종료!")
print(f"최고 accuracy : {best_val_accuracy}, 최고 낮은 loss : {best_val_loss}")

torch.save(model.state_dict(), "model_G.pth")
print("Saved PyTorch Model State to model_G.pth")


### AD
print('----------AD----------')
print(f"{device} is using!")

AD = AgeDataset(train_image_dir, transform=data_transforms)
AD_train, AD_val = random_split(AD, [train_data_length, val_data_length])
AD_train_loader = DataLoader(AD_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
AD_val_loader = DataLoader(AD_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
torch.nn.init.xavier_uniform_(model.fc.weight)
stdv = 1 / math.sqrt(model.fc.weight.size(1))
model.fc.bias.data.uniform_(-stdv, stdv)
model = model.to(device)

best_val_accuracy = 0.
best_val_loss = 0.

dataloaders = {
    "train" : AD_train_loader,
    "val" : AD_val_loader
}

for epoch in range(NUM_EPOCH):
  for phase in ["train", "val"]:
    running_loss = 0.
    running_acc = 0.
    if phase == "train":
      model.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
    elif phase == "val":
      model.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함

    for ind, (images, labels) in enumerate(tqdm(dataloaders[phase])):
      # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
      # hint - with, pbar
      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

      with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
        logits = model(images)
        _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
        loss = loss_fn(logits, labels)

        if phase == "train":
          loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
          optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

      running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
      running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장

    # 한 epoch이 모두 종료되었을 때,
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_acc / len(dataloaders[phase].dataset)

    print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
    if phase == "val" and best_val_accuracy < epoch_acc: # phase가 val일 때, best accuracy 계산
      best_val_accuracy = epoch_acc
    if phase == "val" and best_val_loss < epoch_loss: # phase가 val일 때, best loss 계산
      best_val_loss = epoch_loss
print("학습 종료!")
print(f"최고 accuracy : {best_val_accuracy}, 최고 낮은 loss : {best_val_loss}")

torch.save(model.state_dict(), "model_A.pth")
print("Saved PyTorch Model State to model_A.pth")







# watch -d -n 0.5 nvidia-smi

# import torch
# a = torch.zero(300000000, dtype=torch.int8, device='cuda')
# b = torch.zero(300000000, dtype=torch.int8, device='cuda')
# # Check GPU memory using nvidia-smi
# del a
# torch.cuda.empty_cache()
# # Check GPU memory again

# RuntimeError: Tensor for 'out' is on CPU, Tensor for argument
#1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)




'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(MD_train_loader, model, loss_fn, optimizer)
    val(MD_val_loader, model, loss_fn)
print("Done!")
'''