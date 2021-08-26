from dataset import *
from model import *

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

train_image_dir = '/opt/ml/input/data/train/images'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is using!")

LEARNING_RATE = 0.0001
NUM_EPOCH = 5
BATCH_SIZE = 64


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
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
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss


data_transforms = transforms.Compose([transforms.ToTensor()])


########## Model_M ##########
MD = MaskDataset(train_image_dir, transform=data_transforms)
train_data_length = int(len(MD) * 0.8)
val_data_length = int(len(MD) * 0.2)
MD_train, MD_val = random_split(MD, [train_data_length, val_data_length])
MD_train_loader = DataLoader(MD_train, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
MD_val_loader = DataLoader(MD_val, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, drop_last=True)

model = Model_M().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

lowest_val_loss = 99999
for t in range(NUM_EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    torch.save(model.state_dict(), "model_M.pth")
    train(MD_train_loader, model, loss_fn, optimizer)
    current_val_loss = val(MD_val_loader, model, loss_fn)
    if current_val_loss < lowest_val_loss:
        lowest_val_loss = current_val_loss
    else:
        print('----------Early Stopping----------')
        print("Saved PyTorch Model State to model_M.pth")
        break
else:
    torch.save(model.state_dict(), "model_M.pth")


########## Model_G ##########
GD = GenderDataset(train_image_dir, transform=data_transforms)
GD_train, GD_val = random_split(GD, [train_data_length, val_data_length])
GD_train_loader = DataLoader(GD_train, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
GD_val_loader = DataLoader(GD_val, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, drop_last=True)

model = Model_G().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

lowest_val_loss = 99999
for t in range(NUM_EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    torch.save(model.state_dict(), "model_G.pth")
    train(GD_train_loader, model, loss_fn, optimizer)
    current_val_loss = val(GD_val_loader, model, loss_fn)
    if current_val_loss < lowest_val_loss:
        lowest_val_loss = current_val_loss
    else:
        print('----------Early Stopping----------')
        print("Saved PyTorch Model State to model_G.pth")
        break
else:
    torch.save(model.state_dict(), "model_G.pth")


########## Model_A ##########
AD = GenderDataset(train_image_dir, transform=data_transforms)
AD_train, AD_val = random_split(AD, [train_data_length, val_data_length])
AD_train_loader = DataLoader(AD_train, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
AD_val_loader = DataLoader(AD_val, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, drop_last=True)

model = Model_A().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

lowest_val_loss = 99999
for t in range(NUM_EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    torch.save(model.state_dict(), "model_A.pth")
    train(AD_train_loader, model, loss_fn, optimizer)
    current_val_loss = val(AD_val_loader, model, loss_fn)
    if current_val_loss < lowest_val_loss:
        lowest_val_loss = current_val_loss
    else:
        print('----------Early Stopping----------')
        print("Saved PyTorch Model State to model_A.pth")
        break
else:
    torch.save(model.state_dict(), "model_A.pth")
