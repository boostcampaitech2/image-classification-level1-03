# pip install facenet-pytorch

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from tqdm import tqdm
import shutil


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)
train_data_dir = '/opt/ml/input/data/train/images'
new_train_data_dir = '/opt/ml/input/data/train/mtcnn_cropped_images'
eval_data_dir = '/opt/ml/input/data/eval/images'
new_eval_data_dir = '/opt/ml/input/data/eval/mtcnn_cropped_images'

if not os.path.exists(new_train_data_dir):
    os.mkdir(new_train_data_dir)
if not os.path.exists(new_eval_data_dir):
    os.mkdir(new_eval_data_dir)


# create new train cropped images
profiles = os.listdir(train_data_dir)
center_crop_count = 0
total = 0

for profile in tqdm(profiles):
    if profile.startswith("."):
        continue

    img_folder = os.path.join(train_data_dir, profile)

    for file_name in os.listdir(img_folder):
        if file_name.startswith("."):
            continue

        img_path = os.path.join(img_folder, file_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply mtcnn
        boxes, probs = mtcnn.detect(img)
        # print(f"boxes : {boxes}, probs : {probs}")

        # center crop directly
        if not isinstance(boxes, np.ndarray):
            center_crop_count += 1
            img=img[100:400, 50:350, :]
        else:
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30

            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512

            img = img[ymin:ymax, xmin:xmax, :]

        new_img_folder = os.path.join(new_train_data_dir, profile)
        if not os.path.exists(new_img_folder):
            os.mkdir(new_img_folder)
        plt.imsave(os.path.join(new_img_folder, file_name), img)
        total += 1

print(f"center crop count : {center_crop_count}")
print(f"total : {total}")


# create new eval cropped images
profiles = os.listdir(eval_data_dir)
center_crop_count = 0
total = 0

for profile in tqdm(profiles):
    if profile.startswith("."):
        continue

    img_path = os.path.join(eval_data_dir, profile)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply mtcnn
    boxes, probs = mtcnn.detect(img)

    # center crop directly
    if not isinstance(boxes, np.ndarray):
        center_crop_count += 1
        img=img[100:400, 50:350, :]
    else:
        xmin = int(boxes[0, 0])-30
        ymin = int(boxes[0, 1])-30
        xmax = int(boxes[0, 2])+30
        ymax = int(boxes[0, 3])+30

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > 384: xmax = 384
        if ymax > 512: ymax = 512

        img = img[ymin:ymax, xmin:xmax, :]

    new_img_path = os.path.join(new_eval_data_dir, profile)
    plt.imsave(new_img_path, img)
    total += 1

print(f"center crop count : {center_crop_count}")
print(f"total : {total}")

# create zip files
# shutil.make_archive("train_cropped_images", 'zip', new_train_data_dir)
# shutil.make_archive("eval_cropped_images", 'zip', new_eval_data_dir)
