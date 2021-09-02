# https://github.com/TsykunovDmitriy/retinaface_lightweight.pytorch
# pip install git+https://github.com/TsykunovDmitriy/retinaface_lightweight.pytorch

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from tqdm import tqdm
import shutil
from retinaface import RetinaDetector


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"using {device}")

detector = RetinaDetector(device=0, score_thresh=0.9, top_k=100, nms_thresh=0.4, use_trt=False)
train_data_dir = '/opt/ml/input/data/train/images'
new_train_data_dir = '/opt/ml/input/data/train/retina_cropped_images'
eval_data_dir = '/opt/ml/input/data/eval/images'
new_eval_data_dir = '/opt/ml/input/data/eval/retina_cropped_images'

if not os.path.exists(new_train_data_dir):
    os.mkdir(new_train_data_dir)
if not os.path.exists(new_eval_data_dir):
    os.mkdir(new_eval_data_dir)


# create new retinaface cropped train images
profiles = os.listdir(train_data_dir)
center_crop_count = 0
success_count = 0
choose_one_count = 0
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

        # apply retinaface
        bboxes, landmarks, scores = detector(img)

        if bboxes.shape[0] == 0:
            center_crop_count += 1
            xmin = 50
            ymin = 100
            xmax = 350
            ymax = 400
        elif bboxes.shape[0] == 1:
            success_count += 1
            xmin = int(bboxes[0][0]) - 20
            ymin = int(bboxes[0][1]) - 30
            xmax = int(bboxes[0][2]) + 20
            ymax = int(bboxes[0][3]) + 30
        else:
            choose_one_count += 1
            xmin = int(bboxes[0][0]) - 20
            ymin = int(bboxes[0][1]) - 30
            xmax = int(bboxes[0][2]) + 20
            ymax = int(bboxes[0][3]) + 30

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > 384: xmax = 384
        if ymax > 512: ymax = 512

        retina_cropped_img = img[ymin:ymax, xmin:xmax, :]

        new_img_folder = os.path.join(new_train_data_dir, profile)

        if not os.path.exists(new_img_folder):
            os.mkdir(new_img_folder)

        plt.imsave(os.path.join(new_img_folder, file_name), retina_cropped_img)
        total += 1

print(f"center crop : {center_crop_count}")
print(f"success : {success_count}")
print(f"choose one : {choose_one_count}")
print(f"total : {total}")


# create new retinaface cropped eval images
profiles = os.listdir(eval_data_dir)
center_crop_count = 0
success_count = 0
choose_one_count = 0
total = 0

for profile in tqdm(profiles):
    if profile.startswith("."):
        continue

    img_path = os.path.join(eval_data_dir, profile)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply retinaface
    bboxes, landmarks, scores = detector(img)

    if bboxes.shape[0] == 0:
        center_crop_count += 1
        xmin = 50
        ymin = 100
        xmax = 350
        ymax = 400
    elif bboxes.shape[0] == 1:
        success_count += 1
        xmin = int(bboxes[0][0]) - 20
        ymin = int(bboxes[0][1]) - 30
        xmax = int(bboxes[0][2]) + 20
        ymax = int(bboxes[0][3]) + 30
    else:
        choose_one_count += 1
        xmin = int(bboxes[0][0]) - 20
        ymin = int(bboxes[0][1]) - 30
        xmax = int(bboxes[0][2]) + 20
        ymax = int(bboxes[0][3]) + 30

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > 384: xmax = 384
    if ymax > 512: ymax = 512

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    retina_cropped_img = img[ymin:ymax, xmin:xmax, :]
    new_img_path = os.path.join(new_eval_data_dir, profile)
    plt.imsave(new_img_path, retina_cropped_img)
    total += 1

print(f"center crop : {center_crop_count}")
print(f"success : {success_count}")
print(f"choose one : {choose_one_count}")
print(f"total : {total}")

# create zip files
# shutil.make_archive("retina_cropped_train_images", 'zip', new_train_data_dir)
# shutil.make_archive("retina_cropped_eval_images", 'zip', new_eval_data_dir)
