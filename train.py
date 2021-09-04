import argparse
import glob
import json
import os
import random
import re
import copy
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt1 = gts[0][choice].item()
        gt2 = gts[1][choice].item()
        gt3 = gts[2][choice].item()
        pred1 = preds[0][choice].item()
        pred2 = preds[1][choice].item()
        pred3 = preds[2][choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        # gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        # pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip((gt1, gt2, gt3), (pred1, pred2, pred3), tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def encode_age(pred):
    for i, p in enumerate(pred):
        if p < 30:
            pred[i] = 0
        elif p < 57:
            pred[i] = 1
        else:
            pred[i] = 2
    ret = torch.tensor(pred)
    return ret

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    train_dataset_module = getattr(import_module("dataset"), args.dataset)
    train_set = train_dataset_module(
        data_dir=data_dir, transform='train',
    )
    valid_dataset_module = getattr(import_module("dataset"), args.dataset)
    val_set = valid_dataset_module(
        data_dir=data_dir, transform='valid',
    )
    num_classes = train_set.num_classes  # 18

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: EnsembleModel
    model = model_module(
        num_classes=num_classes,
        mode=args.mode
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: ensemble
    print(criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = OneCycleLR(optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.002, epochs=args.epochs, steps_per_epoch=len(train_loader))

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    patience = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        matches_age = [0, 0, 0]
        nums_age = [0, 0, 0]
        age_acc = [0, 0, 0]

        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            label1, label2, label3, age_label = labels
            label1, label2, label3, age_label = label1.to(device), label2.to(device), label3.to(device), age_label.to(device)

            optimizer.zero_grad()

            out1, out2, out3 = model(inputs)
            pred1 = torch.argmax(out1, dim=-1)
            pred2 = torch.argmax(out2, dim=-1)
            pred3 = torch.argmax(out3, dim=-1)
            if args.mode == 'reg':
                out3 = out3.squeeze()
                pred3 = encode_age(out3.tolist())
                pred3 = pred3.to(device)

            match1 = (pred1 == label1).sum().item()
            match2 = (pred2 == label2).sum().item()
            match3 = (pred3 == label3).sum().item()
            
            match_age = [0, 0, 0]
            num_age = [0, 0, 0]
            for p, l in zip(pred3, label3):
                for i in range(3):
                    if p==i and l==i:
                        match_age[i] += 1
            for i in range(3):
                num_age[i] = (label3==i).sum().item()
                nums_age[i] += num_age[i]
                if num_age[i] != 0:
                    match_age[i] = match_age[i]/num_age[i]
                matches_age[i] += match_age[i]


            loss1 = criterion[0](out1, label1)
            loss2 = criterion[1](out2, label2)
            if args.mode == 'reg':
                loss3 = criterion[2](out3, age_label)
                # print(age_label)
                # print(f"out3 : {out3}")
                # print(f"loss3 : {loss3}")
            else:
                loss3 = criterion[2](out3, label3)

            (loss1+loss2+loss3).backward()
            optimizer.step()
            scheduler.step()

            loss_value += ((loss1+loss2+loss3)/3.0).item()
            matches += (match1 + match2 + match3)/3.0
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                for i in range(3):
                    age_acc[i] = matches_age[i] / args.log_interval#
                    nums_age[i] /= args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"t loss {train_loss:4.4} || t acc {train_acc:4.2%} || lr {current_lr} || "
                    f"age0 acc {age_acc[0]:4.2%} || "
                    f"age1 acc {age_acc[1]:4.2%} || "
                    f"age2 acc {age_acc[2]:4.2%}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                for i in range(3):
                    nums_age[i] = 0
                    matches_age[i] = 0
    
        # scheduler.step()
        

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            matches_age = [0, 0, 0]
            nums_age = [0, 0, 0]
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                label1, label2, label3, age_label = labels
                label1, label2, label3, age_label = label1.to(device), label2.to(device), label3.to(device), age_label.to(device)


                out1, out2, out3 = model(inputs)
                pred1 = torch.argmax(out1, dim=-1)
                pred2 = torch.argmax(out2, dim=-1)
                pred3 = torch.argmax(out3, dim=-1)
                if args.mode == 'reg':
                    out3 = out3.squeeze()
                    pred3 = encode_age(out3.tolist())
                    pred3 = pred3.to(device)

                match1 = (pred1 == label1).sum().item()
                match2 = (pred2 == label2).sum().item()
                match3 = (pred3 == label3).sum().item()
                
                for p, l in zip(pred3, label3):
                    for i in range(3):
                        if p==i and l==i:
                            matches_age[i] += 1
                for i in range(3):
                    nums_age[i] += (label3==i).sum().item()

                loss1 = criterion[0](out1, label1)
                loss2 = criterion[1](out2, label2)
                if args.mode == 'reg':
                    loss3 = criterion[2](out3, age_label)
                else:
                    loss3 = criterion[2](out3, label3)

                loss_item = ((loss1+loss2+loss3)/3.0).item()
                acc_item = (match1 + match2 + match3)/3.0

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)


                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = valid_dataset_module.denormalize_image(inputs_np, val_set.mean, val_set.std)
                    figure = grid_image(
                        inputs_np, labels, (pred1, pred2, pred3), n=16, shuffle=args.dataset != "TrainSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_age_acc = [0, 0, 0]
            for i in range(3):
                val_age_acc[i] = matches_age[i] / nums_age[i]
                print(matches_age[i], nums_age[i])
            val_acc = np.sum(val_acc_items) / len(val_set)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                print(patience)

            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"age0 acc : {val_age_acc[0]:4.2%} || "
                f"age1 acc : {val_age_acc[1]:4.2%} || "
                f"age2 acc : {val_age_acc[2]:4.2%} "
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()

        if patience == args.patience:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    import os

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--dataset', type=str, default='TrainDataset', help='dataset augmentation type (default: TrainDataset)')
    # parser.add_argument('--augmentation', type=str, default='Augmentation', help='data augmentation type (default: Augmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[300, 300], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='EnsembleModel', help='model type (default: EnsembleModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='ensemble', help='criterion type (default: ensemble)')
    # parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int, default=10, help='check early stopping point (default: 10)')
    parser.add_argument('--mode', type=str, default='default', help='select mode')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)

