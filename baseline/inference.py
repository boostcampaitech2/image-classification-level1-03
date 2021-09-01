import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """ """
    phase_to_num_classses_dict = {"Mask" : 3, "Gender" : 2, "Age" : 3}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskBaseDataset.num_classes  # 18
    num_classes = phase_to_num_classses_dict[args.phase]
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, "cropped_images")
    submission_path = os.path.join(data_dir, "info.csv")
    result_path = os.path.join(data_dir, "result.csv")
    submission = pd.read_csv(submission_path)
    result = pd.read_csv(result_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in submission.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print(f"Calculating {args.phase} inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    result[args.phase] = preds
    result.to_csv(os.path.join(data_dir, f"result.csv"), index=False)

    if args.phase == "Age":
        result = pd.read_csv(result_path)
        mask = result["Mask"]
        gender = result["Gender"]
        age = result["Age"]
        all_preds = [6 * M + 3 * G + A for M, G, A in zip(mask, gender, age)]
        submission["ans"] = all_preds
        submission.to_csv(os.path.join(output_dir, f"output.csv"), index=False)
        print(f"Inference Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--batch_size",type=int,default=64,help="input batch size for validing (default: 64)",)
    parser.add_argument("--resize",type=tuple,default=(224, 224),help="resize size for image when you trained (default: (96, 128))",)
    parser.add_argument("--model", type=str, default="VitBasePatch16_224", help="model type (default: ResNet50)")
    parser.add_argument("--phase", type=str, default="Mask", help="phase type (default: Mask)")

    # Container environment
    parser.add_argument("--data_dir",type=str,default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "./model"))
    parser.add_argument("--output_dir",type=str,default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)