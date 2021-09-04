import argparse
import os
from importlib import import_module

import pandas as pd
import torch

from dataset import TestDataset, TrainDataset


def load_model(saved_model, num_classes, mode, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes,
        mode=mode
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def encode_age(pred):
    for i, p in enumerate(pred):
        if p < 30:
            pred[i] = 0
        elif p < 60:
            pred[i] = 1
        else:
            pred[i] = 2
    ret = torch.tensor(pred)
    return ret

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = TrainDataset.num_classes  # 18
    mode = args.mode
    model = load_model(model_dir, num_classes, mode, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
    )

    print("Calculating inference results..")
    pred1, pred2, pred3 = [], [], []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            out1, out2, out3 = model(images)
            out3 = out3.squeeze()
            out1 = out1.argmax(dim=-1)
            out2 = out2.argmax(dim=-1)
            
            if args.mode == 'reg':
                out3 = encode_age(out3.tolist())
            else:
                out3 = out3.argmax(dim=-1)

            pred1.extend(out1.cpu().numpy())
            pred2.extend(out2.cpu().numpy())
            pred3.extend(out3.cpu().numpy())
            
    answer = [6 * p1 + 3 * p2 + p3 for p1, p2, p3 in zip(pred1, pred2, pred3)]
    info['ans'] = answer
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--resize', type=tuple, default=(300, 300), help='resize size for image when you trained (default: (300, 300))')
    parser.add_argument('--model', type=str, default='EnsembleModel', help='model type (default: EnsembleModel)')
    parser.add_argument('--mode', type=str, default='default', help='mode type (default: default)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/reg'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)

