import argparse
import logging
from threading import local
import torch
import torch.nn.functional as F
import torch.utils.data.distributed
from config import config as cfg
from torch.utils.data import DataLoader
from dataset import HaGrid
from torchvision.datasets import ImageFolder

from backbones.iresnet import iresnet100, iresnet50
from backbones.efficientnetv2s import efficientnetv2s
import json
import pandas as pd


def main(args):

    testset = ImageFolder(
        root=args.data_dir,
        transform=cfg.transform
    )

    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # load evaluation

    if cfg.network == "efficientnet":
        backbone = efficientnetv2s().to("cuda:0")
    else:
        backbone = None
        logging.info("load backbone failed!")

    weight = torch.load(args.weights)
    backbone.load_state_dict(weight)
    backbone.eval()

    results = {}

    for batch, (img, _) in enumerate(test_dataloader):
        img = img.cuda()
        with torch.no_grad():
            _, qs = backbone(img)

        qs = qs.detach().cpu().numpy()
        # idnx = idx.detach().cpu().numpy()

        for i in range(qs.shape[0]):
            filename = test_dataloader.dataset.imgs[batch*32 + i][0]
            # filename = test_dataloader.dataset.images[idnx[i]]
            results[filename] = str(qs[i][0])
    
    data = pd.DataFrame(results.items())
    print(data.head())

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch PCR-HIQA sample quality extraction')
    parser.add_argument('--data_dir', type=str, help="input folder")
    parser.add_argument('--weights', type=str, help="path to the model's weight")
    parser.add_argument('--output_path', type=str, help="output json file")
    parser.add_argument('--hand', type=str, default='right',
                        choices=['right', 'left'],
                        help="gesture to train or to evaluate")
    args_ = parser.parse_args()
    main(args_)