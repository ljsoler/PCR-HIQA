import argparse
import logging
import os
from threading import local
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

import losses
from config import config as cfg
from dataset import HaGrid, DataLoaderX
from utils.utils_callbacks import CallBackEDC, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50
from backbones.efficientnetv2s import efficientnetv2s


torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(args.output_path) and rank == 0:
        os.makedirs(args.output_path)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, args.output_path)

    trainset = HaGrid(cfg.rec, args.perception_file, args.hand, human_perception=args.perception, transform=cfg.transform, isTraining=True)

    valset = HaGrid(cfg.val_targets, args.perception_file, args.hand, human_perception=False, transform=cfg.transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)

    # load evaluation
    if cfg.network == "efficientnet":
        backbone = efficientnetv2s().to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")

    if args.resume:
        try:
            backbone_pth = os.path.join(args.output_path, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone student loaded successfully!")

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    # get header
    if args.loss == "CR_FIQA_LOSS":
        header = losses.CR_FIQA_LOSS(in_features=cfg.embedding_size, out_features=trainset.number_identities(), s=cfg.s, m=cfg.m).to(local_rank)
    else:
        print("Header not implemented")
    if args.resume:
        try:
            header_pth = os.path.join(cfg.identity_model, str(cfg.identity_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)        

    criterion = CrossEntropyLoss()
    criterion_qs= torch.nn.SmoothL1Loss(beta=0.5)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    callback_quality = CallBackEDC(cfg.eval_step, valset)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, args.output_path)
    lamb=10.0  #10.0
    loss = AverageMeter()
    global_step = cfg.global_step
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for _, (img, label, pqs) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            pqs = pqs.cuda(local_rank, non_blocking=True)
            pqs = torch.unsqueeze(pqs, dim=1)

            features, qs = backbone(img)
            thetas, _, ccs, _ = header(features, label)
            qsl = args.alpha * ccs + (1 - args.alpha)*pqs
            loss_qs=criterion_qs(qsl, qs)
            loss_v = criterion(thetas, label) + lamb* loss_qs
            loss_v.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            
            callback_logging(global_step, loss, epoch, 0,loss_qs)
            callback_quality(global_step, backbone)

        scheduler_backbone.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone, header)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch PCR-FIQA Training')
    parser.add_argument('--output_path', type=str, help="output dir")
    parser.add_argument('--perception', action='store_true', default=True,
                    help='use the combination between recognition utility and quality human percpetion')
    parser.add_argument('--hand', type=str, default='right',
                        choices=['right', 'left'],
                        help="gesture to train or to evaluate")
    parser.add_argument('--use_mask', action='store_true', default=False,
                    help='use the segmentation map of hands')
    parser.add_argument('--perception_file', type=str, default="/home/fbi1532/Projects/ISO_IEC-29794-5/python-code/results/ISO_HaGrid_Training.csv", help="ISO quality file")
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter to combine recognition utility and human perception')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--loss', type=str, default="CR_FIQA_LOSS", help="loss function")
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)