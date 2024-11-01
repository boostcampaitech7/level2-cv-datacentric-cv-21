import os
import os.path as osp
import time
import math
import json
import random
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from optimizer import optim
from scheduler import sched
from tqdm import tqdm
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset, PickleDataset
from model import EAST
from deteval import calc_deteval_metrics
from utils import get_gt_bboxes, get_pred_bboxes, seed_everything, AverageMeter

import albumentations as A
import numpy as np

def parse_args():
    parser = ArgumentParser()

    # 여러 피클 데이터셋 경로
    parser.add_argument('--train_dataset_dirs', nargs='+', type=str, default=[
        "/data/ephemeral/home/data/chinese_receipt/pickle/[1024, 1536, 2048]_cs[1024]_aug['CJ', 'GB', 'HSV', 'N']/train",
        "/data/ephemeral/home/data/japanese_receipt/pickle/[1024, 1536, 2048]_cs[1024]_aug['CJ', 'GB', 'HSV', 'N']/train",
        "/data/ephemeral/home/data/thai_receipt/pickle/[1024, 1536, 2048]_cs[1024]_aug['CJ', 'GB', 'HSV', 'N']/train",
        "/data/ephemeral/home/data/vietnamese_receipt/pickle/[1024, 1536, 2048]_cs[1024]_aug['CJ', 'GB', 'HSV', 'N']/train"
    ])
    parser.add_argument('--data_dirs', nargs='+', type=str, default=[
        "/data/ephemeral/home/data/chinese_receipt",
        "/data/ephemeral/home/data/japanese_receipt",
        "/data/ephemeral/home/data/thai_receipt",
        "/data/ephemeral/home/data/vietnamese_receipt"
    ])
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--device', default='cuda:0' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    parser.add_argument('-m', '--mode', type=str, default='on', help='wandb logging mode(on: online, off: disabled)')
    parser.add_argument('-p', '--project', type=str, default='datacentric', help='wandb project name')
    parser.add_argument('-d', '--data', default='pickle', type=str, help='description about dataset', choices=['original', 'pickle'])
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['adam', 'adamW'])
    parser.add_argument("--scheduler", type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument("--resume", type=str, default=None, choices=[None, 'resume', 'finetune'])
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed, fold, val_interval,
                optimizer, scheduler, resume):
    print("Training started...")  # 시작 지점 출력

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset Initialization
    print("Initializing dataset...")
    dataset = SceneTextDataset(
        root_dir=data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        color_jitter=True,
        normalize=True
    )
    print(f"Dataset initialized with {len(dataset)} samples.")
    dataset = EASTDataset(dataset)
    
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device(device)
    model = EAST().to(device)

    # Optimizer setup
    optimizer = optim(optimizer, learning_rate, model.parameters())

    # Scheduler setup
    if scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

    # Resume training if specified
    if resume == 'resume' and osp.exists(osp.join(model_dir, 'latest.pth')):
        model.load_state_dict(torch.load(osp.join(model_dir, 'latest.pth')))
        print("Resumed training from latest checkpoint.")
    elif resume == 'finetune' and osp.exists(osp.join(model_dir, 'latest.pth')):
        model.load_state_dict(torch.load(osp.join(model_dir, 'latest.pth')), strict=False)
        print("Finetuning from latest checkpoint.")

    ### WandB ###
    wandb.init(project='level2OCR', config={'learning_rate': learning_rate, 'optimizer': optimizer})
    wandb.watch(model)

    # Training loop
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
    wandb.finish()

def main():
    wandb.init(project='level2OCR', config=default_config)  # Sweep 설정을 위한 기본 설정
    do_training()

if __name__ == '__main__':
    print("Starting train.py...")
    main()