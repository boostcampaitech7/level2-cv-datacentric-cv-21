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

def do_training(args):
    
    for data_dir, train_dataset_dir in zip(args.data_dirs, args.train_dataset_dirs):
        
        ### Train Loader ###
        if args.data == 'pickle':
            train_dataset = PickleDataset(train_dataset_dir)
        else:
            train_dataset = SceneTextDataset(
                data_dir,
                split='train',
                json_name=f'train{args.fold}.json',
                image_size=args.image_size,
                crop_size=args.input_size,
                ignore_tags=args.ignore_tags,
                pin_memory=True,
            )
            train_dataset = EASTDataset(train_dataset)
        
        train_num_batches = math.ceil(len(train_dataset) / args.batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        ### Val Loader ###
        valid_json_file = f'ufo/valid{args.fold}.json'
        val_images = []
        if osp.exists(osp.join(data_dir, valid_json_file)):
            with open(osp.join(data_dir, valid_json_file), 'r', encoding='utf-8') as file:
                val_data = json.load(file)
            val_images = list(val_data['images'].keys())

        model = EAST().to(args.device)
        
        ### Resume or finetune ###
        save_dir = osp.join(args.model_dir, f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}_{osp.basename(data_dir)}')
        os.makedirs(save_dir, exist_ok=True)

        if args.resume == "resume" and osp.exists(osp.join(save_dir, "latest.pth")):
            checkpoint = torch.load(osp.join(save_dir, "latest.pth"))
            model.load_state_dict(checkpoint)
        elif args.resume == "finetune" and osp.exists(osp.join(save_dir, "best.pth")):
            checkpoint = torch.load(osp.join(save_dir, "best.pth"))
            model.load_state_dict(checkpoint)
        
        optimizer = optim(args, filter(lambda p: p.requires_grad, model.parameters()))
        scheduler = sched(args, optimizer)
        
        ### WandB ###
        if args.mode == 'on':
            wandb.init(
                project=args.project,
                entity='cv-21',
                group=osp.basename(data_dir),
                name=f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}_{osp.basename(data_dir)}'
            )
            wandb.config.update(args)
            wandb.watch(model)
        
        ### Training Loop ###
        best_f1_score = 0
        for epoch in range(args.max_epoch):
            model.train()
            with tqdm(total=train_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_dict = {
                        'train total loss': loss.item(),
                        'train cls loss': extra_info['cls_loss'],
                        'train angle loss': extra_info['angle_loss'],
                        'train iou loss': extra_info['iou_loss']
                    }
                    pbar.update(1)
                    pbar.set_postfix(train_dict)
                    if args.mode == 'on':
                        wandb.log(train_dict, step=epoch)

            scheduler.step()

            ### Validation (Every val_interval or last 5 epochs) ###
            if (epoch + 1) % args.val_interval == 0 or epoch >= args.max_epoch - 5:
                print("Calculating validation results...")
                pred_bboxes_dict = get_pred_bboxes(model, data_dir, val_images, args.input_size, args.batch_size, split='train')            
                gt_bboxes_dict = get_gt_bboxes(data_dir, json_file=valid_json_file, valid_images=val_images)
                result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
                precision, recall = result['total']['precision'], result['total']['recall']
                f1_score = 2*precision*recall/(precision+recall) if precision + recall > 0 else 0
                print(f'Precision: {precision} Recall: {recall} F1 Score: {f1_score}')

                val_dict = {'val precision': precision, 'val recall': recall, 'val f1_score': f1_score}
                if args.mode == 'on':
                    wandb.log(val_dict, step=epoch)
                
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    torch.save(model.state_dict(), osp.join(save_dir, 'best.pth'))

            # Save latest model
            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), osp.join(save_dir, 'latest.pth'))

        if args.mode == 'on':
            wandb.alert('Training Task Finished', f"Best F1 Score: {best_f1_score:.4f}")
            wandb.finish()

def main(args):
    do_training(args)

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)
