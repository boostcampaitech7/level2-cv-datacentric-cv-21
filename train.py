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
from tqdm import tqdm
import wandb

# Function from "baseline" folder (수정이 불가한 내용들)
from baseline.east_dataset import EASTDataset
from baseline.model import EAST
from baseline.loss import EASTLoss

# Funtction from code (수정이 가능한 내용들)
from deteval import calc_deteval_metrics
from utils import get_gt_bboxes, get_pred_bboxes, seed_everything, AverageMeter
from dataset import SceneTextDataset, PickleDataset
from optimizer import optim
from scheduler import sched

import albumentations as A
import numpy as np
os.environ['SM_MODEL_DIR'] = '/data/ephemeral/home/github'

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--per_lang', action='store_true', help='If true, conduct experiment language-wise')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--seed', type=int, default=4096)
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
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['adam', 'AdamW'])
    parser.add_argument("--scheduler", type=str, default='multistep', choices=['multistep', 'cosine'])
    parser.add_argument("--resume", type=str, default=None, choices=[None, 'resume', 'finetune'])
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.environ.get('SM_MODEL_DIR', 'trained_models'), 'saved_models'),
                        help='Directory to save models')
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args




def do_training(args):
    if args.per_lang:
    # 다음의 경로를 수정해주세요 : Ln 71, 77, 83
        train_dataset_dirs=[
            "/data/ephemeral/home/data/japanese_receipt/pickle/[1024]_cs[1024]_aug[]/train",
        ]
        data_dirs=[
            "/data/ephemeral/home/data/japanese_receipt",
        ]
    else:
        train_dataset_dirs=["/data/ephemeral/home/data_synth/pickle/[1024]_cs[1024]_aug['CJ', 'N']/train"]
        data_dirs=["/data/ephemeral/home/data_synth/"]
    for data_dir, train_dataset_dir in zip(data_dirs, train_dataset_dirs):
        if args.per_lang:
            dataset_name = osp.basename(data_dir)  # 데이터셋 이름 추출
        else:
            dataset_name = "not-language-wise/[1024]_cs[1024]_aug['CJ', 'N']" # 실험마다 수정해야함
        save_dir = osp.join(args.save_dir, dataset_name)  # 데이터셋별 저장 경로 생성
        os.makedirs(save_dir, exist_ok=True)

        model = EAST().to(args.device)
        optimizer = optim(args.optimizer, args.learning_rate, model.parameters())
        scheduler = sched(args, optimizer)

        # WandB 초기화 및 watch
        if args.mode == 'on':
            wandb.init(
                project=args.project,
                entity='cv-21',
                group=osp.basename(data_dir),
                name=f"{dataset_name}"
            )
            wandb.config.update(args)
            wandb.watch(model)

        if args.data == 'pickle':
            train_dataset = PickleDataset(train_dataset_dir)
        else:
            train_dataset = SceneTextDataset(
                data_dir,
                split='train',
                fold=args.fold,
                per_lang=args.per_lang,
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
        if args.per_lang:
            with open(osp.join(root_dir, f'ufo/valid{args.fold}.json'), 'r') as f:
                val_data = json.load(f)
        else:
            _lang_list = ['chinese', 'japanese', 'thai', 'vietnamese']
            total_anno = dict(images=dict())
            for nation in _lang_list:
                with open(osp.join(data_dir, f'{nation}_receipt/ufo/valid{args.fold}.json'), 'r', encoding='utf-8') as f:
                    anno = json.load(f)
                for im in anno['images']:
                    total_anno['images'][im] = anno['images'][im]
            val_data = total_anno
        val_images = []
        val_images = list(val_data['images'].keys())

        best_f1_score = 0
        for epoch in range(args.max_epoch):
            model.train()
            with tqdm(total=train_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                    img = [image.cpu().numpy().transpose(1, 2, 0) if isinstance(image, torch.Tensor) else image for image in img]
                    img = [torch.from_numpy(image).permute(2, 0, 1) for image in img]
                    img = torch.stack(img)
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

            if (epoch + 1) % args.val_interval == 0 or epoch >= args.max_epoch - 5:
                print("Calculating validation results...")
                pred_bboxes_dict = get_pred_bboxes(model, data_dir, val_images, args.input_size, args.batch_size, split='train')
                gt_bboxes_dict = get_gt_bboxes(dict_from_json=val_data, valid_images=val_images)
                result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
                precision, recall = result['total']['precision'], result['total']['recall']
                f1_score = 2*precision*recall/(precision+recall) if precision + recall > 0 else 0
                print(f'Precision: {precision} Recall: {recall} F1 Score: {f1_score}')
                print(f'Epoch: {epoch}')

                val_dict = {'val precision': precision, 'val recall': recall, 'val f1_score': f1_score}
                if args.mode == 'on':
                    wandb.log(val_dict, step=epoch)

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    torch.save(model.state_dict(), osp.join(save_dir, 'best.pth'))

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
