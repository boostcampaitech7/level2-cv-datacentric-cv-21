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
os.environ['SM_MODEL_DIR'] = '/data/ephemeral/home/github'
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
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['adam', 'adamW'])
    parser.add_argument("--scheduler", type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument("--resume", type=str, default=None, choices=[None, 'resume', 'finetune'])
    parser.add_argument('--no-color_jitter', action='store_false', dest='color_jitter', default=True, help='Disable color jitter augmentation (default: True)')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize', default=True, help='Disable normalization (default: True)')
    parser.add_argument('--apply_flip', action='store_true', help='Apply horizontal flip (default: False)')
    parser.add_argument('--apply_rotate', action='store_true', help='Apply random rotate 90 (default: False)')
    parser.add_argument('--apply_blur', action='store_true', help='Apply Gaussian blur (default: False)')
    parser.add_argument('--apply_enlarge', action='store_true', help='Apply Enlarge image (default: False)')
    parser.add_argument('--apply_brightness', action='store_true', help='Apply Random Brightness image (default: False)')
    parser.add_argument('--apply_padding', action='store_true', help='Apply Padding image (default: False)')
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.environ.get('SM_MODEL_DIR', 'trained_models'), 'saved_models'),
                        help='Directory to save models')
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def create_transforms(args):
    funcs = []

    if args.color_jitter:
        funcs.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    if args.apply_flip:
        funcs.append(A.HorizontalFlip(p=0.5))

    if args.apply_rotate:
        funcs.append(A.RandomRotate90(p=0.5))

    if args.apply_blur:
        funcs.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    def conditional_resize(image, scale_limit=(1.2, 1.5), **kwargs):
        h, w = image.shape[:2]
        if h < args.input_size or w < args.input_size: # h나 w가 둘 중에 하나라도 input_size(default:1024)보다 작으면 1.2~1.5배 사이즈 키우기
            transform = A.RandomScale(scale_limit=scale_limit, p=1.0)
            image = transform(image=image)['image']
        return image
    
    if args.apply_enlarge:
        funcs.append(A.Lambda(image=conditional_resize))

    if args.apply_brightness: # 밝기와 대비를 +-20% 범위 내에서 무작위로 조절, p=0.5: 50% 확률로 밝기와 대비 조정
        funcs.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
    
    def dynamic_padding(image, **kwargs):
        h, w = image.shape[:2]
        padding_ratio = random.uniform(0, 0.2) # 0~20% 패딩 추가
        padding_height = int(h*padding_ratio)
        padding_width = int(w*padding_ratio)
        pad_transform = A.PadIfNeeded(
            min_height = h + padding_height,
            min_width = w + padding_width,
            border_mode = 0,
            value = (255, 255, 255)) # 하얀색 패딩
        return pad_transform(image=image)['image']

    if args.apply_padding:
        funcs.append(A.Lambda(image=dynamic_padding))
        funcs.append(A.Resize(args.input_size, args.input_size)) # 패딩을 하게 되면 torch size에 오류가 생겨서 다시 resize 해줌

    if args.normalize:
        funcs.append(A.Normalize(mean=(0.6831708235495132, 0.6570838514500981, 0.6245893701608299),
                                 std=(0.19835448743425943, 0.20532970462804873, 0.21117810051894778)))

    transform = A.Compose(funcs)
    return transform


def do_training(args):
    for data_dir, train_dataset_dir in zip(args.data_dirs, args.train_dataset_dirs):
        dataset_name = osp.basename(data_dir)  # 데이터셋 이름 추출
        save_dir = osp.join(args.save_dir, dataset_name)  # 데이터셋별 저장 경로 생성
        os.makedirs(save_dir, exist_ok=True)

        transform = create_transforms(args)
        model = EAST().to(args.device)
        optimizer = optim(args.optimizer, args.learning_rate, model.parameters())
        scheduler = sched(args, optimizer)

        # WandB 초기화 및 watch
        if args.mode == 'on':
            wandb.init(
                project=args.project,
                entity='cv-21',
                group=osp.basename(data_dir),
                name=f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}_{dataset_name}'
            )
            wandb.config.update(args)
            wandb.watch(model)

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
        
        best_f1_score = 0
        for epoch in range(args.max_epoch):
            model.train()
            with tqdm(total=train_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                    img = [image.cpu().numpy().transpose(1, 2, 0) if isinstance(image, torch.Tensor) else image for image in img]
                    img = [transform(image=image)['image'] for image in img]
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
