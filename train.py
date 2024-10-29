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
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from utils import increment_path

import wandb
import numpy as np


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--seed", type=int, default=4669)
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--model_dir", type=str, default="trained_models")
    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--ignore_tags", type=list, default=["masked", "excluded-region", "maintable", "stamp"])
    parser.add_argument("--wandb_name", type=str, default="default_run_name")
    parser.add_argument("--validation_split", type=float, default=0.2)  # Validation split 비율 추가
    parser.add_argument("--augmentation", type=int, default=0)
    parser.add_argument("--binarization", type=int, default=0)
    parser.add_argument("--color_jitter", type=int, default=0)
    parser.add_argument("--normalize", type=int, default=0)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def do_training(config, seed, dataset_path, model_dir, device, image_size, input_size, num_workers, batch_size,
                patience, learning_rate, max_epochs, save_interval, ignore_tags, wandb_name, validation_split, augmentation, binarization,
                color_jitter, normalize):
    # 시드 설정
    if seed == -1:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    print(f"seed: {seed}")
    seed_everything(seed)

    # 모델 디렉토리 설정
    model_dir = increment_path(os.path.join(model_dir, wandb_name))
    os.makedirs(model_dir, exist_ok=True)
    model_name = osp.basename(model_dir)

    # config.json 로그 파일 저장
    with open(os.path.join(model_dir, f"{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=4)

    # wandb 로깅 설정
    wandb.init(project="level2_data_centric", name=model_name, config=config)

    # 데이터셋 초기화
    full_dataset = SceneTextDataset(
        dataset_path,
        split="train",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        augmentation=augmentation,
        binarization=binarization,
        color_jitter=color_jitter,
        normalize=normalize
    )
    full_dataset = EASTDataset(full_dataset)

    # 학습 및 검증 데이터셋 분할
    train_size = int((1 - validation_split) * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    valid_num_batches = math.ceil(len(valid_dataset) / batch_size)

    # 데이터 로더 설정
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 모델, 옵티마이저, 스케줄러 설정
    model = EAST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epochs // 2], gamma=0.1)

    # Early stopping 초기화
    counter = 0
    best_val_loss = np.inf

    # ========== 학습 시작 ==========
    for epoch in range(max_epochs):
        model.train()
        train_loss, valid_loss, train_start = 0, 0, time.time()
        for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            train_loss += loss_val

            # CLI 로그
            print(
                f"Epoch[{epoch+1:02}/{max_epochs}]({idx+1:02}/{len(train_loader)}) || "
                f"Learning Rate: {scheduler.get_last_lr()[0]} || "
                f"Train Loss: {loss_val:4.4f} || "
                f"Train Class loss: {extra_info['cls_loss']:4.4f} || "
                f"Train Angle loss: {extra_info['angle_loss']:4.4f} || "
                f"Train IoU loss: {extra_info['iou_loss']:4.4f}"
            )

            # wandb 로그
            wandb.log({
                "Train Cls loss": extra_info['cls_loss'],
                "Train Angle loss": extra_info['angle_loss'],
                "Train IoU loss": extra_info['iou_loss'],
                "Train Loss": loss_val,
                "Learning Rate": scheduler.get_last_lr()[0],
                "Epoch": epoch + 1,
                "Seed": seed
            })

        train_end = time.time() - train_start

        # ========== 검증 ==========
        model.eval()
        with torch.no_grad():
            valid_start = time.time()
            for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(valid_loader):
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                valid_loss += loss.item()

            mean_val_loss = valid_loss / valid_num_batches
            if best_val_loss > mean_val_loss:
                best_val_loss = mean_val_loss
                ckpt_fpath = osp.join(model_dir, f"best_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_fpath)
                counter = 0
            else:
                counter += 1

        valid_end = time.time() - valid_start
        print(f"Valid Mean loss: {mean_val_loss:.4f}")

        if counter > patience:
            print("Early stopping triggered.")
            break

        scheduler.step()


def main(args):
    languages = ["chinese_receipt", "japanese_receipt", "thai_receipt", "vietnamese_receipt"]
    
    for language in languages:
        print(f"Starting training for {language}")
        dataset_path = osp.join(args.data_dir, language)
        
        # wandb 이름을 언어별로 지정하여 구분 가능하도록 설정
        args.wandb_name = f"{language}_run"
        
        # 각 언어 데이터셋에 대해 학습 진행
        do_training(
            config=args,
            seed=args.seed,
            dataset_path=dataset_path,
            model_dir=args.model_dir,
            device=args.device,
            image_size=args.image_size,
            input_size=args.input_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            save_interval=args.save_interval,
            ignore_tags=args.ignore_tags,
            wandb_name=args.wandb_name,
            validation_split=args.validation_split,
            augmentation=args.augmentation,
            binarization=args.binarization,
            color_jitter=args.color_jitter,
            normalize=args.normalize
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)