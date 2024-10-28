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
    parser.add_argument("--data_dir", type=str, default="../data/medical")
    parser.add_argument("--train_ann", type=str, default="train1")
    parser.add_argument("--valid_ann", type=str, default="valid1")
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
    parser.add_argument("--validation", type=int, default=1)
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
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def do_training(config, seed, data_dir, train_ann, valid_ann, model_dir, device, image_size, input_size, num_workers, batch_size,
                patience, learning_rate, max_epochs, save_interval, ignore_tags, wandb_name, validation, augmentation, binarization,
                color_jitter, normalize):
    if seed == -1:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    print(f"seed: {seed}")
    seed_everything(seed)  # set seed
    
    # checkpoint directory initialization
    model_dir = increment_path(os.path.join(model_dir, wandb_name))
    model_name = model_dir.split("/")[-1]
    
    # logging with config.json
    with open(os.path.join(model_dir, f"{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=4)
    
    # logging with wandb
    wandb.init(project="level2_data_centric",
               name=model_name,
               config=config)
    
    train_dataset = SceneTextDataset(
        data_dir,
        split=train_ann,
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        augmentation=augmentation,
        binarization=binarization,
        color_jitter=color_jitter,
        normalize=normalize
    )
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    if validation:
        valid_dataset = SceneTextDataset(
            data_dir,
            split=valid_ann,
            image_size=image_size,
            crop_size=input_size,
            ignore_tags=ignore_tags,
            normalize=normalize
        )
        valid_dataset = EASTDataset(valid_dataset)
        valid_num_batches = math.ceil(len(valid_dataset) / batch_size)
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epochs // 2], gamma=0.1)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, verbose=True)
    
    # early stopping
    counter = 0
    best_val_loss = np.inf
    
    # ========== train ==========
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

            # logging at CLI
            print(
                f"Epoch[{epoch+1:02}/{max_epochs}]({idx+1:02}/{len(train_loader)}) || "
                f"Learning Rate: {scheduler.get_last_lr()[0]} || "
                f"Train Loss: {loss_val:4.4f} || "
                f"Train Class loss: {extra_info['cls_loss']:4.4f} || "
                f"Train Angle loss: {extra_info['angle_loss']:4.4f} || "
                f"Train IoU loss: {extra_info['iou_loss']:4.4f}"
            )
            
            # logging with wandb
            wandb.log({
                "Cls loss": extra_info['cls_loss'],
                "Angle loss": extra_info['angle_loss'],
                "IoU loss": extra_info['iou_loss'],
                "loss": loss_val,
                "Epochs": epoch+1,
                "Learning Rate": scheduler.get_last_lr()[0],
                "Seed": seed
            })
        
        train_end = time.time() - train_start
        print("Train Mean loss: {:.4f} || Elapsed time: {} || ETA: {}".format(
            train_loss / train_num_batches,
            timedelta(seconds=train_end),
            timedelta(seconds=train_end*(max_epochs-epoch+1))))
        
        # ========== valid ==========
        if validation:
            model.eval()
            with torch.no_grad():
                valid_start = time.time()
                print("\nEvaluating validation results...")
                
                for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(valid_loader):
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    loss_val = loss.item()
                    valid_loss += loss_val
                    
                    # logging at CLI
                    print(
                        f"Valid Batch({idx+1}/{len(valid_loader)}) || "
                        f"Valid loss: {loss_val:4.4f} || "
                        f"Valid Class loss: {extra_info['cls_loss']:4.4f} || "
                        f"Valid Angle loss: {extra_info['angle_loss']:4.4f} || "
                        f"Valid IoU loss: {extra_info['iou_loss']:4.4f}"
                    )
                    
                    # logging with wandb
                    wandb.log({
                        "Valid loss": loss_val,
                        "Valid Cls loss": extra_info['cls_loss'],
                        "Valid Angle loss": extra_info['angle_loss'],
                        "Valid IoU loss": extra_info['iou_loss'],
                        })

                # save only when best loss
                mean_val_loss = valid_loss / valid_num_batches
                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_loss_epoch = epoch+1
                    ckpt_fpath = osp.join(model_dir, f"best_epoch_{best_val_loss_epoch}.pth")
                    torch.save(model.state_dict(), ckpt_fpath)
                    counter = 0
                else:
                    counter += 1
                    print(f"Not Val Update.. Counter : {counter}")
                    
            valid_end = time.time() - valid_start
            print("Valid Mean loss: {:.4f} || Elapsed time: {}".format(
                mean_val_loss,
                timedelta(seconds=valid_end)))
            
            print("Best Validation Loss: {:.4f} at Epoch {}".format(
                best_val_loss,
                best_val_loss_epoch))
            
            # early stopping
            if epoch + 1 == 120 and counter > patience:
                ckpt_fpath = osp.join(model_dir, f"epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_fpath)
                print("Early Stopping!")
                break
        else:
            if (epoch + 1) >= 100 and (epoch + 1) % save_interval == 0:
                ckpt_fpath = osp.join(model_dir, f"epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_fpath)
        
        scheduler.step()


def main(args):
    do_training(args, **args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    print(args)  # CLI에 configs 표시
    main(args)
