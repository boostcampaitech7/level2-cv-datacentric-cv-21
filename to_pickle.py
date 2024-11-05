import pickle
from tqdm import tqdm

import os
import os.path as osp
import sys
sys.path.append('/data/ephemeral/home/github/baseline')

# Function from "baseline" folder (수정이 불가한 내용들)
from baseline.east_dataset import EASTDataset

# Funtction from code (수정이 가능한 내용들)
from dataset import SceneTextDataset

import albumentations as A

def main():
    data_dir = '/data/ephemeral/home/data'
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    

    custom_augmentation_dict = {
        'CJ': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
        'GB': A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        'B': A.Blur(blur_limit=7, p=0.5),
        'GN': A.GaussNoise(p=0.5),
        'HSV': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        'RBC': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        'N': A.Normalize(mean=(0.6831708235495132, 0.6570838514500981, 0.6245893701608299),
                        std=(0.19835448743425943, 0.20532970462804873, 0.21117810051894778), p=1.0)
    }

    image_size = [1024]
    crop_size = [1024]
    aug_select = []

    fold = 0
    custom_augmentation = [custom_augmentation_dict[s] for s in aug_select]

    # 단일 데이터셋 폴더에 대해서만 실행 후 pickle 폴더에 저장
    pkl_dir = f'pickle/{image_size}_cs{crop_size}_aug{aug_select}/train/'
    os.makedirs(osp.join(data_dir, pkl_dir), exist_ok=True)


    for i, i_size in enumerate(image_size):
        for j, c_size in enumerate(crop_size):
            if c_size > i_size:
                continue
            train_dataset = SceneTextDataset(
                    root_dir=data_dir,
                    split='train',
                    fold=fold,
                    per_lang=False,
                    image_size=i_size,
                    crop_size=c_size,
                    ignore_tags=ignore_tags,
                    custom_transform=A.Compose(custom_augmentation),
                    color_jitter=False,
                    normalize=False
                )
            train_dataset = EASTDataset(train_dataset)

            ds = len(train_dataset)
            for k in tqdm(range(ds)):
                data = train_dataset.__getitem__(k)
                with open(file=osp.join(data_dir, pkl_dir, f"{ds*i+ds*j+k}.pkl"), mode="wb") as f:
                    pickle.dump(data, f)
            

if __name__ == '__main__':
    main()
