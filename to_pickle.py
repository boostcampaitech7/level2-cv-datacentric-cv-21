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
    data_dirs = [
        '/data/ephemeral/home/data/chinese_receipt',
        '/data/ephemeral/home/data/japanese_receipt',
        '/data/ephemeral/home/data/thai_receipt',
        '/data/ephemeral/home/data/vietnamese_receipt'
    ]
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    

    custom_augmentation_dict = {
        'CJ': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
        'GB': A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        'B': A.Blur(blur_limit=(7, 7), p=0.5), # 파라미터 정리 default값 (blur_limit=7(숫자가 클수록 이미지가 더 많이 흐려짐), p=0.5)
        'GN': A.GaussNoise(var_limit=(500, 5000), p=1.0), # 파라미터 정리 default값 (var_limit=(10, 50), mean=0, per_channel=True(각 채널(R, G, B)에 서로 다른 노이즈 추가), p=0.5)
        'HSV': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        'RBC': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    }

    image_size = [1024]
    crop_size = [1024]
    aug_select = ['CJ']

    fold = 0
    custom_augmentation = [custom_augmentation_dict[s] for s in aug_select]

    for data_dir in data_dirs:  # 각 데이터 디렉토리에 대해 반복
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
                        per_lang=True,
                        image_size=i_size,
                        crop_size=c_size,
                        ignore_tags=ignore_tags,
                        custom_transform=A.Compose(custom_augmentation),
                        color_jitter=False,
                    )
                train_dataset = EASTDataset(train_dataset)

                ds = len(train_dataset)
                for k in tqdm(range(ds)):
                    data = train_dataset.__getitem__(k)
                    with open(file=osp.join(data_dir, pkl_dir, f"{ds*i+ds*j+k}.pkl"), mode="wb") as f:
                        pickle.dump(data, f)
            

if __name__ == '__main__':
    main()
