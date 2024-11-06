"""
Numba는 CPU 연산을 가속화하는 데 매우 유용하지만 
이미지 증강 라이브러리인 albumentations와 같은 고수준 Python 라이브러리 호출은 직접적으로 지원하지 않기 때문에 
Numba와 호환되지 않는 경우가 많습니다. 
이를 해결하기 위해, 가능한 증강 작업을 NumPy 배열에서 직접 수행하거나, Numba가 이해할 수 있는 형태로 재작성했습니다.
"""
import pickle
from tqdm import tqdm
import os
import os.path as osp
import sys
import numpy as np
from numba import njit, prange
import random

sys.path.append('/data/ephemeral/home/github/baseline')

# Function from "baseline" folder (수정이 불가한 내용들)
from baseline.east_dataset import EASTDataset

# Funtction from code (수정이 가능한 내용들)
from dataset import SceneTextDataset

# Custom Numba-compatible augmentation functions
@njit
def apply_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    factor = 1 + random.uniform(-brightness, brightness)
    image = np.clip(image * factor, 0, 255)
    factor = 1 + random.uniform(-contrast, contrast)
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = np.clip((image - mean) * factor + mean, 0, 255)
    # Skipping saturation and hue for simplicity in Numba
    return image.astype(np.uint8)

@njit
def apply_gaussian_blur(image, blur_limit=7):
    k = random.randint(1, blur_limit)
    # Apply simple box blur as an approximation
    for i in prange(image.shape[0] - k):
        for j in prange(image.shape[1] - k):
            image[i, j] = image[i:i + k, j:j + k].mean(axis=(0, 1))
    return image

@njit
def apply_gauss_noise(image, noise_level=10):
    noise = np.random.normal(0, noise_level, image.shape)
    image = np.clip(image + noise, 0, 255)
    return image.astype(np.uint8)

@njit
def apply_normalize(image, mean, std):
    image = (image / 255.0 - mean) / std
    return image

def main():
    data_dir = '/data/ephemeral/home/data'
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    image_size = [1024]
    crop_size = [1024]
    aug_select = []

    fold = 0

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
                custom_transform=None,
                color_jitter=False,
                normalize=False
            )
            train_dataset = EASTDataset(train_dataset)

            ds = len(train_dataset)
            for k in tqdm(range(ds)):
                data = train_dataset.__getitem__(k)
                image = data['image']

                # Apply Numba-compatible augmentations
                image = apply_color_jitter(image)
                image = apply_gaussian_blur(image)
                image = apply_gauss_noise(image)
                image = apply_normalize(image, mean=(0.683, 0.657, 0.624), std=(0.198, 0.205, 0.211))

                data['image'] = image
                with open(file=osp.join(data_dir, pkl_dir, f"{ds*i+ds*j+k}.pkl"), mode="wb") as f:
                    pickle.dump(data, f)

if __name__ == '__main__':
    main()
