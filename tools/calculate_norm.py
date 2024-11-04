import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import argparse

def calculate_norm(img_list):
    mean_ = np.array([np.mean(x, axis=(0, 1)) for x in tqdm(img_list, ascii=True)])
    mean_r = mean_[..., 0].mean() / 255.0
    mean_g = mean_[..., 1].mean() / 255.0
    mean_b = mean_[..., 2].mean() / 255.0

    std_ = np.array([np.std(x, axis=(0, 1)) for x in tqdm(img_list, ascii=True)])
    std_r = std_[..., 0].mean() / 255.0
    std_g = std_[..., 1].mean() / 255.0
    std_b = std_[..., 2].mean() / 255.0

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def main():
    parser = argparse.ArgumentParser(description='Calculate mean and standard deviation of images in directories.')
    parser.add_argument('data_dirs', nargs='+', type=str, help='List of directories containing images')

    args = parser.parse_args()

    all_img_list = []
    for data_dir in args.data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: The directory {data_dir} does not exist, skipping.")
            continue

        img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
        for m in img_path:
            img = Image.open(m)
            # 이미지가 RGB 모드가 아닐 경우 변환
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
            all_img_list.append(img)

    if not all_img_list:
        print("No images found in the specified directories.")
        return

    mean, std = calculate_norm(all_img_list)
    print("Mean: ", mean)
    print("Standard Deviation: ", std)

if __name__ == "__main__":
    main()
