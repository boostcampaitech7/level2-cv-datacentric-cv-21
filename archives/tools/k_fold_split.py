import json
import os
import os.path as osp
from sklearn.model_selection import KFold
import argparse

def main():
    parser = argparse.ArgumentParser(description="KFold split of image data")
    parser.add_argument('--seed', type=int, default=137, help="Random seed for KFold")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds for KFold")
    parser.add_argument('--root_dirs', nargs='+', default=[
        '/data/ephemeral/home/data/chinese_receipt',
        '/data/ephemeral/home/data/japanese_receipt',
        '/data/ephemeral/home/data/thai_receipt',
        '/data/ephemeral/home/data/vietnamese_receipt'
    ], help="Root directories of the datasets")

    args = parser.parse_args()

    seed = args.seed
    folds = args.folds
    root_dirs = args.root_dirs

    for root_dir in root_dirs:
        json_file = osp.join(root_dir, 'ufo/train.json')

        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 이미지 목록을 가져옴
        image_list = list(data['images'].items())

        # KFold 인스턴스 생성
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        # 각 fold에 대한 훈련 및 검증 데이터셋 생성
        for fold, (train_idx, val_idx) in enumerate(kf.split(image_list)):
            train_images = dict([image_list[i] for i in train_idx])
            val_images = dict([image_list[i] for i in val_idx])

            train_data = {'images': train_images}
            val_data = {'images': val_images}

            # JSON 파일 생성
            fold_train_path = osp.join(root_dir, f'ufo/train{fold}.json')
            fold_val_path = osp.join(root_dir, f'ufo/valid{fold}.json')

            with open(fold_train_path, 'w', encoding='utf-8') as file:
                json.dump(train_data, file, indent=4, ensure_ascii=False)
            with open(fold_val_path, 'w', encoding='utf-8') as file:
                json.dump(val_data, file, indent=4, ensure_ascii=False)

            print(f"[{root_dir}] Fold {fold}:")
            print(f"  - Train images: {len(train_images)} saved to {fold_train_path}")
            print(f"  - Validation images: {len(val_images)} saved to {fold_val_path}")

if __name__ == "__main__":
    main()
