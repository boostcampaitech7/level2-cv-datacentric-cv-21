import pickle
from tqdm import tqdm
import os
import os.path as osp

from east_dataset import EASTDataset
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
    
    # 이미지를 그대로 저장하기 때문에 custom_augmentation은 빈 리스트로 설정
    aug_select = []
    custom_augmentation = [A.NoOp()]  # 아무 작업도 하지 않도록 설정

    for data_dir in data_dirs:  # 각 데이터 디렉토리에 대해 반복
        pkl_dir = 'pickle/original/train/'  # 원본 형식 유지
        os.makedirs(osp.join(data_dir, pkl_dir), exist_ok=True)
        
        # 크기 조절과 크롭 없이 SceneTextDataset 초기화
        train_dataset = SceneTextDataset(
            root_dir=data_dir,
            split='train',
            json_name='train0.json',  # fold 0으로 가정
            ignore_tags=ignore_tags,
            custom_transform=A.Compose(custom_augmentation),
            color_jitter=False,
            normalize=False
        )
        
        train_dataset = EASTDataset(train_dataset)  # EASTDataset 초기화

        # 각 이미지를 pkl로 저장
        ds = len(train_dataset)
        for k in tqdm(range(ds)):
            data = train_dataset.__getitem__(k)
            with open(file=osp.join(data_dir, pkl_dir, f"{k}.pkl"), mode="wb") as f:
                pickle.dump(data, f)

if __name__ == '__main__':
    main()
