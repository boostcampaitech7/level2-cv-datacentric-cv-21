import json
import os
import random

def split_dataset(file_path, train_ratio=0.8):
    # merged_data.json 파일 불러오기
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    images = list(data["images"].items())  # 이미지 이름과 내용을 리스트로 변환
    random.shuffle(images)  # 데이터를 랜덤으로 섞기

    # train/val split
    train_size = int(len(images) * train_ratio)
    train_images = dict(images[:train_size])
    val_images = dict(images[train_size:])

    # train 데이터셋 만들기
    train_data = {"images": train_images}
    with open("train_data.json", 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)

    # val 데이터셋 만들기
    val_data = {"images": val_images}
    with open("val_data.json", 'w', encoding='utf-8') as val_file:
        json.dump(val_data, val_file, ensure_ascii=False, indent=4)

    print(f"Train data saved to train_data.json with {len(train_images)} items.")
    print(f"Validation data saved to val_data.json with {len(val_images)} items.")

# 실행 예시
split_dataset("/data/ephemeral/home/.GitHub/dataset/ufo/train.json", train_ratio=0.8)
