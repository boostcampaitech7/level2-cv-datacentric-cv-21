import os
import json
import time
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import timedelta

import torch
import cv2
from torch import cuda
from model import EAST

from detect import detect
from utils import increment_path

CHECKPOINT_EXTENSIONS = [".pth", ".ckpt"]

# 언어 감지 함수를 예시로 추가
def detect_language(image):
    # 간단한 예시로, 이 부분은 사전 학습된 언어 감지 모델을 호출하는 코드가 들어가야 합니다.
    # 여기서는 임의로 'chinese'를 반환한다고 가정합니다.
    # 실제 구현 시, 언어 감지 모델을 활용하여 예측된 언어를 반환하도록 수정 필요.
    return 'chinese'  # 예시: 실제로는 감지된 언어를 반환

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--data_dir", default="../data/medical")
    parser.add_argument("--model_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="predictions")
    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--input_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args

def load_language_models(model_dir, device):
    # 각 언어별로 모델을 로드하여 딕셔너리에 저장
    language_models = {}
    languages = ["chinese", "japanese", "thai", "vietnamese"]

    for language in languages:
        ckpt_path = os.path.join(model_dir, f"{language}_model.pth")
        if os.path.exists(ckpt_path):
            model = EAST(pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            language_models[language] = model
        else:
            print(f"Warning: Checkpoint for {language} not found at {ckpt_path}")

    return language_models

def do_inference(language_models, data_dir, input_size, batch_size, split="test"):
    image_fnames, by_sample_bboxes = [], []
    images, languages = [], []

    for image_fpath in tqdm(glob(os.path.join(data_dir, "img/{}/*".format(split)))):
        image_fnames.append(os.path.basename(image_fpath))
        image = cv2.imread(image_fpath)[:, :, ::-1]
        images.append(image)

        # 언어 감지 및 해당 언어 리스트에 추가
        detected_language = detect_language(image)
        languages.append(detected_language)

        # 배치 크기에 도달하면 추론 수행
        if len(images) == batch_size:
            for img, lang in zip(images, languages):
                model = language_models.get(lang)
                if model:
                    by_sample_bboxes.extend(detect(model, [img], input_size))
                else:
                    print(f"No model available for detected language: {lang}")
            images, languages = [], []

    if len(images):
        for img, lang in zip(images, languages):
            model = language_models.get(lang)
            if model:
                by_sample_bboxes.extend(detect(model, [img], input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result["images"][image_fname] = dict(words=words_info)

    return ufo_result

def main(args):
    # 언어별 모델 로드
    language_models = load_language_models(args.model_dir, args.device)

    model_name = args.model_dir.split("/")[-1]
    output_dir = increment_path(os.path.join(args.output_dir, model_name))

    print("Inference in progress")
    ufo_result = dict(images=dict())

    # 전체 테스트 데이터에 대해 언어별 모델을 사용한 추론 수행
    split_result = do_inference(language_models, args.data_dir, args.input_size, args.batch_size, split="test")
    ufo_result["images"].update(split_result["images"])

    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(ufo_result, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    tik = time.time()
    main(args)
    tok = time.time()
    print(f"Inference time: {timedelta(seconds=round(tok - tik, 2))}")