import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from tqdm import tqdm

# Function from "baseline" folder (수정이 불가한 내용들)
from baseline.detect import detect
from baseline.model import EAST


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # 실험마다 (증강에 따른) 모델 경로 및 결과 저장 경로를 수정해주셔야합니다! 25~28, 36
    parser.add_argument('--model_dir', nargs='+', type=str, default=[
        "/data/ephemeral/home/github/saved_models/chinese_receipt/aug[]",
        "/data/ephemeral/home/github/saved_models/japanese_receipt/aug[]",
        "/data/ephemeral/home/github/saved_models/thai_receipt/aug[]",
        "/data/ephemeral/home/github/saved_models/vietnamese_receipt/aug[]",
        ])
    parser.add_argument('--data_dirs', nargs='+', type=str, default=[
        "/data/ephemeral/home/data/chinese_receipt",
        "/data/ephemeral/home/data/japanese_receipt",
        "/data/ephemeral/home/data/thai_receipt",
        "/data/ephemeral/home/data/vietnamese_receipt",
        ])
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def draw_boxes_on_image(image, bboxes):
    """Draw bounding boxes on an image."""
    for bbox in bboxes:
        bbox = bbox.astype(int)
        cv2.polylines(image, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    output_image_dir = osp.join(data_dir, 'output_images')  # Output directory for images with bounding boxes
    os.makedirs(output_image_dir, exist_ok=True)

    images = []
    for image_fpath in tqdm(glob(osp.join(data_dir, 'img/{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))
        images.append(cv2.imread(image_fpath)[:, :, ::-1])

        if len(images) == batch_size:
            predictions = detect(model, images, input_size)
            by_sample_bboxes.extend(predictions)

            # Draw bounding boxes on images and save them
            for img, bboxes, fname in zip(images, predictions, image_fnames[-batch_size:]):
                img_with_boxes = draw_boxes_on_image(img.copy(), bboxes)
                cv2.imwrite(osp.join(output_image_dir, fname), img_with_boxes[:, :, ::-1])  # Save image with boxes
            
            images = []

    if len(images):
        predictions = detect(model, images, input_size)
        by_sample_bboxes.extend(predictions)
        for img, bboxes, fname in zip(images, predictions, image_fnames[-batch_size:]):
            img_with_boxes = draw_boxes_on_image(img.copy(), bboxes)
            cv2.imwrite(osp.join(output_image_dir, fname), img_with_boxes[:, :, ::-1])


    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # 최종 결과를 저장할 딕셔너리 초기화
    ufo_result = dict(images=dict())

    for model_dir, data_dir in zip(args.model_dir, args.data_dirs):
        # 모델과 체크포인트 초기화
        model = EAST(pretrained=False).to(args.device)
        best_checkpoint_fpath = osp.join(model_dir, 'best.pth')

        if os.path.isfile(best_checkpoint_fpath):
            print(f'{model_dir}의 best checkpoint 찾음')
            ckpt_fpath = best_checkpoint_fpath
        else:
            print(f'{model_dir}의 best checkpoint 찾지 못함, latest checkpoint로 설정')
            ckpt_fpath = osp.join(model_dir, 'latest.pth')

        # 각 모델과 데이터 디렉토리에서 추론 수행
        split_result = do_inference(model, ckpt_fpath, data_dir, args.input_size, args.batch_size, split='test')

        # 추론 결과를 최종 결과에 업데이트
        ufo_result['images'].update(split_result['images'])

    # csv를 저장할 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 최종 결과를 하나의 CSV 파일로 저장
    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)