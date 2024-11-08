import os
import os.path as osp
import random
import numpy as np
import torch

from baseline.detect import detect
from tqdm import tqdm
import cv2
import json

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_gt_bboxes(dict_from_json, valid_images) :

    gt_bboxes = dict()
    # ufo_file_root = osp.join(root_dir, json_file)

    # with open(ufo_file_root, 'r') as f:
    #     ufo_file = json.load(f)

    # ufo_file_images = ufo_file['images']
    ufo_file_images = dict_from_json['images']
    for valid_image in tqdm(valid_images) :
        gt_bboxes[valid_image] = []
        for idx in ufo_file_images[valid_image]['words'].keys() :
            gt_bboxes[valid_image].append(ufo_file_images[valid_image]['words'][idx]['points'])

    return gt_bboxes

def get_pred_bboxes(model, data_dir, valid_images, input_size, batch_size, split='train') :

    image_fnames, by_sample_bboxes = [], []

    images = []
    for valid_image in tqdm(valid_images) :
        image_fpath = osp.join(infer_dir(data_dir, split, valid_image), valid_image)
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    pred_bboxes = dict()
    for idx in range(len(image_fnames)) :
        image_fname = image_fnames[idx]
        sample_bboxes = by_sample_bboxes[idx]
        pred_bboxes[image_fname] = sample_bboxes

    return pred_bboxes


def infer_dir(data_dir, split, fname):
        if "data_synth" in data_dir:
            if data_dir == "/data/ephemeral/home/data_synth/chinese_receipt":
                lang = 'chinese'
            elif data_dir == "/data/ephemeral/home/data_synth/japanese_receipt":
                lang = 'japanese'
            elif data_dir == "/data/ephemeral/home/data_synth/thai_receipt":
                lang = 'thai'
            elif data_dir == "/data/ephemeral/home/data_synth/vietnamese_receipt":
                lang = 'vietnamese'
            else:
                raise ValueError
        else:
            lang_indicator = fname.split('.')[1]
            if lang_indicator == 'zh':
                lang = 'chinese'
            elif lang_indicator == 'ja':
                lang = 'japanese'
            elif lang_indicator == 'th':
                lang = 'thai'
            elif lang_indicator == 'vi':
                lang = 'vietnamese'
            else:
                raise ValueError
        return osp.join(data_dir, 'img', split)