import argparse
import glob
import multiprocessing as mp
import os
import torch
from PIL import Image

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import matplotlib.pyplot as plt

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args)
    cfg.merge_from_list(['MODEL.WEIGHTS',
                          '/home/hitesh/dualmapfree/4M_dataset_dev/semantic_segmentation/Mask2Former/weights/model_final_54b88a.pkl'])
    cfg.freeze()
    return cfg

def transform_path(image_path):
    return image_path.replace("data", "semantic").replace(".jpg",".png")

cfg = setup_cfg("/home/hitesh/dualmapfree/4M_dataset_dev/semantic_segmentation/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml")
model = build_model(cfg)
DetectionCheckpointer(model).load('/home/hitesh/dualmapfree/4M_dataset_dev/semantic_segmentation/Mask2Former/weights/model_final_54b88a.pkl') 
model.train(False)
img_paths = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")

batch_size = 24
for i in tqdm(range(0, len(img_paths), batch_size)):
    inputs = []
    batch_paths = img_paths[i:i + batch_size]
    
    first_save_path = transform_path(batch_paths[0])
    last_save_path = transform_path(batch_paths[-1])
    
    if os.path.exists(first_save_path) and os.path.exists(last_save_path):
        continue
    
    for image_path in batch_paths:
        img = cv2.imread(image_path)
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        inputs.append({"image":img_tensor})
    
    with torch.no_grad():
        outputs = model(inputs)
    
    for i,image_path in enumerate(batch_paths):
        argmax_tensor = outputs[i]['sem_seg'].argmax(dim=0)
        argmax_array = argmax_tensor.cpu().numpy()
        argmax_array = (argmax_array - argmax_array.min()) / (argmax_array.max() - argmax_array.min()) * 255
        argmax_array = argmax_array.astype(np.uint8)
        argmax_image = Image.fromarray(argmax_array)
        output_path = transform_path(image_path)
        dir_path = os.path.dirname(output_path)
        os.makedirs(dir_path,exist_ok = True)
        argmax_image.save(output_path)
