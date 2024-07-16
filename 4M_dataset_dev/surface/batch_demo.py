import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb
import time

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image or directory of images")
parser.set_defaults(im_name='NONE')


parser.add_argument('--batch_size', dest='batch_size', type=int, help="batch size for processing images")
parser.set_defaults(batch_size=10)

args = parser.parse_args()

root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# get target task and model
if args.task == 'normal':
    image_size = 384
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        get_transform('rgb', image_size=None)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(512)])


def process_batch(img_paths):
    img_tensors = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)
        img_tensors.append(img_tensor)
    img_batch = torch.cat(img_tensors, dim=0)

    with torch.no_grad():
        outputs = model(img_batch).clamp(min=0, max=1)

        for img_path, output in zip(img_paths, outputs):
            output_file_path = os.path.dirname(img_path.replace('data','surface'))
            os.makedirs(output_file_path, exist_ok=True)
            output_file_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_file_path, f'{output_file_name}.jpg')
            trans_topil(output).save(save_path)


img_path = Path(args.img_path)
img_paths = []

if img_path.is_file():
    img_paths = [args.img_path]
elif img_path.is_dir():
    img_paths = glob.glob(f'{img_path}/**/**/*.jpg')
    print(len(img_paths))
else:
    print("Invalid file path!")
    sys.exit()

for i in tqdm(range(0, len(img_paths), args.batch_size)):
    batch_paths = img_paths[i:i + args.batch_size]

    first_output_file_path = os.path.dirname(batch_paths[0].replace('data', 'surface'))
    first_output_file_name = os.path.splitext(os.path.basename(batch_paths[0]))[0]
    first_save_path = os.path.join(first_output_file_path, f'{first_output_file_name}.jpg')
    
    last_output_file_path = os.path.dirname(batch_paths[-1].replace('data', 'surface'))
    last_output_file_name = os.path.splitext(os.path.basename(batch_paths[-1]))[0]
    last_save_path = os.path.join(last_output_file_path, f'{last_output_file_name}.jpg')
    
    if os.path.exists(first_save_path) and os.path.exists(last_save_path):
        continue

    process_batch(batch_paths)
