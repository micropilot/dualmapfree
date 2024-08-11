from PIL import Image
import glob
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fourm.vq.vqvae import VQVAE, DiVAE
from torchvision import transforms
from fourm.utils import denormalize, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms import Normalize
from pathlib import Path
import torch.nn as nn
import re

class ChannelReducer(nn.Module):
    def __init__(self):
        super(ChannelReducer, self).__init__()
        self.channel_expander = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.spatial_reducer = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=9, stride=2, padding=1)
    
    def forward(self, x):
        x = self.channel_expander(x)  
        x = self.spatial_reducer(x)   
        return x
channel_reducer = ChannelReducer().cuda()


transform = transforms.ToTensor()
resize = transforms.Resize((224, 224))

images =  glob.glob("/mnt/SSD1/Niantic/feat_im/train/**/**/*.png")
tok = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448').cuda()
normalize = Normalize(mean=IMAGENET_INCEPTION_MEAN[0], std=IMAGENET_INCEPTION_STD[0])
output_dir = "/mnt/SSD1/Niantic/dinov2_tok/train/"
os.makedirs(output_dir, exist_ok=True)
batch_size = 48

file_count = 0
part_index = 0
current_part_dir = os.path.join(output_dir, f"part_{part_index}")
os.makedirs(current_part_dir, exist_ok=True)

def parse_min_max(txt_file):
    min_vals = []
    max_vals = []
    
    with open(txt_file, 'r') as file:
        for line in file:
            min_match = re.search(r'Min:\s*(-?\d+\.\d+)', line)
            max_match = re.search(r'Max:\s*(-?\d+\.\d+)', line)
            
            if min_match and max_match:
                min_vals.append(float(min_match.group(1)))
                max_vals.append(float(max_match.group(1)))
    
    return np.array(min_vals), np.array(max_vals)

def reconstruct_tensor(img_path):
    img_path = Path(img_path)
    img_file = img_path.with_suffix('.png')
    txt_file = str(img_path).replace("feature.png", "metadata.txt")
    try:
        img = Image.open(img_file)
        normalized_slice = np.array(img, dtype=np.float32).reshape(1369,1024)
        min_vals, max_vals = parse_min_max(txt_file)

        tensor_slice = normalized_slice / 255.0 * (max_vals - min_vals) + min_vals
        return torch.tensor(tensor_slice)
    
    except Exception as e:
        print(f"Error processing {txt_file}: {e}")
        shape = normalized_slice.shape 
        return torch.zeros(shape, dtype=torch.float32)
    



for i in tqdm(range(0, len(images), batch_size)):
    batch_images = images[i:i+batch_size]
    tensors_b3hw = []
    token_fname = []
    for image_path in batch_images:
        rgb_b3hw = reconstruct_tensor(image_path).to(dtype=torch.float32).cuda()  
        tensor_b3hw = rgb_b3hw.view(1, 1024, 37, 37)
        tensors_b3hw.append(channel_reducer(tensor_b3hw)) 
        base_name = os.path.basename(image_path).split(".png")[0].replace("_feature","")
        dir_name = os.path.dirname(image_path)  
        parts = dir_name.split(os.path.sep)
        prefix = parts[-2] 
        suffix = parts[-1] 
        new_filename = f"{prefix}_{suffix}_{base_name}"
        token_fname.append(new_filename)
    
    stacked_tensors_b3hw = torch.cat(tensors_b3hw).to(dtype=torch.float32).cuda()
    _, _, tokens = tok.encode(normalize(stacked_tensors_b3hw).cuda())
    
    for i, file in enumerate(token_fname):
        tokens_np = tokens[i].reshape(1, 256).cpu().numpy()
        if file_count >= 10000:
            part_index += 1
            current_part_dir = os.path.join(output_dir, f"part_{part_index}")
            os.makedirs(current_part_dir, exist_ok=True)
            file_count = 0
        
        np.save(os.path.join(current_part_dir, f"{file}.npy"), tokens_np)
        file_count += 1
