from PIL import Image
import glob
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fourm.vq.vqvae import DiVAE
from torchvision import transforms
from fourm.utils import denormalize, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms import Normalize

transform = transforms.ToTensor()
resize = transforms.Resize((224, 224))

images = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")
tok = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_rgb_16k_224-448').cuda()
normalize = Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
output_dir = "/mnt/SSD1/Niantic/rgb_token/train/"
os.makedirs(output_dir, exist_ok=True)
batch_size = 52

file_count = 0
part_index = 0
current_part_dir = os.path.join(output_dir, f"part_{part_index}")
os.makedirs(current_part_dir, exist_ok=True)

for i in tqdm(range(0, len(images), batch_size)):
    batch_images = images[i:i+batch_size]
    tensors_b3hw = []
    token_fname = []
    for image_path in batch_images:
        image = Image.open(image_path).convert('RGB')
        rgb_b3hw = transform(resize(image)).unsqueeze(0)
        tensors_b3hw.append(rgb_b3hw)
        base_name = os.path.basename(image_path).split(".jpg")[0]
        dir_name = os.path.dirname(image_path)  
        parts = dir_name.split(os.path.sep)
        prefix = parts[-2] 
        suffix = parts[-1] 
        new_filename = f"{prefix}_{suffix}_{base_name}"
        token_fname.append(new_filename)
    
    stacked_tensors_b3hw = torch.cat(tensors_b3hw, dim=0)
    _, _, tokens = tok.encode(normalize(stacked_tensors_b3hw).cuda())
    
    for i, file in enumerate(token_fname):
        tokens_np = tokens[i].reshape(1, 196).cpu().numpy()
        if file_count >= 10000:
            part_index += 1
            current_part_dir = os.path.join(output_dir, f"part_{part_index}")
            os.makedirs(current_part_dir, exist_ok=True)
            file_count = 0
        
        np.save(os.path.join(current_part_dir, f"{file}.npy"), tokens_np)
        file_count += 1
