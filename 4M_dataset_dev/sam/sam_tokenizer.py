import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List, Tuple
from PIL import Image
import glob
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fourm.vq.vqvae import DiVAE, VQVAE
from torchvision import transforms
from fourm.utils import denormalize, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms import Normalize

transform = transforms.ToTensor()
resize = transforms.Resize((224, 224))

images = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")
tok = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_sam-instance_1k_64').cuda()
normalize = Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
output_dir = "/mnt/SSD1/Niantic/sam_token/train/"
os.makedirs(output_dir, exist_ok=True)
batch_size = 52

file_count = 0
part_index = 0
current_part_dir = os.path.join(output_dir, f"part_{part_index}")
os.makedirs(current_part_dir, exist_ok=True)

def run_length_decode(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1 
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape)

def extract_labels_and_iou(file_path: str) -> Tuple[List[str], List[float]]:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    label_names = eval(lines[0].strip().split('Objects: ')[1].strip())

    iou_predictions = eval(lines[-1].strip().split('iou_predictions: ')[1].strip())
    return label_names, iou_predictions

def overlay_masks_and_labels(image: np.ndarray, masks_info: str, label_names: List[str], iou_predictions: List[float]) -> np.ndarray:
    masks = {}
    lines = masks_info.splitlines()
    for line in lines:
        if line.startswith('Label'):
            label_id = int(line.split(':')[0].split()[1])
            encoded_mask = line.split(':')[1].strip()
            masks[label_id] = run_length_decode(encoded_mask, image.shape[:2])

    color_image = np.ones_like(image) * 255
    
    for label_id, mask in masks.items():
        color = np.random.randint(0, 255, size=3).tolist()  
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_image, contours, -1, color, 2)

        color_mask = np.zeros_like(image)
        color_mask[mask == 1] = color
        alpha = 0.3 
        color_image = cv2.addWeighted(color_image, 1 - alpha, color_mask, alpha, 0)
        
        label_name = label_names[label_id]
        text = f'{label_name}'

        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = 10, 30 + 20 * label_id
        
        cv2.putText(color_image, text, (cx - 100, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return color_image

def sam_preprocess(image_path: str):
    base_path = '/mnt/SSD1/Niantic/'
    sam_path = image_path.replace('/data/', '/sam/').replace('.jpg', '.txt')
    image = cv2.imread(image_path)
    label_names, iou_predictions = extract_labels_and_iou(sam_path)
    with open(sam_path, 'r') as file:
        lines = file.readlines()
    masks_info = ''.join(lines[1:-1])
    result_image = overlay_masks_and_labels(image, masks_info, label_names, iou_predictions)
    result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result

for i in tqdm(range(0, len(images), batch_size)):
    batch_images = images[i:i+batch_size]
    tensors_b3hw = []
    token_fname = []
    for image_path in batch_images:
        image = Image.fromarray(sam_preprocess(image_path)).convert('L')
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
    _, _, tokens = tok.encode(stacked_tensors_b3hw.cuda())
    
    for i, file in enumerate(token_fname):
        tokens_np = tokens[i].reshape(1, 196).cpu().numpy()
        if file_count >= 10000:
            part_index += 1
            current_part_dir = os.path.join(output_dir, f"part_{part_index}")
            os.makedirs(current_part_dir, exist_ok=True)
            file_count = 0
        
        np.save(os.path.join(current_part_dir, f"{file}.npy"), tokens_np)
        file_count += 1

