import torch
import glob
import os
import clip
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()
image_files = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")
batch_size = 2000

common_txt_path = '/mnt/SSD1/Niantic/clip_feat/batch_paths.txt'
with open(common_txt_path, 'w') as common_file:
    common_file.write("")

for index in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
    batch_start = index
    batch_end = min(index + batch_size, len(image_files))
    batch_path = image_files[batch_start:batch_end]

    if os.path.exists(f'/mnt/SSD1/Niantic/clip_feat/batch_{index}.pt'):
        continue
    
    batch_images = [preprocess(Image.open(image_path)).unsqueeze(0) for image_path in batch_path]
    batch_tensor = torch.cat(batch_images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)

    image_features_cpu = image_features.cpu()

    torch.save(image_features_cpu, f'/mnt/SSD1/Niantic/clip_feat/batch_{index}.pt')
    
    with open(common_txt_path, 'a') as common_file:
        common_file.write(f"batch_{index}: {batch_path}\n")
