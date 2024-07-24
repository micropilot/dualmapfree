import glob
import torch
import ast
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image

def process_slice(args):
    tensor_slice, output_path_base = args

    jpg_path = output_path_base + '.jpg'
    txt_path = output_path_base + '.txt'

    if os.path.exists(jpg_path) and os.path.exists(txt_path):
        return

    min_vals = tensor_slice.min(axis=0)
    max_vals = tensor_slice.max(axis=0)

    normalized_slice = ((tensor_slice - min_vals) / (max_vals - min_vals) * 255).astype(np.uint8)

    img = Image.fromarray(normalized_slice, mode='L')
    img.save(jpg_path)

    np.savetxt(txt_path, np.vstack((min_vals, max_vals)))


def process_batch(batch_tensor, batch_paths):
    tensor = torch.load(batch_tensor)
    path_string = batch_paths.split(":")[-1].strip()
    file_paths_list = ast.literal_eval(path_string)

    tasks = []
    for i in range(tensor.shape[0]):
        output_path = file_paths_list[i].replace('data', 'clip').replace('.jpg', '')
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        tasks.append((tensor[i].numpy(), output_path))

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_slice, tasks), total=len(tasks)))

batch_tensor_paths = glob.glob("/mnt/SSD1/Niantic/clip_feat/*.pt")

with open("/mnt/SSD1/Niantic/clip_feat/batch_paths.txt", "r") as file:
    batch_paths = file.readlines()

for i, batch_tensor in enumerate(tqdm(batch_tensor_paths)):
    process_batch(batch_tensor, batch_paths[i])
