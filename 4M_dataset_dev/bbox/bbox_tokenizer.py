from tokenizers import Tokenizer
import glob
import os
from tqdm import tqdm
import torch
import numpy as np

tokenizer_path = "/home/hitesh/dualmapfree/4M_dataset_dev/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

captions_path = glob.glob("/mnt/SSD1/Niantic/bbox/train/**/**/*.txt")
output_dir = "/mnt/SSD1/Niantic/bbox_token/train/"
os.makedirs(output_dir, exist_ok=True)
batch_size = 52

file_count = 0
part_index = 0
current_part_dir = os.path.join(output_dir, f"part_{part_index}")
os.makedirs(current_part_dir, exist_ok=True)

def convert_bbox_to_tokens(label, bbox, resolution=1000):
    xmin, ymin, xmax, ymax = bbox
    bbox_string = (
        f'xmin = {int((xmin * resolution) / 540)} '
        f'ymin = {int((ymin * resolution) / 720)} '
        f'xmax = {int((xmax * resolution) / 540)} '
        f'ymax = {int((ymax * resolution) / 720)} '
        f'{str(label).replace("[", "").replace("]", "")}'
    )
    return bbox_string

def process_bboxes(caption_path):
    with open(caption_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    labels = eval(lines[0].split("Objects: ")[-1])
    bboxes = []

    bbox_start_index = 2 
    for line in lines[bbox_start_index+1:]:
        parts = line.strip().split(', ',2)
        if len(parts) < 3:
            continue
        bbox_str = parts[2].strip()

        try:
            bbox = eval(bbox_str) 
        except:
            continue
        if len(bbox) == 4:
            bboxes.append(convert_bbox_to_tokens(labels[int(parts[0])],bbox))
    return bboxes, os.path.basename(caption_path).split(".txt")[0]

for i in tqdm(range(0, len(captions_path), batch_size)):
    batch_captions = captions_path[i:i+batch_size]
    all_bboxes = []
    token_fname = []
    
    for caption_path in batch_captions:
        bboxes, base_name = process_bboxes(caption_path)
        if bboxes:
            all_bboxes.append(str(bboxes))
        else:
            all_bboxes.append(str([]))
        base_name = os.path.basename(caption_path).split(".txt")[0]
        dir_name = os.path.dirname(caption_path)  
        parts = dir_name.split(os.path.sep)
        prefix = parts[-2] 
        suffix = parts[-1] 
        new_filename = f"{prefix}_{suffix}_{base_name}"
        token_fname.append(new_filename)
    
    encoding = tokenizer.encode_batch(all_bboxes)

    for i, file in enumerate(token_fname):
        tokens_np = torch.tensor(encoding[i].ids).cpu().numpy()

        if file_count >= 10000:
            part_index += 1
            current_part_dir = os.path.join(output_dir, f"part_{part_index}")
            os.makedirs(current_part_dir, exist_ok=True)
            file_count = 0
        
        np.save(os.path.join(current_part_dir, f"{file}.npy"), tokens_np)
        file_count += 1
