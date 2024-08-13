from tokenizers import Tokenizer
import json
import glob
import os
from tqdm import tqdm
import torch
import numpy as np

tokenizer_path = "/home/hitesh/dualmapfree/4M_dataset_dev/ml-4m/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

captions_path = glob.glob("/mnt/SSD1/Niantic/captions/train/**/**/*")


output_dir = "/mnt/SSD1/Niantic/caption_token/train/"
os.makedirs(output_dir, exist_ok=True)
batch_size = 52

file_count = 0
part_index = 0
current_part_dir = os.path.join(output_dir, f"part_{part_index}")
os.makedirs(current_part_dir, exist_ok=True)
for i in tqdm(range(0, len(captions_path), batch_size)):
    batch_captions = captions_path[i:i+batch_size]
    tensors_captions = []
    token_fname = []
    for caption_path in batch_captions:
        with open(caption_path, 'r', encoding='utf-8') as file:
            captions =[]
            for line in file:
                caption = line.strip()  
                if caption:
                    captions.append(caption)

        tensors_captions.append(captions[0])
        base_name = os.path.basename(caption_path).split(".txt")[0]
        dir_name = os.path.dirname(caption_path)  
        parts = dir_name.split(os.path.sep)
        prefix = parts[-2] 
        suffix = parts[-1] 
        new_filename = f"{prefix}_{suffix}_{base_name}"
        token_fname.append(new_filename)
        
    encoding = tokenizer.encode_batch(tensors_captions)
    for i, file in enumerate(token_fname):
        tokens_np = torch.tensor(encoding[i].ids).cpu().numpy()
        if file_count >= 10000:
            part_index += 1
            current_part_dir = os.path.join(output_dir, f"part_{part_index}")
            os.makedirs(current_part_dir, exist_ok=True)
            file_count = 0
        
        np.save(os.path.join(current_part_dir, f"{file}.npy"), tokens_np)
        file_count += 1
