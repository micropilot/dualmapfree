import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
import shutil

image_dir = '/mnt/SSD1/Niantic/data/'
output_dir = '/mnt/SSD1/Niantic/tar_files/train/rgb'
os.makedirs(output_dir, exist_ok=True)

image_files = glob.glob(os.path.join(image_dir, 'train/**/**/*.jpg'))
images_per_dir = 10000

def create_image_directory(chunk_info):
    images, dir_path = chunk_info
    os.makedirs(dir_path, exist_ok=True)  
    for image in images:
        base_name = os.path.basename(image)
        dir_name = os.path.dirname(image)  
        parts = dir_name.split(os.path.sep)
        prefix = parts[-2] 
        suffix = parts[-1] 
        new_filename = f"{prefix}_{suffix}_{base_name}"
        shutil.copy(image, os.path.join(dir_path, new_filename))
    return dir_path

def chunk_image_files(image_files, chunk_size):
    for i in range(0, len(image_files), chunk_size):
        if len(image_files) < i+ chunk_size:
            yield image_files[i:len(image_files)]
        else:
            yield image_files[i:i + chunk_size]

chunk_infos = [(chunk, os.path.join(output_dir, f'part_{i + 1}')) 
               for i, chunk in enumerate(chunk_image_files(image_files, images_per_dir))]

with Pool(12) as pool:
    for dir_path in tqdm(pool.imap_unordered(create_image_directory, chunk_infos), total=len(chunk_infos)):
        print(f"Created directory {dir_path} and copied images")
