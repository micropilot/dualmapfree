from imagebind import data
import torch
import glob
import os
from tqdm import tqdm
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

image_files = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")
batch_size = 500

common_txt_path = '/mnt/SSD1/Niantic/image_bind/batch_paths.txt'
if os.path.exists(common_txt_path):
    pass
else:
    with open(common_txt_path, 'w') as common_file:
        common_file.write("")
        
for i,index in enumerate(tqdm(range(0, len(image_files), batch_size), desc="Processing batches")):
    batch_start = index
    batch_end = min(index + batch_size, len(image_files))
    batch_path = image_files[batch_start:batch_end]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(batch_path, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)
        image_features = embeddings[ModalityType.VISION]

    image_features_cpu = image_features.cpu()
    torch.save(image_features_cpu, f'/mnt/SSD1/Niantic/image_bind/batch_{i}.pt')
    with open(common_txt_path, 'a') as common_file:
        common_file.write(f"batch_{i}: {batch_path}\n")
