import numpy as np
from PIL import Image
import torch

def reconstruct_tensor(img_path_base):
    # Load the grayscale image
    img = Image.open(img_path_base + '.jpg')
    normalized_slice = np.array(img, dtype=np.float32)

    # Load the min and max values from the text file
    min_max_vals = np.loadtxt(img_path_base + '.txt')
    min_vals = min_max_vals[0, :]
    max_vals = min_max_vals[1, :]

    # Denormalize the grayscale image to get the original tensor slice
    tensor_slice = normalized_slice / 255.0 * (max_vals - min_vals) + min_vals

    return torch.tensor(tensor_slice)
  
te = reconstruct_tensor("/mnt/SSD1/Niantic/clip/train/s00000/seq0/frame_00000")
