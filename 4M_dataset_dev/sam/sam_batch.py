from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import cv2
import torch
import glob
import re
import os
from tqdm import tqdm
from multiprocessing import Pool, get_context

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()

def run_length_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def save_masks_to_file(args):
    masks, label_list, labels, iou_score, filename = args
    with open(filename, 'w') as f:
        f.write(f"{label_list}\n")
        for i, label in enumerate(labels):
            encoded_mask = run_length_encode(masks[i, 0])
            f.write(f"Label {label}: {encoded_mask}\n")
        f.write(f"iou_predictions: {iou_score}\n")

def process_image(image_path, resize_transform, device):
    img = cv2.imread(image_path)
    with open(image_path.replace('.jpg', '.txt').replace('data', 'bbox'), 'r') as file:
        data = file.read()

    lines = data.strip().split('\n')[3:]
    label_list = data.strip().split('\n')[0]
    boxes = []

    labels_name = []
    for line in lines:
        labels_name.append(line.split(",")[0])
        bbox_pattern = re.compile(r'\[([^\]]+)\]')
        bbox_matches = bbox_pattern.findall(line)
        bbox = [float(coord) for bbox in bbox_matches for coord in bbox.split(',')]
        boxes.append(bbox)

    image_boxes = torch.tensor(boxes, device=device)
    prepared_image = prepare_image(img, resize_transform, device)
    resized_boxes = resize_transform.apply_boxes_torch(image_boxes, img.shape[:2])
    return {
        'image': prepared_image,
        'boxes': resized_boxes,
        'original_size': img.shape[:2]
    }, label_list, labels_name

def process_batch(images, resize_transform, device):
    with get_context('spawn').Pool() as pool:
        results = pool.starmap(process_image, [(image_path, resize_transform, device) for image_path in images])
    
    batched_input = [result[0] for result in results]
    batched_labels = [result[1] for result in results]
    labels_name = [result[2] for result in results]
    
    return batched_input, batched_labels, labels_name

def save_batch_results(batched_output, batched_labels, labels_name, image_paths):
    args_list = []
    for i, image_path in enumerate(image_paths):
        masks = batched_output[i]['masks']
        iou_score = batched_output[i]['iou_predictions'].view(-1).tolist()
        labels = labels_name[i]
        txt_filename = image_path.replace("data", "sam").replace('.jpg', '.txt')
        if not os.path.exists(txt_filename):
            text_path = os.path.dirname(txt_filename)
            os.makedirs(text_path, exist_ok=True)
            args_list.append((masks.cpu(), batched_labels[i], labels, iou_score, txt_filename))
    
    with get_context('spawn').Pool() as pool:
        pool.map(save_masks_to_file, args_list)

def main():
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    images = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")
    batch_size = 4

    for batch_start in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch_images = images[batch_start:batch_start + batch_size]
        first_save_path = batch_images[0].replace("data", "sam").replace('.jpg', '.txt')
        last_save_path = batch_images[-1].replace("data", "sam").replace('.jpg', '.txt')
        if os.path.exists(first_save_path) and os.path.exists(last_save_path):
            continue

        batched_input, batched_labels, labels_name = process_batch(batch_images, resize_transform, device)
        
        batched_output = sam(batched_input, multimask_output=False)

        save_batch_results(batched_output, batched_labels, labels_name, batch_images)

    print("Processing completed.")

if __name__ == "__main__":
    main()
