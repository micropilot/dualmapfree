from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import pandas as pd
import cv2
import torch
import glob
import re
import os
import ast
from tqdm import tqdm
from multiprocessing import Pool, get_context
import argparse
import requests
import time

global start_time, csv_no
start_time = time.time()

def send_slack_alert(webhook_url, message, csv_no):
    msg = f'GPU_{csv_no:}' + message
    if webhook_url:
        try:
            payload = {'text': msg}
            requests.post(webhook_url, json=payload)
        except requests.RequestException as e:
            print(f"Failed to send Slack alert: {e}")

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
    try:
        with open(filename, 'w') as f:
            f.write(f"{label_list}\n")
            for i, label in enumerate(labels):
                encoded_mask = run_length_encode(masks[i, 0])
                f.write(f"Label {label}: {encoded_mask}\n")
            f.write(f"iou_predictions: {iou_score}\n")
    except IOError as e:
        print(f"Failed to save masks to file {filename}: {e}")
        send_slack_alert(webhook_url, f"Failed to save masks to file {filename}",csv_no)

def process_image(image_path, resize_transform, device):
    try:
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
        if not boxes:
            file_path = 'unique.txt'
            if not os.path.exists(file_path):
                with open(file_path, 'w') as file:
                    pass
            with open(file_path, 'a') as file:
                file.write(f"{image_path}\n")

            actual_list = ast.literal_eval(label_list.split("Objects: ")[-1])
            for i in range(len(actual_list)):
                labels_name.append(i)
                boxes.append([0,0,540,720])
        image_boxes = torch.tensor(boxes, device=device)
        prepared_image = prepare_image(img, resize_transform, device)
        resized_boxes = resize_transform.apply_boxes_torch(image_boxes, img.shape[:2])
        return {
            'image': prepared_image,
            'boxes': resized_boxes,
            'original_size': img.shape[:2]
        }, label_list, labels_name
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        send_slack_alert(webhook_url, f"Failed to process image {image_path}", csv_no)
        return None, None, None

def process_batch(images, resize_transform, device):
    try:
        with get_context('spawn').Pool() as pool:
            results = pool.starmap(process_image, [(image_path, resize_transform, device) for image_path in images])
        
        batched_input = [result[0] for result in results if result[0] is not None]
        batched_labels = [result[1] for result in results if result[1] is not None]
        labels_name = [result[2] for result in results if result[2] is not None]
        
        return batched_input, batched_labels, labels_name
    except Exception as e:
        print(f"Failed to process batch: {e}")
        send_slack_alert(webhook_url, f"Failed to process batch", csv_no)
        return [], [], []

def save_batch_results(batched_output, batched_labels, labels_name, image_paths):
    try:
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
    except Exception as e:
        print(f"Failed to save batch results: {e}")
        send_slack_alert(webhook_url, f"Failed to save batch results", csv_no)

def main(csv_file, webhook_url):
    global csv_no, start_time
    csv_no = csv_file
    
    try:
        sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

        df = pd.read_csv(f'csv/scenes_part_{csv_file}.csv')
        folders = df.iloc[:, 0].tolist()

        images = []
        for folder in folders:
            images.extend(glob.glob(os.path.join(folder, '**', '*.jpg')))
            
        batch_size = 4
        total_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)
        
        for batch_start in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
            batch_images = images[batch_start:batch_start + batch_size]
            first_save_path = batch_images[0].replace("data", "sam").replace('.jpg', '.txt')
            last_save_path = batch_images[-1].replace("data", "sam").replace('.jpg', '.txt')
            if os.path.exists(first_save_path) and os.path.exists(last_save_path):
                continue

            batched_input, batched_labels, labels_name = process_batch(batch_images, resize_transform, device)

            batched_output = sam(batched_input, multimask_output=False)

            save_batch_results(batched_output, batched_labels, labels_name, batch_images)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:  
                percentage_complete = (batch_start + batch_size) / len(images) * 100
                send_slack_alert(webhook_url, f"Processing progress: {percentage_complete:.2f}%", csv_no)
                start_time = time.time() 

        print("Processing completed.")
        send_slack_alert(webhook_url, "Processing completed successfully.", csv_no)
    except Exception as e:
        print(f"An error occurred: {e}")
        send_slack_alert(webhook_url, f"An error occurred", csv_no)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images using SAM.')
    parser.add_argument('csv_file', type=str)
    args = parser.parse_args()
    webhook_url = "https://hooks.slack.com/services/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" 
    main(args.csv_file, webhook_url)
