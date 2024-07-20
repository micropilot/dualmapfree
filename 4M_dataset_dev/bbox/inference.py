import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
import yolo_world
import glob
import spacy
import os
from collections import Counter
from tqdm import tqdm

cfg = Config.fromfile(
        "/home/hitesh/dualmapfree/4M_dataset_dev/bbox/YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    )
cfg.work_dir = "."
cfg.load_from = "pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth" 

runner = Runner.from_cfg(cfg)
runner.call_hook("before_run")
runner.load_or_resume()
pipeline = cfg.test_dataloader.dataset.pipeline
runner.pipeline = Compose(pipeline)
runner.model.eval()

import PIL.Image
import cv2
import supervision as sv
import time
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
mask_annotator = sv.MaskAnnotator()



def process_captions(caption):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(caption)
    common_objects = []
    for token in doc:
        if token.pos_ == 'NOUN' and not token.is_stop:
            lemma = token.lemma_.lower()
            common_objects.append(lemma)

    non_object_nouns = {
        'image', 'couple', 'area', 'element', 'scene', 'point', 'feature', 'mix', 'writing', 'focus', 'spot',
        'part', 'section', 'portion', 'aspect', 'item', 'thing', 'place', 'location', 
        'background', 'foreground', 'middle', 'center', 'edge', 'side', 'view', 'perspective', 
        'context', 'content', 'appearance', 'condition', 'state', 'form', 'structure', 
        'composition', 'arrangement', 'layout', 'setting', 'environment', 'surrounding', 
        'atmosphere', 'ambience', 'mood', 'feeling', 'impression', 'experience', 'instance',
        'example', 'case', 'type', 'kind', 'sort', 'category', 'class', 'group', 'range', 
        'variety', 'category', 'concept', 'notion', 'idea', 'perception', 'understanding','picture','interest',
        'sidewalk','passersby','placement'
    }

    filtered_objects = [obj for obj in common_objects if obj not in non_object_nouns]
    unique_objects = list(set(filtered_objects))

    return unique_objects

def run_image(
        runner,
        class_names,
        input_image,
        max_num_boxes=10,
        score_thr=0.10,
        nms_thr=0.5,
):
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))
    
    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    
    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    # nms
    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    text_path = os.path.dirname(input_image.replace("data","bbox").replace(".jpg",".txt"))
    os.makedirs(text_path, exist_ok=True)
    pred_instances = pred_instances.cpu().numpy()
    labels = pred_instances.labels
    scores = pred_instances.scores
    bboxes = pred_instances.bboxes
    text_file_path = input_image.replace("data","bbox").replace(".jpg",".txt")
    with open(text_file_path, "w") as file:
        file.write(f"Objects: {texts}\n\n")
        file.write("Labels, Scores, Bounding Boxes\n")
        for label, score, bbox in zip(labels, scores, bboxes):
            bbox_str = ", ".join(map(str, bbox))
            file.write(f"{label}, {score:.2f}, [{bbox_str}]\n")
    
image_paths = glob.glob("/mnt/SSD1/Niantic/data/train/**/**/*.jpg")

for image in tqdm(image_paths, desc="Processing images"):
    output_file_path = image.replace("data","bbox").replace(".jpg",".txt")
    if os.path.exists(output_file_path):
        continue                              
    with open(image.replace('data','captions').replace('.jpg','.txt'), 'r') as file:
        content = file.read()
    
    items = process_captions(content)
    objects = ", ".join(items)
    run_image(runner, objects, image)
