import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image, UnidentifiedImageError
import multiprocessing as mp
from tqdm import tqdm
import glob
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
import warnings
warnings.filterwarnings("ignore")
import requests
import time
import argparse
import torch

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def open_image(path):
    try:
        return Image.open(path), path
    except (UnidentifiedImageError, OSError) as e:
        print(f"Error opening image {path}: {e}")
        return None

class ImageCaptioner:
    def __init__(self, model_id, batch_size=10, webhook_url=None, gpu_no=0):
        self.model_id = model_id
        self.batch_size = batch_size
        self.webhook_url = webhook_url
        self.gpu_no = gpu_no
        self.csv_file = f'csv/scenes_part_{gpu_no}.csv'
        self.processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
        self.image_paths = []
        if os.path.exists(self.csv_file):
            with open(self.csv_file, 'r') as file:
                scene_paths = [line.strip() for line in file]
            for scene_path in scene_paths:
                self.image_paths.extend(glob.glob(os.path.join(scene_path, '**/*.jpg'), recursive=True))
        else:
            print("csv doesn't exist")
        self.start_time = time.time()
            
    def send_slack_alert(self, message):
        """Send a message to Slack."""
        if self.webhook_url:
            try:
                payload = {'text': message}
                requests.post(self.webhook_url, json=payload)
            except requests.RequestException as e:
                print(f"Failed to send Slack alert: {e}")

    def process_images(self, image_batch):
        with mp.Pool(8) as pool:
            images = pool.map(open_image, image_batch)
        return [img for img in images if img is not None]

    def generate_captions(self, image_batches, total_batches,batch_index):
        for image_batch_tuple in image_batches:
            try:
                if time.time() - self.start_time > 900:  
                    percentage = (batch_index + 1) / total_batches * 100
                    progress_message = f"GPU {self.gpu_no}: Processed batch {batch_index + 1} of {total_batches}: {percentage:.2f}%."
                    self.send_slack_alert(progress_message)
                    self.start_time = time.time()
                
                prompts = [
                    "USER: <image>\nWrite a descriptive caption for the image, highlighting the visible elements and key features.\nASSISTANT:",
                ] * len(image_batch_tuple)
                images = [image_tuple[0] for image_tuple in image_batch_tuple]
                inputs = self.processor(prompts, images, return_tensors="pt", padding=True).to('cuda')
                output = self.model.generate(**inputs, max_new_tokens=1024)
                captions = self.processor.batch_decode(output, skip_special_tokens=True)
                self.save_captions(image_batch_tuple, captions)
            except Exception as e:
                print(f"GPU {self.gpu_no}: Error during caption generation: {str(e)}")
                self.send_slack_alert(f"GPU {self.gpu_no}: Error during caption generation.")
                continue

    def save_captions(self, image_batch_tuple, captions):
        for i, caption in enumerate(captions):
            try:
                caption_path = image_batch_tuple[i][1].replace("original", "caption").rsplit(".", 1)[0] + ".txt"
                os.makedirs(os.path.dirname(caption_path), exist_ok=True)
                with open(caption_path, "w") as f:
                    f.write(caption.split("\nASSISTANT: ")[-1])
            except IOError as e:
                print(f"Error saving caption for {image_batch_tuple[i][1]}: {e}")

    def run(self):
        try:
            image_batches = list(chunks(self.image_paths, self.batch_size))
            total_batches = len(image_batches)
            # self.send_slack_alert(f"GPU {self.gpu_no}: Caption generation started.")
            for batch_index, image_batch in enumerate(tqdm(image_batches, desc=f"Loading images and generating captions on GPU {self.gpu_no}")):
                image_batch_tuples = self.process_images(image_batch)
                first_image_path = image_batch_tuples[0][1]
                first_image_caption_path = first_image_path.replace("original", "caption").rsplit(".", 1)[0] + ".txt"
                last_image_path = image_batch_tuples[-1][1]
                last_image_caption_path = last_image_path.replace("original", "caption").rsplit(".", 1)[0] + ".txt"
                if os.path.exists(first_image_caption_path) and os.path.exists(last_image_caption_path):
                    continue
                self.generate_captions([image_batch_tuples], total_batches,batch_index)
            self.send_slack_alert(f"GPU {self.gpu_no}: Caption generation completed successfully.")
        except Exception as e:
            print(f"GPU {self.gpu_no}: Caption generation process has stopped due to an error: {str(e)}")
            self.send_slack_alert(f"GPU {self.gpu_no}: Caption generation process has stopped due to an error.")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning with Multiple GPUs")
    parser.add_argument("--gpu_no", type=int, required=True, help="GPU number to use")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch Size to use")
    args = parser.parse_args()
    model_id = "llava-hf/llava-1.5-7b-hf"
    batch_size = args.batch_size
    webhook_url = "https://hooks.slack.com/services/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"
    image_captioner = ImageCaptioner(model_id, batch_size, webhook_url, args.gpu_no)
    image_captioner.run()
