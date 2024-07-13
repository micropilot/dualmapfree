from PIL import Image, UnidentifiedImageError
import multiprocessing as mp
from tqdm import tqdm
import glob
import os
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
import warnings
warnings.filterwarnings("ignore")
import requests
import time
import argparse

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
        self.csv_file = f'csv/scenes_part_{gpu_no + 1}.csv'
        self.processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
        self.image_paths = []
        if os.path.exists(self.csv_file):
            with open(self.csv_file, 'r') as file:
                scene_paths = [line.strip() for line in file]
            for scene_path in scene_paths:
                self.image_paths.extend(glob.glob(os.path.join(scene_path, '**/*.jpg'), recursive=True))

    def send_slack_alert(self, message):
        """Send a message to Slack."""
        if self.webhook_url:
            try:
                payload = {'text': message}
                requests.post(self.webhook_url, json=payload)
            except requests.RequestException as e:
                print(f"Failed to send Slack alert: {e}")

    def process_images(self):
        all_images = []
        for image_batch in tqdm(chunks(self.image_paths, self.batch_size), total=len(self.image_paths) // self.batch_size):
            first_image_path = image_batch[0]
            caption_path = first_image_path.replace("data", "caption").rsplit(".", 1)[0] + ".txt"
            if os.path.exists(caption_path):
                continue 
                
            with mp.Pool(mp.cpu_count()) as pool:
                images = pool.map(open_image, image_batch)
            images = [img for img in images if img is not None]
            all_images.append(images)
        return all_images

    def generate_captions(self, image_batches):
        total_batches = len(image_batches)
        start_time = time.time()
        for batch_index, image_batch_tuple in enumerate(tqdm(image_batches, desc=f"Generating captions on GPU {self.gpu_no}")):
            try:
                if time.time() - start_time > 25:  
                    percentage = (batch_index + 1) / total_batches * 100
                    progress_message = f"GPU {self.gpu_no}: Processed batch {batch_index + 1} of {total_batches}: {percentage:.2f}%."
                    self.send_slack_alert(progress_message)
                    start_time = time.time()
                
                prompts = [
                    "USER: <image>\nWrite a descriptive caption for the image, highlighting the visible elements and key features.\nASSISTANT:",
                ] * len(image_batch_tuple)
                images = [image_tuple[0] for image_tuple in image_batch_tuple]
                inputs = self.processor(prompts, images, return_tensors="pt", padding=True)
                output = self.model.generate(**inputs, max_new_tokens=1024)
                captions = self.processor.batch_decode(output, skip_special_tokens=True)
                self.save_captions(image_batch_tuple, captions)
            except Exception as e:
                self.send_slack_alert(f"GPU {self.gpu_no}: Error during caption generation: {str(e)}")
                continue

    def save_captions(self, image_batch_tuple, captions):
        for i, caption in enumerate(captions):
            try:
                caption_path = image_batch_tuple[i][1].replace("data", "caption").rsplit(".", 1)[0] + ".txt"
                os.makedirs(os.path.dirname(caption_path), exist_ok=True)
                with open(caption_path, "w") as f:
                    f.write(caption.split("\nASSISTANT: ")[-1])
            except IOError as e:
                print(f"Error saving caption for {image_batch_tuple[i][1]}: {e}")

    def run(self):
        try:
            image_batches = self.process_images()
            self.send_slack_alert(f"GPU {self.gpu_no}: Caption generation started.")
            self.generate_captions(image_batches)
            self.send_slack_alert(f"GPU {self.gpu_no}: Caption generation completed successfully.")
        except Exception as e:
            self.send_slack_alert(f"GPU {self.gpu_no}: Caption generation process has stopped due to an error: {str(e)}")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning with Multiple GPUs")
    parser.add_argument("--gpu_no", type=int, required=True, help="GPU number to use")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch Size to use")
    args = parser.parse_args()

    model_id = "llava-hf/llava-1.5-7b-hf"
    batch_size = args.batch_size
    webhook_url = None
    
    image_captioner = ImageCaptioner(model_id, batch_size, webhook_url, args.gpu_no)
    image_captioner.run()
