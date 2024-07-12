from PIL import Image
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

class ImageCaptioner:
    def __init__(self, model_id, image_dir, batch_size=10, webhook_url=None):
        self.model_id = model_id
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.webhook_url = webhook_url  # Add webhook_url parameter
        self.processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
        self.image_paths = glob.glob(f'{image_dir}/*.jpg')

    def send_slack_alert(self, message):
        """Send a message to Slack."""
        if self.webhook_url:
            payload = {'text': message}
            requests.post(self.webhook_url, json=payload)

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
        start_time = time.time()
        for image_batch_tuple in tqdm(image_batches, desc="Generating captions"):
            if time.time() - start_time > 900:  # 15 minutes in seconds
                self.send_slack_alert("Caption generation process is still running.")
                start_time = time.time()  # Reset the timer
            
            prompts = [
                "USER: <image>\nWrite a descriptive caption for the image, highlighting the visible elements and key features.\nASSISTANT:",
            ] * len(image_batch_tuple)
            images = [image_tuple[0] for image_tuple in image_batch_tuple]
            inputs = self.processor(prompts, images, return_tensors="pt", padding=True)
            output = self.model.generate(**inputs, max_new_tokens=1024)
            captions = self.processor.batch_decode(output, skip_special_tokens=True)
            self.save_captions(image_batch_tuple, captions)

    def save_captions(self, image_batch_tuple, captions):
        for i, caption in enumerate(captions):
            caption_path = image_batch_tuple[i][1].replace("data", "caption").rsplit(".", 1)[0] + ".txt"
            os.makedirs(os.path.dirname(caption_path), exist_ok=True)
            with open(caption_path, "w") as f:
                f.write(caption.split("\nASSISTANT: ")[-1])

    def run(self):
        try:
            image_batches = self.process_images()
            self.generate_captions(image_batches)
            self.send_slack_alert("Caption generation completed successfully.")
        except Exception as e:
            self.send_slack_alert(f"Caption generation process has stopped due to an error: {str(e)}")

            
if __name__ == "__main__":
    image_dir = '/mnt/SSD1/Niantic/data/test/s00525/seq1'
    model_id = "llava-hf/llava-1.5-7b-hf"
    batch_size = 15
    webhook_url = 'https://hooks.slack.com/services/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'  
    
    image_captioner = ImageCaptioner(model_id, image_dir, batch_size, webhook_url)
    image_captioner.run()
