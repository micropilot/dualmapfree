import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    import faiss
except:
    print('Please install faiss via pip install faiss-gpu to perform retrieval.')

from fourm.models.fm import FM
from fourm.vq.vqvae import VQVAE, DiVAE
from fourm.models.generate import GenerationSampler, build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality, custom_text
from fourm.data.modality_transforms import RGBTransform
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import MetadataTransform
from fourm.utils.plotting_utils import decode_dict, visualize_bboxes, plot_text_in_square, decode_tok_depth, decode_tok_semseg
from torchvision import transforms

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

device = 'cuda'
torch.set_grad_enabled(False)

class FourmDinov2(nn.Module):
    def __init__(self, model_size='XL'):
        super(FourmDinov2, self).__init__()
        
        self.device = 'cpu'
        tokenizer_path = os.getcwd() + "/lib/models/MicKey/modules/ml_fourm/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
        self.text_tok = Tokenizer.from_file(tokenizer_path)
        self.toks = {
            'tok_dinov2': VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448').eval().to(self.device),
        }
        
        self.fm = FM.from_pretrained(f'EPFL-VILAB/4M-21_{model_size}').eval().to(self.device)
        self.sampler = GenerationSampler(self.fm)
        
        self.cond_domains = ['rgb@224']
        self.target_domains =  ['tok_dinov2@224'] 
        self.tokens_per_target = [256]
        autoregression_schemes = ['roar']
        decoding_steps = [1]
        token_decoding_schedules = ['linear']
        temps = [0.01]
        temp_schedules = ['constant'] 
        cfg_scales = [2.0]
        cfg_schedules = ['constant'] 
        cfg_grow_conditioning = True
        self.top_p, self.top_k = 0.8, 0.0
        
        self.schedule = build_chained_generation_schedules(
            cond_domains=self.cond_domains, target_domains=self.target_domains, tokens_per_target=self.tokens_per_target, 
            autoregression_schemes=autoregression_schemes, decoding_steps=decoding_steps, 
            token_decoding_schedules=token_decoding_schedules, temps=temps, temp_schedules=temp_schedules,
            cfg_scales=cfg_scales, cfg_schedules=cfg_schedules, cfg_grow_conditioning=cfg_grow_conditioning
        )

        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def preprocess_image(self, image_tensor):
        batch_size, channels, height, width = image_tensor.size()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        image_tensor = image_tensor.view(-1, channels, height, width)
        transformed_images = torch.stack([transform(image_tensor[i]) for i in range(image_tensor.size(0))])
        
        return transformed_images

    def forward(self, img, batch_size):
        self.device = 'cuda'
        img_process = self.preprocess_image(img)
        batched_sample = {
            'rgb@224': {
                'tensor': img_process.to(self.device),  # Batched tensor
                'input_mask': torch.zeros(batch_size, 196, dtype=torch.bool, device=self.device), 
                'target_mask': torch.ones(batch_size, 196, dtype=torch.bool, device=self.device),
            }
        }

        for target_mod, ntoks in zip(self.target_domains, self.tokens_per_target):
            batched_sample = init_empty_target_modality(batched_sample, MODALITY_INFO, target_mod, batch_size, ntoks, self.device)
        
        for cond_mod in self.cond_domains:
            batched_sample = init_full_input_modality(batched_sample, MODALITY_INFO, cond_mod, self.device, eos_id=self.text_tok.token_to_id("[EOS]"))
        
        out_dict = self.sampler.generate(
            batched_sample, self.schedule, text_tokenizer=self.text_tok, 
            verbose=True, seed=0,
            top_p=self.top_p, top_k=self.top_k,
        )

        dec_dict = decode_dict(
            out_dict, self.toks, self.text_tok, 
            image_size=224, patch_size=14,
            decoding_steps=50
        )
        return dec_dict
