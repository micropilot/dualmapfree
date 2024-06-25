import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from math import sqrt
from lib.models.MicKey.modules.DINO_modules.dinov2 import vit_large
from lib.models.MicKey.modules.att_layers.transformer import Transformer_self_att
from lib.models.MicKey.modules.utils.extractor_utils import desc_l2norm, BasicBlock

class MicKey_Extractor(nn.Module):
    def __init__(self, cfg, dinov2_weights=None):
        super().__init__()

        self.cfg = cfg 

        if self.cfg.DATASET.DATA_SOURCE == 'MapFree':
            # Define DINOv2 extractor
            self.dino_channels = cfg['MICKEY']['DINOV2']['CHANNEL_DIM']
            self.dino_downfactor = cfg['MICKEY']['DINOV2']['DOWN_FACTOR']
            if dinov2_weights is None:
                dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/"
                                                                    "dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                                                                    map_location="cpu")
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )

            self.dinov2_vitl14 = vit_large(**vit_kwargs)
            self.dinov2_vitl14.load_state_dict(dinov2_weights)
            self.dinov2_vitl14.requires_grad_(False)
            self.dinov2_vitl14.eval()

            # Define whether DINOv2 runs on float16 or float32
            if cfg['MICKEY']['DINOV2']['FLOAT16']:
                self.amp_dtype = torch.float16
                self.dinov2_vitl14.to(self.amp_dtype)
            else:
                self.amp_dtype = torch.float32

        # Define MicKey's heads
        self.depth_head = DeepResBlock_depth(cfg['MICKEY'])
        self.det_offset = DeepResBlock_offset(cfg['MICKEY'])
        self.dsc_head = DeepResBlock_desc(cfg['MICKEY'])
        self.det_head = DeepResBlock_det(cfg['MICKEY'])

    def forward(self, x, gt_depth_path=None):
        if self.cfg.DATASET.DATA_SOURCE == 'MapFree':
            B, C, H, W = x.shape
            x = x[:, :, :self.dino_downfactor * (H//self.dino_downfactor), :self.dino_downfactor * (W//self.dino_downfactor)]

            with torch.no_grad():
                dinov2_features = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
                dinov2_features = dinov2_features['x_norm_patchtokens'].permute(0, 2, 1).\
                    reshape(B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor).float()
        elif self.cfg.DATASET.DATA_SOURCE == 'RapidLoad':
            dinov2_features = x.view(x.shape[0], x.shape[1], int(sqrt(x.shape[2]))
                                     ,int(sqrt(x.shape[2])))
            
        if self.cfg.VARIANTS.FROZEN_DEPTH:
            scrs = self.det_head(dinov2_features)
            kpts = self.det_offset(dinov2_features)
            with torch.no_grad():
                gt_depth = self.ground_depth(gt_depth_path)
                depths = gt_depth.to('cuda')
            dscs = self.dsc_head(dinov2_features)
        else:
            scrs = self.det_head(dinov2_features)
            kpts = self.det_offset(dinov2_features)
            depths = self.depth_head(dinov2_features)
            dscs = self.dsc_head(dinov2_features)
            

        return kpts, depths, scrs, dscs

    def train(self, mode: bool = True):
            
        self.dsc_head.train(mode)
        if not self.cfg.VARIANTS.FROZEN_DEPTH:
            self.depth_head.train(mode)
        self.det_offset.train(mode)
        self.det_head.train(mode)

    def ground_depth(self, paths, patch_size=14):
        batch_outputs = []
        for path in paths:
            try:
                im = Image.open(path)
                im_gray = im.convert('L')
    
                if self.cfg.DATASET.DATA_SOURCE == 'MapFree':
                    resize_dim = im_gray.size  
                elif self.cfg.DATASET.DATA_SOURCE == 'RapidLoad':
                    resize_dim = (518, 518) 
    
                im_resized = im_gray.resize(resize_dim)
                im_array = np.array(im_resized)
    
                patches = []
                for i in range(0, im_array.shape[0], patch_size):
                    for j in range(0, im_array.shape[1], patch_size):
                        patch = im_array[i:i+patch_size, j:j+patch_size]
                        if patch.shape == (patch_size, patch_size):  
                            patches.append(patch)
    
                patches_array = np.array(patches)
                patch_means = patches_array.mean(axis=(1, 2))
                patch_means_reshaped = patch_means.reshape(resize_dim[0] // patch_size, 
                                                           resize_dim[1] // patch_size)
                batch_outputs.append(patch_means_reshaped)
    
            except Exception as e:
                print(f"An error occurred while processing {path}: {e}")
                if self.cfg.DATASET.DATA_SOURCE == 'MapFree':
                    resize_dim = (540,720)
                elif self.cfg.DATASET.DATA_SOURCE == 'RapidLoad':
                    resize_dim = (518, 518)
                zeros_shape = (resize_dim[0] // patch_size, resize_dim[1] // patch_size)
                batch_outputs.append(np.zeros(zeros_shape))

        gt_depth = np.stack(batch_outputs)
        tensor_gt_depth = torch.tensor(gt_depth)
        scaled_tensor_gt_depth = (tensor_gt_depth - 0) / (255 - 0)
        return scaled_tensor_gt_depth.to(torch.float32)

class DeepResBlock_det(torch.nn.Module):
    def __init__(self, config, padding_mode = 'zeros'):
        super().__init__()

        bn = config['KP_HEADS']['BN']
        in_channels = config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['KP_HEADS']['POS_ENCODING']

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.score = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.use_softmax = config['KP_HEADS']['USE_SOFTMAX']
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        # Allow more exploration with reinforce algorithm
        self.tmp_softmax = 100

        self.eps = nn.Parameter(torch.tensor(1e-16), requires_grad=False)
        self.offset_par1 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.offset_par2 = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.ones_kernel = nn.Parameter(torch.ones((1, 1, 3, 3)), requires_grad=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def remove_borders(self, score_map: torch.Tensor, borders: int):
        '''
        It removes the borders of the image to avoid detections on the corners
        '''
        shape = score_map.shape
        mask = torch.ones_like(score_map)

        mask[:, :, 0:borders, :] = 0
        mask[:, :, :, 0:borders] = 0
        mask[:, :, shape[2] - borders:shape[2], :] = 0
        mask[:, :, :, shape[3] - borders:shape[3]] = 0

        return mask * score_map

    def remove_brd_and_softmax(self, scores, borders):

        B = scores.shape[0]

        scores = scores - (scores.view(B, -1).mean(-1).view(B, 1, 1, 1) + self.eps).detach()
        exp_scores = torch.exp(scores / self.tmp_softmax)

        # remove borders
        exp_scores = self.remove_borders(exp_scores, borders=borders)

        # apply softmax
        sum_scores = exp_scores.sum(-1).sum(-1).view(B, 1, 1, 1)
        return exp_scores / (sum_scores + self.eps)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy scores
        scores = self.score(x)

        if self.use_softmax:
            scores = self.remove_brd_and_softmax(scores, 3)
        else:
            scores = self.remove_borders(self.sigmoid(scores), borders=3)

        return scores


class DeepResBlock_offset(torch.nn.Module):
    def __init__(self, config, padding_mode = 'zeros'):
        super().__init__()

        bn = config['KP_HEADS']['BN']
        in_channels = config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['KP_HEADS']['POS_ENCODING']
        self.sigmoid = torch.nn.Sigmoid()

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.xy_offset = nn.Conv2d(block_dims[3], 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy offsets
        xy_offsets = self.xy_offset(x)

        # Offset goes from 0 to 1
        xy_offsets = self.sigmoid(xy_offsets)

        return xy_offsets


class DeepResBlock_depth(torch.nn.Module):
    def __init__(self, config, padding_mode = 'zeros'):
        super().__init__()

        bn = config['KP_HEADS']['BN']
        in_channels = config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['KP_HEADS']['POS_ENCODING']

        self.use_depth_sigmoid = config['KP_HEADS']['USE_DEPTHSIGMOID']
        self.max_depth = config['KP_HEADS']['MAX_DEPTH']
        self.sigmoid = torch.nn.Sigmoid()

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.depth = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy depths
        # depths = torch.clip(self.depth(x), min=1e-3, max=500)
        if self.use_depth_sigmoid:
            depths = self.max_depth * self.sigmoid(self.depth(x))
        else:
            depths = self.depth(x)

        return depths


class DeepResBlock_desc(torch.nn.Module):
    def __init__(self, config, padding_mode = 'zeros'):
        super().__init__()

        bn = config['KP_HEADS']['BN']
        last_dim = config['DSC_HEAD']['LAST_DIM']
        in_channels = config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['DSC_HEAD']['POS_ENCODING']
        self.norm_desc = config['DSC_HEAD']['NORM_DSC']

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], last_dim, stride=1, bn=bn, padding_mode=padding_mode)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)


    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x, relu=False)

        if self.norm_desc:
            x = desc_l2norm(x)

        return x

