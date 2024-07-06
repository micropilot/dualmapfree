import torch
import pytorch_lightning as pl

from lib.models.MicKey.modules.loss.loss_class import MetricPoseLoss
from lib.models.MicKey.modules.compute_correspondences import ComputeCorrespondences
from lib.models.MicKey.modules.utils.training_utils import log_image_matches, debug_reward_matches_log, vis_inliers
from lib.models.MicKey.modules.utils.probabilisticProcrustes import e2eProbabilisticProcrustesSolver

from lib.utils.metrics import pose_error_torch, vcre_torch
from lib.benchmarks.utils import precision_recall
from PIL import Image
import numpy as np

class MicKeyTrainingModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Store MicKey's configuration
        self.cfg = cfg

        # Define MicKey architecture and matching module:
        self.compute_matches = ComputeCorrespondences(cfg)
        self.is_eval_model(False)

        # Loss function class
        self.loss_fn = MetricPoseLoss(cfg)

        # Metric solvers
        self.e2e_Procrustes = e2eProbabilisticProcrustesSolver(cfg)

        # Logger parameters
        self.counter_batch = 0
        self.log_store_ims = True
        self.log_max_ims = 5
        self.log_im_counter_train = 0
        self.log_im_counter_val = 0
        self.log_interval = cfg.TRAINING.LOG_INTERVAL

        # Define curriculum learning parameters:
        self.curriculum_learning = cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_CURRICULUM
        self.topK = cfg.LOSS_CLASS.CURRICULUM_LEARNING.TOPK_INIT
        self.topK_max = cfg.LOSS_CLASS.CURRICULUM_LEARNING.TOPK

        # Lightning configurations
        self.automatic_optimization = False # This property activates manual optimization.
        self.multi_gpu = True
        self.validation_step_outputs = []
        # torch.autograd.set_detect_anomaly(True)
        self.is_train = True

    def forward(self, data):
        self.compute_matches(data)

    def training_step(self, batch, batch_idx):

        self(batch)
        self.prepare_batch_for_loss(batch, batch_idx)
        
        if self.cfg.VARIANTS.GT_DEPTH:
            gt_depth1 = self.ground_depth(batch['gt_depth1_path'])
            gt_depth2 = self.ground_depth(batch['gt_depth2_path'])
            avg_loss, outputs, probs_grad, num_its = self.loss_fn(batch,self.is_train,gt_depth1,gt_depth2)
        else:
            avg_loss, outputs, probs_grad, num_its = self.loss_fn(batch,self.is_train)
        
        training_step_ok = self.backward_step(batch, outputs, probs_grad, avg_loss, num_its)
        
        if self.cfg.DATASET.DATA_SOURCE == "MapFree":
            self.tensorboard_log_step(batch, avg_loss, outputs, probs_grad, training_step_ok)

    def on_train_epoch_end(self):
        if self.curriculum_learning:
            self.topK = min(self.topK_max, self.topK + 5)
            self.loss_fn.topK = self.topK

    def validation_step(self, batch, batch_idx):
        self.is_train = False
        self.is_eval_model(True)
        self(batch)
        self.prepare_batch_for_loss(batch, batch_idx)
        
        # validation metrics
        avg_loss, outputs, probs_grad, num_its = self.loss_fn(batch,self.is_train)
        outputs['loss'] = avg_loss

        # Metric pose evaluation
        R_ours, t_m_ours, inliers_ours = self.e2e_Procrustes.estimate_pose(batch)
        outputs_metric_ours = pose_error_torch(R_ours, t_m_ours, batch['T_0to1'], reduce=None)
        outputs['metric_ours_t_err_ang'] = outputs_metric_ours['t_err_ang']
        outputs['metric_ours_t_err_euc'] = outputs_metric_ours['t_err_euc']
        outputs['metric_ours_R_err'] = outputs_metric_ours['R_err']
        outputs['metric_inliers'] = inliers_ours

        outputs_vcre_ours = vcre_torch(R_ours, t_m_ours, batch['T_0to1'], batch['Kori_color0'], reduce=None)
        outputs['metric_ours_vcre'] = outputs_vcre_ours['repr_err']

        self.validation_step_outputs.append(outputs)

        return outputs

    def backward_step(self, batch, outputs, probs_grad, avg_loss, num_its):
        opt = self.optimizers()

        # update model
        opt.zero_grad()

        if num_its == 0:
            print('No valid hypotheses were generated')
            return False

        # Generate gradients for learning keypoint offsets
        avg_loss.backward()

        invalid_probs = torch.isnan(probs_grad[0]).any()
        invalid_kps0 = (torch.isnan(outputs['kps0'].grad).any() or torch.isinf(outputs['kps0'].grad).any())
        invalid_kps1 = (torch.isnan(outputs['kps1'].grad).any() or torch.isinf(outputs['kps1'].grad).any())
        invalid_depth0 = (torch.isnan(outputs['depth0'].grad).any() or torch.isinf(outputs['depth0'].grad).any())
        invalid_depth1 = (torch.isnan(outputs['depth1'].grad).any() or torch.isinf(outputs['depth1'].grad).any())

        if invalid_probs:
            print('Found NaN/Inf in probs!')
            return False

        if invalid_depth0 or invalid_depth1:
            print('Found NaN/Inf in depth0/depth1 gradients!')
            return False

        if batch['kps0'].requires_grad:

            if invalid_kps0 or invalid_kps1:
                print('Found NaN/Inf in kps0/kps1 gradients!')
                return False

            torch.autograd.backward((torch.log(batch['final_scores'] + 1e-16),
                                 batch['kps0'], batch['kps1'], batch['depth_kp0'], batch['depth_kp1']),
                                (probs_grad[0], outputs['kps0'].grad, outputs['kps1'].grad,
                                 outputs['depth0'].grad, outputs['depth1'].grad))
        elif batch['depth0'].requires_grad:
            torch.autograd.backward((torch.log(batch['final_scores'] + 1e-16),
                                     batch['depth_kp0'], batch['depth_kp1']),
                                    (probs_grad[0], outputs['depth0'].grad, outputs['depth1'].grad))
        else:
            torch.autograd.backward((torch.log(batch['final_scores'] + 1e-16)),
                                    (probs_grad[0]))

        # add gradient clipping after backward to avoid gradient exploding
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)

        # check if the gradients of the training parameters contain nan values
        nans = sum([torch.isnan(param.grad).any() for param in list(self.parameters()) if param.grad is not None])
        if nans != 0:
            print("parameter gradients includes {} nan values".format(nans))
            return False

        opt.step()

        return True

    def tensorboard_log_step(self, batch, avg_loss, outputs, probs_grad, training_step_ok):

        self.log('train/loss', avg_loss.detach())
        self.log('train/loss_rot', outputs['avg_loss_rot'].detach())
        self.log('train/loss_trans', outputs['avg_loss_trans'].detach())

        if self.log_store_ims:
            if self.counter_batch % self.log_interval == 0:
                self.counter_batch = 0

                # If training with curriculum learning, not all image pairs have valid gradients
                # ensure selecting one image pair for logging that has reward information
                batch_id = torch.where(outputs['mask_topk'] == 1.)[0][0].item()

                # Metric pose evaluation
                R_ours, t_m_ours, inliers_ours, inliers_list_ours = self.e2e_Procrustes.estimate_pose(batch, return_inliers=True)
                outputs_metric_ours = pose_error_torch(R_ours, t_m_ours, batch['T_0to1'], reduce=None)
                self.log('train_metric_pose/ours_t_err_ang', outputs_metric_ours['t_err_ang'].mean().detach())
                self.log('train_metric_pose/ours_t_err_euc', outputs_metric_ours['t_err_euc'].mean().detach())
                self.log('train_metric_pose/ours_R_err', outputs_metric_ours['R_err'].mean().detach())

                outputs_vcre_ours = vcre_torch(R_ours, t_m_ours, batch['T_0to1'], batch['Kori_color0'], reduce=None)
                self.log('train_vcre/repr_err', outputs_vcre_ours['repr_err'].mean().detach())

                im_inliers = vis_inliers(inliers_list_ours, batch, batch_i=batch_id)

                im_matches, sc_map0, sc_map1, depth_map0, depth_map1 = log_image_matches(self.compute_matches.matcher,
                                                                                         batch, train_depth=True,
                                                                                         batch_i=batch_id)

                tensorboard = self.logger.experiment
                tensorboard.add_image('training_matching/best_inliers', im_inliers, global_step=self.log_im_counter_train)
                tensorboard.add_image('training_matching/best_matches_desc', im_matches, global_step=self.log_im_counter_train)
                tensorboard.add_image('training_scores/map0', sc_map0, global_step=self.log_im_counter_train)
                tensorboard.add_image('training_scores/map1', sc_map1, global_step=self.log_im_counter_train)
                tensorboard.add_image('training_depth/map0', depth_map0[0], global_step=self.log_im_counter_train)
                tensorboard.add_image('training_depth/map1', depth_map1[0], global_step=self.log_im_counter_train)
                if training_step_ok:
                    try:
                        im_rewards, rew_kp0, rew_kp1 = debug_reward_matches_log(batch, probs_grad, batch_i=batch_id)
                        tensorboard.add_image('training_rewards/pair0', im_rewards, global_step=self.log_im_counter_train)
                    except ValueError:
                        print('[WARNING]: Failed to log reward image. Selected image is not in topK image pairs. ')

                self.log_im_counter_train += 1

        torch.cuda.empty_cache()
        self.counter_batch += 1

    def prepare_batch_for_loss(self, batch, batch_idx):

        batch['batch_idx'] = batch_idx
        batch['final_scores'] = batch['scores'] * batch['kp_scores']

        return batch

    def on_validation_epoch_end(self):

        # aggregates metrics/losses from all validation steps
        aggregated = {}
        for key in self.validation_step_outputs[0].keys():
            aggregated[key] = torch.stack([x[key] for x in self.validation_step_outputs])

        # compute stats
        mean_R_loss = aggregated['avg_loss_rot'].mean()
        mean_t_loss = aggregated['avg_loss_trans'].mean()
        mean_loss = aggregated['loss'].mean()

        # Metric stats:
        metric_ours_t_err_ang = aggregated['metric_ours_t_err_ang'].mean()
        metric_ours_t_err_euc = aggregated['metric_ours_t_err_euc'].mean()
        metric_ours_R_err = aggregated['metric_ours_R_err'].mean()

        metric_ours_vcre = aggregated['metric_ours_vcre'].mean()

        # compute precision/AUC for pose error
        t_threshold = 0.25
        R_threshold = 5
        accepted_poses_ours = (aggregated['metric_ours_t_err_euc'].view(-1) < t_threshold) * \
                         (aggregated['metric_ours_R_err'].view(-1) < R_threshold)

        inliers = aggregated['metric_inliers'].view(-1).detach().cpu().numpy()

        prec_pose_ours = accepted_poses_ours.sum()/len(accepted_poses_ours)

        _, _, auc_pose = precision_recall(inliers=inliers, tp=accepted_poses_ours.detach().cpu().numpy(), failures=0)

        # compute precision/AUC for pose error
        t_threshold = 0.5
        R_threshold = 10
        accepted_poses_ours = (aggregated['metric_ours_t_err_euc'].view(-1) < t_threshold) * \
                              (aggregated['metric_ours_R_err'].view(-1) < R_threshold)

        inliers = aggregated['metric_inliers'].view(-1).detach().cpu().numpy()

        prec_pose_ours_10 = accepted_poses_ours.sum() / len(accepted_poses_ours)

        _, _, auc_pose_10 = precision_recall(inliers=inliers, tp=accepted_poses_ours.detach().cpu().numpy(), failures=0)


        # compute precision/AUC for reprojection errors
        px_threshold = 90
        accepted_vcre_ours = aggregated['metric_ours_vcre'].view(-1) < px_threshold

        prec_vcre_ours = accepted_vcre_ours.sum()/len(accepted_vcre_ours)

        _, _, auc_vcre = precision_recall(inliers=inliers, tp=accepted_vcre_ours.detach().cpu().numpy(), failures=0)

        # log stats
        self.log('val_loss/loss_R', mean_R_loss, sync_dist=self.multi_gpu)
        self.log('val_loss/loss_t', mean_t_loss, sync_dist=self.multi_gpu)
        self.log('val_loss/loss', mean_loss, sync_dist=self.multi_gpu)

        self.log('val_metric_pose/ours_t_err_ang', metric_ours_t_err_ang, sync_dist=self.multi_gpu)
        self.log('val_metric_pose/ours_t_err_euc', metric_ours_t_err_euc, sync_dist=self.multi_gpu)
        self.log('val_metric_pose/ours_R_err', metric_ours_R_err, sync_dist=self.multi_gpu)

        self.log('val_vcre/auc_vcre', auc_vcre, sync_dist=self.multi_gpu)
        self.log('val_vcre/prec_vcre_ours', prec_vcre_ours, sync_dist=self.multi_gpu)
        self.log('val_vcre/metric_ours_vcre', metric_ours_vcre, sync_dist=self.multi_gpu)

        self.log('val_AUC_pose/prec_pose_ours', prec_pose_ours, sync_dist=self.multi_gpu)
        self.log('val_AUC_pose/auc_pose', torch.tensor(auc_pose), sync_dist=self.multi_gpu)

        self.log('val_AUC_pose/prec_pose_ours_10', prec_pose_ours_10, sync_dist=self.multi_gpu)
        self.log('val_AUC_pose/auc_pose_10', torch.tensor(auc_pose_10), sync_dist=self.multi_gpu)

        self.validation_step_outputs.clear()  # free memory

        self.is_eval_model(False)

        return mean_loss

    def configure_optimizers(self):
        tcfg = self.cfg.TRAINING
        opt = torch.optim.Adam(self.parameters(), lr=tcfg.LR, eps=1e-6)
        if tcfg.LR_STEP_INTERVAL:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, tcfg.LR_STEP_INTERVAL, tcfg.LR_STEP_GAMMA)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        return opt

    def on_save_checkpoint(self, checkpoint):
        # As DINOv2 is pre-trained (and no finetuned, avoid saving its weights (it should help the memory).
        dinov2_keys = []
        for key in checkpoint['state_dict'].keys():
            if 'dinov2' in key:
                dinov2_keys.append(key)
        for key in dinov2_keys:
            del checkpoint['state_dict'][key]

    def on_load_checkpoint(self, checkpoint):

        # Recover DINOv2 features from pretrained weights.
        for param_tensor in self.compute_matches.state_dict():
            if 'dinov2'in param_tensor:
                checkpoint['state_dict']['compute_matches.'+param_tensor] = \
                    self.compute_matches.state_dict()[param_tensor]



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
        reshaped_gt_depth = np.reshape(gt_depth, (gt_depth.shape[0], 1,
                                                gt_depth.shape[1]*gt_depth.shape[2]))
        tensor_gt_depth = torch.tensor(reshaped_gt_depth)
        scaled_tensor_gt_depth = (tensor_gt_depth - 0) / (255 - 0)
        return scaled_tensor_gt_depth
        
    def is_eval_model(self, is_eval):
        if is_eval:
            self.compute_matches.extractor.depth_head.eval()
            self.compute_matches.extractor.det_offset.eval()
            self.compute_matches.extractor.dsc_head.eval()
            self.compute_matches.extractor.det_head.eval()
        else:
            self.compute_matches.extractor.depth_head.train()
            self.compute_matches.extractor.det_offset.train()
            self.compute_matches.extractor.dsc_head.train()
            self.compute_matches.extractor.det_head.train()
