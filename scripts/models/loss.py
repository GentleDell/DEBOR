#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:05:16 2020

@author: zhantao
"""
from os.path import abspath
import sys
if __name__ == '__main__':
    if abspath('../') not in sys.path:
        sys.path.append(abspath('../'))
    
import torch
import torch.nn as nn
import open3d as o3d
from pytorch3d.structures import Meshes

from third_party.pytorch_msssim import MS_SSIM
from .geometric_layers import axisAngle_to_rotationMatrix

class VIBELoss(nn.Module):
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            e_disp_loss_weight = 1,
            e_tex_loss_weight=1,
            d_motion_loss_weight=1.,
            device='cuda',
    ):
        super(VIBELoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.e_disp_loss_weight = e_disp_loss_weight
        self.e_tex_loss_weight = e_tex_loss_weight
        self.d_motion_loss_weight = d_motion_loss_weight

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_displace  = nn.L1Loss().to(self.device)    # MSELoss would smoothen details
        self.criterion_regr = nn.L1Loss().to(self.device)
        self.criterion_tex  = nn.L1Loss().to(self.device)
        
        print('\n====Weights of loss:====')
        for key, val in vars(self).items():
            if 'e_' in key and 'loss' in key:
                print('%-25s:'%(key), val) 

    def forward(
            self,
            generator_outputs,
            data_2d,
            data_3d,
            data_body_mosh=None,
            data_motion_mosh=None,
            body_discriminator=None,
            motion_discriminator=None,
    ):

        real_2d = data_2d                   # 24 joints on image plane
        real_3d = data_3d['target_3d']      # 24 joints in camera coordinate
        real_bv = data_3d['target_bvt']     # 6890 vertices in camera coordinate 
        real_dp = data_3d['target_dp']      # 6890 displacement in ccd
        real_uv = data_3d['target_uv']
        data_3d_theta = data_3d['theta']

        preds = generator_outputs[0]
        pred_j3d = preds['kp_3d']
        pred_j2d = preds['kp_2d']
        pred_bvt = preds['verts']
        pred_dsp = preds['verts_disp']
        pred_tex = preds['tex_image']
        pred_theta = preds['theta']

        # <======== Generator Loss
        loss_kp_2d =  self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1., gt_weight=1.) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d) * self.e_3d_loss_weight
        loss_bvt_3d = self.keypoint_3d_loss(pred_bvt, real_bv) * self.e_3d_loss_weight
        
        loss_dsp_3d = self.displacement_loss(pred_dsp, real_dp) * self.e_disp_loss_weight
        loss_tex_2d = self.tex_loss(pred_tex, real_uv) * self.e_tex_loss_weight
        
        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
            'loss_bvt_3d': loss_bvt_3d,
            'loss_dsp_3d': loss_dsp_3d,
            'loss_tex': loss_tex_2d
        }
        
        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]
        
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

        gen_loss = torch.stack(list(loss_dict.values())).sum()

        return gen_loss, loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        # conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        # conf[:, :25] *= openpose_weight
        # conf[:, 25:] *= gt_weight
        conf = torch.ones_like(gt_keypoints_2d)
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        """
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()


    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        # smpl losses drops the global rotation
        pred_rotmat_valid = axisAngle_to_rotationMatrix(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)[:, 1:]
        gt_rotmat_valid = axisAngle_to_rotationMatrix(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)[:, 1:]
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
    
    def displacement_loss(self, pred_disp, gt_disp):
        """
        Compute 3D displacement loss
        """
        return self.criterion_displace(pred_disp, gt_disp)
    
    def tex_loss(self, pred_tex, gt_tex):
        """
        Compute 2D texture map loss
        """
        return self.criterion_tex(pred_tex, gt_tex)
    