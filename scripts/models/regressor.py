#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:09:31 2020
"""

import torch 
import torch.nn as nn

from .smpl import SMPL
from .resnet import resnet50
from .geometric_layers import rot6d_to_axisAngle, rot6d_to_rotmat
from .camera import cameraPerspective as Cam
from .graph_cnn import GraphCNN
from .unet import UNet
from utils import renderer

class frameVIBE(nn.Module):
    def __init__(
            self,
            model_file,
            mesh,
            init_pose, 
            init_shape, 
            init_cam,
            GCN_channels,
            GCN_layers,
            smpl_verts_uvs,
            smpl_tri_ind
    ):
        super(frameVIBE, self).__init__()
        
        # image encoder to extract latent feature vector
        self.encoder = ImageEncoder()

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor_pose(model_file, init_pose, init_shape, init_cam)
        
        # cloth regressor        
        self.GCN_regressor = GraphCNN(mesh.adjmat, mesh.ref_vertices.t(), 2048, GCN_layers, GCN_channels)

        # texture regressor
        self.tex_regressor = UNet(input_shape = (224, 224, 3), output_dims = 3)   

        # <=========== rendering parameters
        # differentiable renderer
        self.renderer = renderer(batch_size = 1)
        
        # an image of uv coordinate as texture to unwarp image to uv map
        u = torch.arange(start=0, end=1, step=1/224)
        v = torch.arange(start=0, end=1, step=1/224)
        grid_v, grid_u = torch.meshgrid(u, v)
        uvImage = torch.ones([224,224,3])
        uvImage[:,:,0], uvImage[:,:,1] = grid_u, 1-grid_v    # x,y i.e. col, row
        self.register_buffer('uvImage', uvImage[None])
        
        # auxiliary parameters for rendering
        faces = mesh.faces.type(torch.LongTensor)
        R = torch.eye(3)[None]
        f = torch.ones([1,1])*1000
        C = torch.ones([1,2])*112
        self.camRes = 224
        self.register_buffer('faces', faces[None])
        self.register_buffer('smpl_verts_uvs', smpl_verts_uvs[None])
        self.register_buffer('smpl_tri_ind', smpl_tri_ind[None])
        self.register_buffer('camR', R)
        self.register_buffer('camf', f)
        self.register_buffer('camC', C)
    
    def forward(self, img):
        # input size NTF
        batch_size = img.shape[0]

        feature = self.encoder(img)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature)
        disp_output = self.GCN_regressor(feature)

        with torch.no_grad():
            uvimages  = self.render_uvpose(smpl_output[0]['verts'], 
                                           smpl_output[0]['theta'][:,:3])
            unwrapTex = self.unwarp(img, uvimages)
        tex_output  = self.tex_regressor(unwrapTex.permute(0,3,1,2))    # N*3*224*224
            
        for ind, s in enumerate(smpl_output):
            s['theta'] = s['theta'].reshape(batch_size, -1)
            s['verts'] = s['verts'].reshape(batch_size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)        
            s['verts_disp'] = disp_output.reshape(batch_size, -1, 3)
            s['tex_image'] = tex_output.reshape(batch_size, 224, 224, 3)
        return smpl_output
    
    def render_uvpose(self, verts, trans):
        
        images = []  
        device = verts.device
        
        predTrans = torch.stack(
            [trans[:, 1],
             trans[:, 2],
             2 * 1000. / (224. * trans[:, 0] + 1e-9)], dim=-1)
        
        for cnt in range(verts.shape[0]): 
            image = self.renderer(
                verts = verts[cnt][None],
                faces = self.faces,
                verts_uvs = self.smpl_verts_uvs,
                faces_uvs = self.smpl_tri_ind,
                tex_image = self.uvImage,
                R = self.camR,
                T = predTrans[cnt][None],
                f = self.camf,
                C = self.camC,
                imgres = self.camRes
            )
            
            images.append(image)
        
        return torch.cat(images, dim=0).to(device)

    def unwarp(self, rgbImage, uvimage):
        
        textures = []
        device = rgbImage.device
        
        for cnt in range(rgbImage.shape[0]):
            uv = uvimage[cnt,:,:,:3]
            rgb = rgbImage[cnt].permute(1,2,0)
            
            mask = uv.sum(dim = 2) > 0
            rgb  = rgb[mask,:]
            uvCd = (uv[mask][:,:2]*224).round().long()
            
            # better to use inverse mapping
            tex  = torch.zeros_like(uv)
            tex[uvCd[:,1], uvCd[:,0]] = rgb
            
            textures.append(tex)
            
        return torch.stack(textures, dim=0).to(device)

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.encoder = resnet50(pretrained=True)

    def forward(self, img):
        return self.encoder(img)

class Regressor_pose(nn.Module):
    def __init__(self, model_file, init_pose, init_shape, init_cam, 
                 device='cuda'):
        super(Regressor_pose, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(model_file, device)
        
        self.projcam = Cam(smpl_obj=False)
        
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def projection(self, pred_joints, pred_camera):
        
        batch_size = pred_joints.shape[0]
        
        pred_cam_t = torch.stack(
            [pred_camera[:, 1],
             pred_camera[:, 2],
             2 * 1000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
        
        pred_keypoints_2d, _ = self.projcam(
            fx = 1000,
            fy = 1000,
            cx = 112,
            cy = 112,
            rotation = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
            translation = pred_cam_t[:,None,:],
            points = pred_joints, 
            visibilityOn = False)
        pred_keypoints_2d = torch.cat(pred_keypoints_2d, dim=0).reshape([batch_size, 24, 2])
        
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = (pred_keypoints_2d - 112) / (224. / 2.)
        
        return pred_keypoints_2d

    def forward(self, x, init_pose=None, init_shape=None, 
                init_cam=None, n_iter=3, J_regressor=None):
        
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_axisangle = rot6d_to_axisAngle(pred_pose).reshape(-1, 72)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 72).reshape(batch_size, 24, 3, 3)
        
        pred_vertices = self.smpl(
            pose = pred_axisangle,
            beta = pred_shape
            )
        pred_joints =  self.smpl.get_joints(pred_vertices)
        
        pred_keypoints_2d = self.projection(pred_joints, pred_cam)

        output = [{
            'theta'  : torch.cat([pred_cam, pred_axisangle, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output
