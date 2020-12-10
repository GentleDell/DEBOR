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


class lossfunc(nn.Module):
    def __init__(self, options, meshConnection, vertEdgeTable, device):
        '''
        Parameters
        ----------
        enable_vertLoss, enable_uvmapLoss, enable_normalLoss, enable_edgeLoss,
        enable_renderingLoss, enable_MS_DSSIM_Loss control the components of 
        loss.
            
        meshConnection : Tensor
            The triangles of the mesh, [B,C,3].
        vertEdgeTable : Tensor
            SMPL vertex-edges correspondence table, [20664, 2].
        device : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''        
        super(lossfunc, self).__init__()
        
        self.device = device
        
        self.use_vertLoss   = options.enable_vertLoss           # vertices
        self.use_edgeLoss   = options.enable_edgeLoss
        self.use_normalLoss = options.enable_normalLoss
        self.use_renderLoss = options.enable_renderingLoss
        self.use_MS_DSSIMvt = options.enable_MS_DSSIM_Loss_verts
        self.use_uvmapLoss  = options.enable_uvmapLoss          # UV Maps
        self.use_MS_DSSIMuv = options.enable_MS_DSSIM_Loss_uv
        
        self.weight_vertLoss   = options.weight_disps           # vertices
        self.weight_edgeLoss   = options.weight_edges
        self.weight_normalLossvt = options.weight_vertex_normal
        self.weight_normalLosstr = options.weight_triangle_normal
        self.weight_renderLoss = options.weight_rendering
        self.weight_MS_DSSIMvt = options.weight_MS_DSSIM_verts
        self.weight_uvmapLoss  = options.weight_uvmap           # UV Maps  
        self.weight_MS_DSSIMuv = options.weight_MS_DSSIM_uvmap
        
        # use L1 loss for vertices loss
        if self.use_vertLoss:
            self.vertLoss= nn.L1Loss().to(self.device)
        
        # use L1 as uvMap loss
        if self.use_uvmapLoss:
            self.uvMapLoss= nn.L1Loss().to(self.device)
                
        # Multiple scale similarity calculator for vertices after rendering
        if self.use_MS_DSSIMvt and self.use_renderLoss:
            self.MS_SSIMvt_cls = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, 
                                       size_average=True, channel=3)
            
        if self.use_MS_DSSIMuv:
            self.MS_SSIMuv_cls = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, 
                                       size_average=True, channel=3)

        # use normal and edge loss
        self.triplets, self.vpe = None, None
        if self.use_normalLoss:
            self.triplets = meshConnection
        if self.use_edgeLoss:
            self.vpe = vertEdgeTable

    def normal_loss(self, pred_verts, gt_verts):
        '''
        Compute the normal differences of triangles(facet) of the predicted verts
        and GT verts, through the pytorch3d package.
    
        Parameters
        ----------
        pred_verts : Tensor
            Predicted displacements + body vertices, [B,V,3].
        gt_verts : Tensor
            GT displacements + body vertices, [B,V,3].

        Returns
        -------
        loss_face_normal : Tensor
            Normal difference.
    
        '''
        # prepare and cast data type
        batchSize = pred_verts.shape[0]
        num_verts = pred_verts.shape[1]
        num_faces = self.triplets.shape[1]
        
        # create pytorch3d tri mesh
        pred_mesh = Meshes(verts=pred_verts, faces=self.triplets)
        gt_mesh   = Meshes(verts=gt_verts, faces=self.triplets)
        
        # compute vertices and faces normals
        pred_mesh_verts_norm = pred_mesh.verts_normals_packed()
        pred_mesh_faces_norm = pred_mesh.faces_normals_packed()
        
        gt_mesh_verts_norm   = gt_mesh.verts_normals_packed()
        gt_mesh_faces_norm   = gt_mesh.faces_normals_packed()
        
        # Debug:
        #     residual error comparing to open3d is:
        #     1e-7 for verts, 3e-4 or faces, on average.
        # o3dMesh = o3d.geometry.TriangleMesh()
        # o3dMesh.vertices = o3d.utility.Vector3dVector(gt_verts[0].detach().cpu())
        # o3dMesh.triangles= o3d.utility.Vector3iVector(self.triplets[0].detach().cpu()) 
        # o3dMesh.compute_vertex_normals()     
        # o3dMesh.compute_triangle_normals()   
        # pred_vn = np.array(o3dMesh.vertex_normals)
        # pred_fn = np.array(o3dMesh.triangle_normals)
        # o3dMesh.vertex_colors = o3d.utility.Vector3dVector(pred_fn[-6890:] - 
        #                                     gt_mesh_faces_norm[13776-6890:13776]
        #                                    .detach().cpu().numpy())
        # o3d.visualization.draw_geometries([o3dMesh])
        
        # triangle verts' normal loss
        loss_vert_normal = torch.sum( 
            1 - torch.sum( gt_mesh_verts_norm*pred_mesh_verts_norm, dim=1
                         ).abs()) / (batchSize * num_verts)
        
        # triangle faces' normal loss
        loss_face_normal = torch.sum( 
            1 - torch.sum( gt_mesh_faces_norm*pred_mesh_faces_norm, dim=1
                         ).abs()) / (batchSize * num_faces)
        return loss_vert_normal, loss_face_normal

    def edge_loss(self, pred_verts, gt_verts):
        '''
        Calculate edge loss measured by difference in the length. 
        
        Adapted from CAPE.
        
        args:
            pred_verts: prediction, [batch size, num_verts (6890), 3]
            gt_verts: ground truth, [batch size, num_verts (6890), 3]
        returns:
            loss_edge: L2 norm loss of edge difference.
        '''
        # get vectors of all edges, have size (batch_size, 20664, 3)
        edges_pred = pred_verts[:, self.vpe[:,0]] - pred_verts[:, self.vpe[:,1]] 
        edges_gt   = gt_verts[:, self.vpe[:,0]] - gt_verts[:, self.vpe[:,1]] 
        
        edge_diff = edges_pred - edges_gt
        loss_edge = edge_diff.norm(dim = 2).mean(dim = 1).sum()/pred_verts.shape[0]
    
        return loss_edge

    def MS_DSSIM(self, pred_image, target_image, on = 'vertices'):
        '''
        Multi-Scale DisSImilarity Metrics.
        
        This loss keeps high frequency information of image and has been 
        proved to be helpful for image restoration task and recovering 
        detials of 3D shapes.
        
        The implementation is based on the git repo:
            https://github.com/VainF/pytorch-msssim/
        
        '''
        
        # denormalize the image to 0-1
        raise_predic_image  = pred_image - pred_image.min()
        denorm_predic_image = raise_predic_image/raise_predic_image.max()
        
        # the formula used in tex2shape
        if on == 'vertices':
            loss_ms_dssim = 1 - self.MS_SSIMvt_cls(
                denorm_predic_image.permute(0,3,1,2), 
                target_image.permute(0,3,1,2))
            
        elif on == 'uvMaps':
            loss_ms_dssim = 1 - self.MS_SSIMuv_cls(
                denorm_predic_image.permute(0,3,1,2), 
                target_image.permute(0,3,1,2))
            
        return loss_ms_dssim
    
    def rendering_loss(self):
        
        raise NotImplementedError('To be implemented.')


    def forward(self, predVerts = None, GTVerts = None, 
                      predUVMaps = None, GTUVMaps = None):
        
        # predefine losses
        outloss = {'loss_vertices': torch.Tensor([0]),
                   'loss_egdes': torch.Tensor([0]),
                   'loss_verts_normal': torch.Tensor([0]),
                   'loss_faces_normal': torch.Tensor([0]),
                   'loss_uvMap': torch.Tensor([0]),
                   'loss_render': torch.Tensor([0]),
                   'loss_MS_DSSIMvt': torch.Tensor([0]),
                   'loss_MS_DSSIMuv': torch.Tensor([0])
                   }
        for key, val in outloss.items():
            outloss[key] = val.to(self.device)
        
        # loss for displacements
        if predVerts is not None and GTVerts is not None:
            if self.use_vertLoss:
                outloss['loss_vertices'] = self.vertLoss(predVerts, GTVerts)
                
            if self.use_edgeLoss:
                outloss['loss_egdes'] = self.edge_loss(predVerts, GTVerts)
                
            if self.use_normalLoss:
                outloss['loss_verts_normal'], outloss['loss_faces_normal'] = \
                    self.normal_loss(predVerts, GTVerts)
        
        # rendering loss if enabled
        if self.use_renderLoss:
            outloss['loss_render'], predImg, GTImg= self.loss_render()
            if self.use_MS_DSSIMvt:
                outloss['loss_MS_DSSIMvt'] = self.MS_DSSIM(predImg, GTImg, 'vertices')
        
        # loss for uvmaps
        if predUVMaps is not None and GTUVMaps is not None:
            if self.use_uvmapLoss:
                outloss['loss_uvMap'] = self.uvMapLoss(predUVMaps, GTUVMaps)         
            if self.use_MS_DSSIMuv:
                outloss['loss_MS_DSSIMuv'] = self.MS_DSSIM(predUVMaps, GTUVMaps, 'uvMaps')
                            
        outloss['loss'] = \
            self.weight_vertLoss * outloss['loss_vertices'] + \
            self.weight_edgeLoss * outloss['loss_egdes'] +\
            self.weight_normalLossvt * outloss['loss_verts_normal'] +\
            self.weight_normalLosstr * outloss['loss_faces_normal'] +\
            self.weight_renderLoss * outloss['loss_render'] +\
            self.weight_MS_DSSIMvt * outloss['loss_MS_DSSIMvt'] +\
            self.weight_uvmapLoss * outloss['loss_uvMap'] +\
            self.weight_MS_DSSIMuv* outloss['loss_MS_DSSIMuv']
        
        return outloss
                   